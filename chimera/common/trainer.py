from __future__ import annotations

from pathlib import Path
import math
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from .device import RuntimeContext, move_to_device, set_seed
from .metrics import regression_metrics, classification_metrics, summarize_fold_metrics
from .config import save_json


def _validate_protocol_config(cfg: dict[str, Any]) -> None:
    split_cfg = cfg.get("split", {})
    if split_cfg.get("inner_strategy") != "rotating_outer_fold":
        raise ValueError("The official 8/1/1 protocol requires split.inner_strategy='rotating_outer_fold'.")
    if int(split_cfg.get("num_folds", 0)) != 10:
        raise ValueError("The official 8/1/1 protocol requires exactly 10 outer folds.")
    if _selection_source(cfg) != "validation":
        raise ValueError("training.selection_source must be 'validation'; outer-test selection is forbidden.")


def _label_of(sample) -> int:
    return int(sample[0].y.view(-1)[0].item())


def _target_of(sample) -> float:
    return float(sample[0].y.view(-1)[0].item())


def _make_loader(dataset, batch_size: int, shuffle: bool, ctx: RuntimeContext, drop_last: bool = False, seed: int = 42):
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last, seed=seed) if ctx.is_distributed and shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=(shuffle and sampler is None), sampler=sampler, drop_last=drop_last)


def _pad_training_dataset(dataset, batch_size: int, seed: int):
    items = list(dataset)
    remainder = len(items) % batch_size
    if remainder == 0 or not items:
        return items, 0
    padding_count = batch_size - remainder
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(items)).tolist()
    padding = [items[order[index % len(order)]] for index in range(padding_count)]
    return items + padding, padding_count


def _regression_loss(output, y, loss_name: str):
    mse = F.mse_loss(output.float(), y.view_as(output).float())
    if loss_name in {"rmse", "batch_rmse"}:
        return torch.sqrt(mse + 1e-12)
    if loss_name == "mse":
        return mse
    raise ValueError(f"Unsupported training.regression_loss: {loss_name}")


def _forward(model, batch, task_type: str):
    out = model(batch)
    if task_type == "regression":
        pred, rep = out[0], out[1]
        return pred, rep, pred
    log_probs, rep = out[0], out[1]
    raw_logits = out[-1] if isinstance(out, (tuple, list)) and hasattr(out[-1], "shape") and out[-1].ndim == 2 else log_probs
    return log_probs, rep, raw_logits


def _supervised_loss(output, batch, task_type: str, regression_loss: str = "mse"):
    if task_type == "regression":
        y = batch[0].y.view_as(output).float()
        return _regression_loss(output, y, regression_loss)
    y = batch[0].y.view(-1).long()
    return F.nll_loss(output, y)


@torch.no_grad()
def evaluate(model, loader, task_type: str, device: torch.device, regression_loss: str = "mse") -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    y_true: list[Any] = []
    y_pred: list[Any] = []
    y_score: list[Any] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        output, _, _ = _forward(model, batch, task_type)
        loss = _supervised_loss(output, batch, task_type, regression_loss)
        losses.append(float(loss.detach().cpu()))
        if task_type == "regression":
            y = batch[0].y.view_as(output).detach().cpu().numpy().reshape(-1)
            pred = output.detach().cpu().numpy().reshape(-1)
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
        else:
            y = batch[0].y.view(-1).detach().cpu().numpy()
            probs = torch.exp(output).detach().cpu().numpy()
            pred = probs.argmax(axis=1)
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
            y_score.extend(probs.tolist())
    metrics = {"loss": float(np.mean(losses)) if losses else None}
    if task_type == "regression":
        metrics.update(regression_metrics(y_true, y_pred))
        if regression_loss in {"rmse", "batch_rmse"}:
            metrics["loss"] = metrics["rmse"]
    else:
        metrics.update(classification_metrics(y_true, y_pred, y_score))
    return metrics


def _split_train_val(train_indices, dataset, task_type: str, val_ratio: float, seed: int, groups=None):
    if val_ratio <= 0:
        return list(train_indices), []
    if groups is not None:
        group_values = np.asarray(groups, dtype=object)[list(train_indices)]
        if len(np.unique(group_values)) < 2:
            raise ValueError("At least two component groups are required for inner group validation.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        inner_pos, val_pos = next(splitter.split(np.zeros(len(train_indices)), groups=group_values))
        train_array = np.asarray(train_indices, dtype=int)
        return train_array[inner_pos].tolist(), train_array[val_pos].tolist()
    stratify = None
    if task_type == "classification":
        labels = np.array([_label_of(dataset[i]) for i in train_indices])
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) > 1 and counts.min() >= 2:
            stratify = labels
    train_idx, val_idx = train_test_split(list(train_indices), test_size=val_ratio, random_state=seed, stratify=stratify)
    return list(train_idx), list(val_idx)


def _make_outer_splits(dataset, task_type: str, cfg: dict[str, Any], split_groups=None):
    split_cfg = cfg.get("split", {})
    mode = split_cfg.get("mode", "kfold")
    seed = int(cfg.get("seed", 42))
    indices = np.arange(len(dataset))
    if mode == "group_kfold":
        group_key = str(split_cfg.get("group_key", "")).lower()
        if not split_groups or group_key not in split_groups:
            raise ValueError(f"Missing component groups for split.group_key={group_key!r}")
        groups = np.asarray(split_groups[group_key], dtype=object)
        if len(groups) != len(dataset):
            raise ValueError(f"Group vector length {len(groups)} does not match dataset length {len(dataset)}")
        n_splits = int(split_cfg.get("num_folds", 10))
        if len(np.unique(groups)) < n_splits:
            raise ValueError(f"Component split {group_key!r} has fewer than {n_splits} unique groups")
        splitter = GroupKFold(n_splits=n_splits)
        yield from splitter.split(indices, groups=groups)
    elif mode == "kfold":
        n_splits = int(split_cfg.get("num_folds", 10))
        if task_type == "classification":
            labels = np.array([_label_of(s) for s in dataset])
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) > 1 and counts.min() >= n_splits:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                yield from splitter.split(indices, labels)
                return
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(indices)
    elif mode == "repeated_holdout":
        n_repeats = int(split_cfg.get("num_repeats", 10))
        test_size = float(split_cfg.get("test_size", 0.1))
        if task_type == "classification":
            labels = np.array([_label_of(s) for s in dataset])
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) > 1 and counts.min() >= 2:
                splitter = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=seed)
                yield from splitter.split(indices, labels)
                return
        splitter = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=seed)
        yield from splitter.split(indices)
    else:
        raise ValueError(f"Unsupported split mode: {mode}")


def _select_metric(task_type: str, cfg: dict[str, Any]) -> tuple[str, bool]:
    metric = cfg.get("training", {}).get("monitor")
    if metric:
        maximize = bool(cfg.get("training", {}).get("monitor_maximize", metric in {"auc", "f1_macro", "r2", "corr", "accuracy"}))
        return metric, maximize
    return ("r2", True) if task_type == "regression" else ("auc", True)


def _selection_source(cfg: dict[str, Any]) -> str:
    train_cfg = cfg.get("training", {})
    value = str(train_cfg.get("selection_source", "validation")).lower()
    if value in {"test", "test_fold", "original", "github"}:
        return "test"
    return "validation"


def _better(value, best, maximize: bool) -> bool:
    if value is None:
        return False
    if best is None:
        return True
    return value > best if maximize else value < best


def train_one_fold(model_class, dataset, train_indices, validation_indices, test_indices, fold_id: int, validation_fold_id: int, cfg: dict[str, Any], task_type: str, out_dir: Path, ctx: RuntimeContext, all_groups=None) -> dict[str, Any]:
    train_cfg = cfg.get("training", {})
    split_cfg = cfg.get("split", {})
    base_seed = int(cfg.get("seed", 42))
    seed = base_seed
    set_seed(seed)

    val_indices = list(validation_indices)
    train_indices = list(train_indices)
    regression_loss = str(train_cfg.get("regression_loss", "mse")).lower()

    assert set(train_indices).isdisjoint(test_indices)
    assert set(val_indices).isdisjoint(test_indices)
    assert set(train_indices).isdisjoint(val_indices)
    train_groups = sorted({str(all_groups[i]) for i in train_indices}) if all_groups is not None else []
    val_groups = sorted({str(all_groups[i]) for i in val_indices}) if all_groups is not None else []
    test_groups = sorted({str(all_groups[i]) for i in test_indices}) if all_groups is not None else []
    if all_groups is not None:
        assert set(train_groups).isdisjoint(val_groups)
        assert set(train_groups).isdisjoint(test_groups)
        assert set(val_groups).isdisjoint(test_groups)

    train_set = [dataset[i] for i in train_indices]
    if not val_indices:
        raise RuntimeError("Inner validation split is empty; refusing to fall back to training or test data.")
    val_set = [dataset[i] for i in val_indices]
    per_device_batch_size = int(train_cfg.get("batch_size", 32))
    if per_device_batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    batch_size = per_device_batch_size
    global_batch_size = per_device_batch_size * int(ctx.world_size)
    pad_train_to_full_batch = bool(train_cfg.get("pad_train_to_full_batch", False))
    train_loader_set, train_padding_count = (
        _pad_training_dataset(train_set, batch_size, seed)
        if pad_train_to_full_batch
        else (train_set, 0)
    )
    train_loader = _make_loader(train_loader_set, batch_size, True, ctx, seed=seed)
    val_loader = _make_loader(val_set, batch_size, False, ctx)

    model_cfg = cfg.get("model", {})
    hidden = int(model_cfg.get("hidden", 32))
    num_layers = int(model_cfg.get("gin_layers", model_cfg.get("num_layers", 5)))
    os.environ["CHIMERA_CHIRALITY_ALPHA"] = str(float(model_cfg.get("chirality_alpha", 2.0)))
    try:
        model = model_class(train_set, num_layers, hidden, cfg=cfg)
    except TypeError:
        model = model_class(train_set, num_layers, hidden)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    model = model.to(ctx.device)
    if ctx.is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                                                                              
                                                                              
                                                                               
                                                                              
                                                                            
        ablation_cfg = cfg.get("ablation", {})
        default_find_unused = any(bool(v) for v in ablation_cfg.values())
        ddp_find_unused = bool(cfg.get("distributed", {}).get("find_unused_parameters", default_find_unused))
        model = DDP(
            model,
            device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
            find_unused_parameters=ddp_find_unused,
        )

    lr = float(train_cfg.get("lr", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    selection_source = _selection_source(cfg)
    monitor_name, maximize = _select_metric(task_type, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg.get("lr_factor", 0.1)),
        patience=int(train_cfg.get("lr_patience", 10)),
        min_lr=float(train_cfg.get("min_lr", 1e-7)),
    )

    best_metric = None
    best_state = None
    best_epoch = None
    history: list[dict[str, Any]] = []
    max_epochs = int(train_cfg.get("epochs", 150))
    fold_dir = out_dir / f"fold_{fold_id:02d}"
    if ctx.is_main:
        fold_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        model.train()
        if ctx.is_distributed and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        epoch_losses = []
        for batch in train_loader:
            batch = move_to_device(batch, ctx.device)
            optimizer.zero_grad(set_to_none=True)
            output, _, _ = _forward(model, batch, task_type)
            loss = _supervised_loss(output, batch, task_type, regression_loss=regression_loss)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        if ctx.is_main:
            eval_model = model.module if isinstance(model, DDP) else model
            val_metrics = evaluate(eval_model, val_loader, task_type, ctx.device, regression_loss=regression_loss)
            selection_value = val_metrics.get(monitor_name)
        else:
            val_metrics = {}
            selection_value = None

        scheduler_value = val_metrics.get("loss") if ctx.is_main else None
        if ctx.is_distributed:
            scheduler_tensor = torch.tensor(
                [float("nan") if scheduler_value is None else float(scheduler_value)],
                dtype=torch.float64,
                device=ctx.device,
            )
            torch.distributed.broadcast(scheduler_tensor, src=0)
            scheduler_value = float(scheduler_tensor.item())
        if scheduler_value is None or not math.isfinite(float(scheduler_value)):
            raise RuntimeError("Validation loss is unavailable; refusing to drive the LR scheduler from test data.")
        scheduler.step(float(scheduler_value))
        if ctx.is_main:
            row = {
                "fold": fold_id,
                "epoch": epoch,
                "selection_source": selection_source,
                "monitor": monitor_name,
                "lr": epoch_lr,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)
            if _better(selection_value, best_metric, maximize):
                best_metric = selection_value
                best_epoch = epoch
                eval_model = model.module if isinstance(model, DDP) else model
                best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
                torch.save(best_state, fold_dir / "best_model.pt")

    if not ctx.is_main:
        return {"fold": fold_id}

    eval_model = model.module if isinstance(model, DDP) else model
    if best_state is not None:
        eval_model.load_state_dict(best_state)
                                                                                
    test_set = [dataset[i] for i in test_indices]
    test_loader = _make_loader(test_set, batch_size, False, ctx)
    test_metrics = evaluate(eval_model, test_loader, task_type, ctx.device, regression_loss=regression_loss)
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    torch.save(best_state if best_state is not None else eval_model.state_dict(), fold_dir / "best_model.pt")
    save_json({
        "fold": fold_id,
        "validation_fold": validation_fold_id,
        "base_seed": base_seed,
        "fold_seed": seed,
        "sampler_seed": seed,
        "train_indices": list(map(int, train_indices)),
        "validation_indices": list(map(int, val_indices)),
        "selection_source": selection_source,
        "selection_monitor": monitor_name,
        "selection_maximize": maximize,
        "best_epoch": best_epoch,
        "best_validation_metric": best_metric,
        "test_indices": list(map(int, test_indices)),
        "group_key": split_cfg.get("group_key") if all_groups is not None else None,
        "train_groups": train_groups,
        "validation_groups": val_groups,
        "test_groups": test_groups,
        "pad_train_to_full_batch": pad_train_to_full_batch,
        "train_padding_count": int(train_padding_count),
        "num_train_loader_samples": int(len(train_loader_set)),
        "global_batch_size": int(global_batch_size),
        "per_device_batch_size": int(per_device_batch_size),
        "world_size": int(ctx.world_size),
        "outer_test_label_evaluations": 1,
        "epochs_completed": max_epochs,
        "lr_scheduler": "ReduceLROnPlateau(validation_loss)",
        "protocol_note": "Fixed 8/1/1 folds. All 200 epochs are trained; validation controls LR decay and best-checkpoint selection. Outer-test labels are read once after checkpoint selection.",
    }, fold_dir / "split_indices.json")
    row = {"fold": fold_id, "validation_fold": validation_fold_id, "base_seed": base_seed, "fold_seed": seed, "best_epoch": best_epoch, "best_validation_metric": best_metric, "num_train": len(train_indices), "num_val": len(val_indices), "selection_source": selection_source, "selection_monitor": monitor_name, "num_test": len(test_indices), **test_metrics}
    return row


def train_cross_validate(model_class, dataset, cfg: dict[str, Any], task_type: str, out_dir: str | Path, ctx: RuntimeContext, split_groups=None) -> None:
    _validate_protocol_config(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if ctx.is_main:
        save_json(cfg, out_dir / "config_used.json")
        save_json({
            "protocol": "leakage_free_nested_validation",
            "base_seed": int(cfg.get("seed", 42)),
            "fold_seed_rule": "every fold and every stochastic component uses the fixed base seed",
            "deterministic_algorithms": True,
            "test_usage": "outer-test data are excluded from training and validation and are accessed once after best-checkpoint selection",
            "validation_usage": "one complete rotating fold exclusively controls LR scheduling and best-checkpoint selection; training always runs all 200 epochs",
            "ablation": cfg.get("ablation", {}),
        }, out_dir / "protocol.json")
    split_cfg = cfg.get("split", {})
    group_key = str(split_cfg.get("group_key", "")).lower() if split_cfg.get("mode") == "group_kfold" else None
    all_groups = split_groups.get(group_key) if group_key and split_groups else None
    outer_splits = [(list(map(int, tr)), list(map(int, te))) for tr, te in _make_outer_splits(dataset, task_type, cfg, split_groups=split_groups)]
    selected_folds = split_cfg.get("selected_folds")
    if selected_folds is not None:
        selected = {int(value) for value in selected_folds}
        indexed_splits = [(fold_id, split) for fold_id, split in enumerate(outer_splits, 1) if fold_id in selected]
    else:
        indexed_splits = list(enumerate(outer_splits, 1))
    if ctx.is_main:
        save_json({
            "base_seed": int(cfg.get("seed", 42)),
            "split_mode": split_cfg.get("mode", "kfold"),
            "group_key": group_key,
            "folds": [
                {"fold": i, "outer_train_indices": tr, "outer_test_indices": te,
                 "outer_train_groups": sorted({str(all_groups[j]) for j in tr}) if all_groups is not None else [],
                 "outer_test_groups": sorted({str(all_groups[j]) for j in te}) if all_groups is not None else []}
                for i, (tr, te) in indexed_splits
            ],
        }, out_dir / "outer_folds.json")
    rows: list[dict[str, Any]] = []
    split_by_fold = {fold_id: split for fold_id, split in enumerate(outer_splits, 1)}
    for fold_id, (outer_train_idx, test_idx) in indexed_splits:
        validation_fold_id = fold_id % len(outer_splits) + 1
        val_idx = split_by_fold[validation_fold_id][1]
        val_set = set(val_idx)
        train_idx = [index for index in outer_train_idx if index not in val_set]
        if len(train_idx) + len(val_idx) != len(outer_train_idx):
            raise RuntimeError("The rotating validation fold is not an exact partition of the outer training folds.")
        row = train_one_fold(model_class, dataset, train_idx, val_idx, test_idx, fold_id, validation_fold_id, cfg, task_type, out_dir, ctx, all_groups=all_groups)
        if ctx.is_main:
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "fold_metrics.csv", index=False)
            summary = summarize_fold_metrics(rows)
            save_json(summary, out_dir / "summary_metrics.json")
            pd.DataFrame([summary]).to_csv(out_dir / "summary_metrics.csv", index=False)
    if ctx.is_distributed:
        torch.distributed.barrier()
