#!/usr/bin/env python3
"""Regenerate manuscript Tables 4--7 from saved fold-level metrics only.

The script does not load datasets, configurations, checkpoints, or predictions.
Every input is a ``fold_metrics.csv`` under ``results/`` and must contain exactly
the ten outer-test folds numbered 1 through 10. Reported standard deviations use
the sample definition (pandas ``std(ddof=1)``), matching the manuscript.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = (
    "dataA_classification",
    "dataA_regression",
    "dataB_classification",
    "dataB_regression",
)
COMPONENTS = (
    ("catalyst", "Catalyst"),
    ("product", "Product"),
    ("reactant1", "Reactant1"),
    ("reactant2", "Reactant2"),
)
BASELINES = (
    "DESC_SVM",
    "ECFP_GB",
    "GCN",
    "GIN",
    "GAT",
    "CHIENN",
    "RXNFP",
    "MFP_MLP",
)
ABLATIONS = (
    ("no_interaction", "w/o Interaction"),
    ("no_motif", "w/o Motifs"),
    ("no_fingerprint", "w/o Fingerprint"),
    ("no_graph", "w/o Atom-level Graph"),
)
METRICS = {
    "classification": ("recall_macro", "f1_macro", "auc"),
    "regression": ("r2", "rmse"),
}
DISPLAY = {
    "recall_macro": "REC",
    "f1_macro": "F1",
    "auc": "AUC",
    "r2": "R2",
    "rmse": "RMSE",
}

# Student-t 97.5th percentile with df=9. All input files are fixed at ten folds.
T_CRITICAL_DF9 = 2.2621571628540993


def load_folds(path: Path, task: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing fold metrics: {path}")
    frame = pd.read_csv(path)
    metrics = METRICS[task]
    missing = [column for column in ("fold", *metrics) if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing columns: {missing}")
    frame = frame.loc[:, ["fold", *metrics]].copy()
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame = frame.sort_values("fold").reset_index(drop=True)
    if frame["fold"].tolist() != list(range(1, 11)):
        raise RuntimeError(f"{path} must contain each fold from 1 through 10 exactly once")
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
        if not np.isfinite(frame[metric].to_numpy(dtype=float)).all():
            raise RuntimeError(f"{path} contains a non-finite {metric} value")
    return frame


def summarize(frame: pd.DataFrame, metric: str) -> str:
    return f"{frame[metric].mean():.3f} ± {frame[metric].std(ddof=1):.3f}"


def paired_paths(results: Path, dataset: str, task: str) -> tuple[Path, Path]:
    task_id = f"data{dataset}_{task}"
    return (
        results / "CHIMERA" / task_id / "fold_metrics.csv",
        results / "BASELINE" / "MFP_MLP" / task_id / "fold_metrics.csv",
    )


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def exact_wilcoxon_two_sided(differences: np.ndarray) -> float:
    differences = differences[~np.isclose(differences, 0.0)]
    if not len(differences):
        return 1.0
    ranks = average_ranks(np.abs(differences))
    observed = float(ranks[differences > 0].sum())
    center = float(ranks.sum() / 2.0)
    extreme = 0
    total = 2 ** len(ranks)
    for signs in itertools.product((0, 1), repeat=len(ranks)):
        value = float(ranks[np.asarray(signs, dtype=bool)].sum())
        if abs(value - center) >= abs(observed - center) - 1e-12:
            extreme += 1
    return extreme / total


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p) - rank) * p[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def make_table4(results: Path) -> pd.DataFrame:
    rows = []
    for dataset in ("A", "B"):
        for directory, label in COMPONENTS:
            root = results / "COMPONENT_SPLIT" / directory
            cls = load_folds(root / f"data{dataset}_classification/fold_metrics.csv", "classification")
            reg = load_folds(root / f"data{dataset}_regression/fold_metrics.csv", "regression")
            rows.append([
                f"Data {dataset}", label,
                summarize(cls, "recall_macro"), summarize(cls, "f1_macro"), summarize(cls, "auc"),
                summarize(reg, "r2"), summarize(reg, "rmse"),
            ])
    return pd.DataFrame(rows, columns=["Dataset", "Split", "REC", "F1", "AUC", "R2", "RMSE"])


def make_table5(results: Path) -> pd.DataFrame:
    rows = []
    for dataset in ("A", "B"):
        for model in (*BASELINES, "CHIMERA"):
            root = results / ("CHIMERA" if model == "CHIMERA" else f"BASELINE/{model}")
            cls = load_folds(root / f"data{dataset}_classification/fold_metrics.csv", "classification")
            reg = load_folds(root / f"data{dataset}_regression/fold_metrics.csv", "regression")
            rows.append([
                f"Data {dataset}", model,
                summarize(cls, "recall_macro"), summarize(cls, "f1_macro"), summarize(cls, "auc"),
                summarize(reg, "r2"), summarize(reg, "rmse"),
            ])
    return pd.DataFrame(rows, columns=["Dataset", "Model", "REC", "F1", "AUC", "R2", "RMSE"])


def make_table6(results: Path) -> pd.DataFrame:
    comparisons = (
        ("A", "classification", "auc", True),
        ("A", "regression", "r2", True),
        ("A", "regression", "rmse", False),
        ("B", "classification", "auc", True),
        ("B", "regression", "r2", True),
        ("B", "regression", "rmse", False),
    )
    records = []
    p_values = []
    for dataset, task, metric, higher_is_better in comparisons:
        chimera_path, mfp_path = paired_paths(results, dataset, task)
        chimera = load_folds(chimera_path, task)[metric].to_numpy(dtype=float)
        mfp = load_folds(mfp_path, task)[metric].to_numpy(dtype=float)
        difference = chimera - mfp
        direction = 1.0 if higher_is_better else -1.0
        advantage = direction * difference
        standard_error = float(difference.std(ddof=1) / math.sqrt(len(difference)))
        raw_low = float(difference.mean() - T_CRITICAL_DF9 * standard_error)
        raw_high = float(difference.mean() + T_CRITICAL_DF9 * standard_error)
        ci_low, ci_high = sorted((direction * raw_low, direction * raw_high))
        p_value = exact_wilcoxon_two_sided(difference)
        p_values.append(p_value)
        records.append({
            "Dataset": f"Data {dataset}",
            "Task": task.capitalize(),
            "Metric": DISPLAY[metric],
            "CHIMERA": f"{chimera.mean():.3f} ± {chimera.std(ddof=1):.3f}",
            "MFP_MLP": f"{mfp.mean():.3f} ± {mfp.std(ddof=1):.3f}",
            "CHIMERA advantage (95% paired t CI)": f"{advantage.mean():.3f} [{ci_low:.3f}, {ci_high:.3f}]",
            "Wins (CHIMERA/MFP_MLP)": f"{int((advantage > 0).sum())}/{int((advantage < 0).sum())}",
            "Wilcoxon p (two-sided)": f"{p_value:.3f}",
        })
    adjusted = holm_adjust(p_values)
    for record, value in zip(records, adjusted):
        record["Holm-adjusted p"] = f"{value:.3f}"
    return pd.DataFrame(records)


def make_table7(results: Path) -> pd.DataFrame:
    rows = []
    for dataset in ("A", "B"):
        variants = [(results / "CHIMERA", "CHIMERA (full)")]
        variants.extend((results / "ABLATION" / directory, label) for directory, label in ABLATIONS)
        for root, label in variants:
            cls = load_folds(root / f"data{dataset}_classification/fold_metrics.csv", "classification")
            reg = load_folds(root / f"data{dataset}_regression/fold_metrics.csv", "regression")
            rows.append([
                f"Data {dataset}", label,
                summarize(cls, "recall_macro"), summarize(cls, "f1_macro"), summarize(cls, "auc"),
                summarize(reg, "r2"), summarize(reg, "rmse"),
            ])
    return pd.DataFrame(rows, columns=["Dataset", "Variant", "REC", "F1", "AUC", "R2", "RMSE"])


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project / "results",
        help="Results directory containing CHIMERA/COMPONENT_SPLIT/BASELINE/ABLATION",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Generated table directory (default: RESULTS_ROOT/TABLES)",
    )
    args = parser.parse_args()
    results = args.results_root.resolve()
    output = (args.output_dir or results / "TABLES").resolve()

    tables = {
        "Table4": make_table4(results),
        "Table5": make_table5(results),
        "Table6": make_table6(results),
        "Table7": make_table7(results),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = output / f"{name.lower()}.csv"
        table.to_csv(path, index=False)
        print(f"\n{name}\n{'=' * len(name)}")
        print(table.to_string(index=False))
        print(f"Saved: {path}")
    print("\nDONE: Generated Tables 4-7 from the saved fold-level metrics.")


if __name__ == "__main__":
    main()
