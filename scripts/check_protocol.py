#!/usr/bin/env python
from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Check split files for unintended overlap/leakage.")
    parser.add_argument("result_dir")
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    config = json.loads((result_dir / "config_used.json").read_text(encoding="utf-8"))
    metadata = json.loads((result_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    expected_samples = int(metadata["num_rows"])
    expected_seed = int(config.get("seed", 42))
    expected_epochs = int(config.get("training", {}).get("epochs", 0))
    errors = []
    for split_file in result_dir.glob("fold_*/split_indices.json"):
        obj = json.loads(split_file.read_text(encoding="utf-8"))
        test = set(obj.get("test_indices", []))
        train = set(obj.get("train_indices", []))
        validation = set(obj.get("validation_indices", []))
        if obj.get("base_seed") != expected_seed or obj.get("fold_seed") != expected_seed or obj.get("sampler_seed") != expected_seed:
            errors.append((str(split_file), "fixed_seed", (obj.get("base_seed"), obj.get("fold_seed"), obj.get("sampler_seed"))))
        if obj.get("selection_source") != "validation":
            errors.append((str(split_file), "selection_source", obj.get("selection_source")))
        if not obj.get("validation_indices"):
            errors.append((str(split_file), "validation_indices", "empty"))
        if obj.get("outer_test_label_evaluations") != 1:
            errors.append((str(split_file), "outer_test_label_evaluations", obj.get("outer_test_label_evaluations")))
        for key in ["train_indices", "validation_indices"]:
            overlap = test.intersection(obj.get(key, []))
            if overlap:
                errors.append((str(split_file), key, sorted(overlap)[:10]))
        if train & validation:
            errors.append((str(split_file), "train_validation_overlap", sorted(train & validation)[:10]))
        if len(train) + len(validation) + len(test) != len(train | validation | test):
            errors.append((str(split_file), "8/1/1_partition", "indices are not disjoint"))
        if len(train | validation | test) != expected_samples:
            errors.append((str(split_file), "8/1/1_partition", f"covers {len(train | validation | test)} of {expected_samples}"))
        if obj.get("epochs_completed") != expected_epochs:
            errors.append((str(split_file), "epochs_completed", obj.get("epochs_completed")))
        history_path = split_file.parent / "history.csv"
        if not history_path.exists() or len(pd.read_csv(history_path)) != expected_epochs:
            errors.append((str(split_file), "history_epochs", "incomplete"))
        if not (split_file.parent / "best_model.pt").exists():
            errors.append((str(split_file), "best_model.pt", "missing"))
        train_groups = set(obj.get("train_groups", []))
        val_groups = set(obj.get("validation_groups", []))
        test_groups = set(obj.get("test_groups", []))
        if obj.get("group_key"):
            if not train_groups or not val_groups or not test_groups:
                errors.append((str(split_file), "component_groups", "empty"))
            if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
                errors.append((str(split_file), "component_groups", "overlap"))
    if errors:
        print("Found unintended split overlap:")
        for e in errors:
            print(e)
        raise SystemExit(1)
    print(f"OK: fixed-seed 8/1/1 training, complete epoch histories, validation-selected checkpoints, and no test overlap in {result_dir}")


if __name__ == "__main__":
    main()
