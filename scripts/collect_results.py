#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASETS = {
    "DataA": ("dataA_classification", "dataA_regression"),
    "DataB": ("dataB_classification", "dataB_regression"),
}
OUTPUT_COLUMNS = [
    "dataset",
    "model",
    "num_folds",
    "recall_macro_mean",
    "recall_macro_std",
    "f1_macro_mean",
    "f1_macro_std",
    "auc_mean",
    "auc_std",
    "r2_mean",
    "r2_std",
    "rmse_mean",
    "rmse_std",
]


def load_folds(root: Path, task: str, metrics: tuple[str, ...]) -> pd.DataFrame:
    path = root / task / "fold_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing result: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in ("fold", *metrics) if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing columns: {missing}")
    if len(frame) != 10 or set(frame["fold"].astype(int)) != set(range(1, 11)):
        raise RuntimeError(f"{path} must contain exactly folds 1..10")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the four CHIMERA tasks into one table")
    parser.add_argument("--root", type=Path, default=Path("output"), help="Directory containing four task results")
    parser.add_argument("--output", type=Path, default=Path("output/chimera_summary.csv"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
        help="Datasets to collect (default: DataA DataB)",
    )
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for dataset in args.datasets:
        classification_task, regression_task = DATASETS[dataset]
        classification = load_folds(
            args.root, classification_task, ("recall_macro", "f1_macro", "auc")
        )
        regression = load_folds(args.root, regression_task, ("r2", "rmse"))
        row: dict[str, float | int | str] = {
            "dataset": dataset,
            "model": "CHIMERA",
            "num_folds": 10,
        }
        for metric in ("recall_macro", "f1_macro", "auc"):
            row[f"{metric}_mean"] = classification[metric].mean()
            row[f"{metric}_std"] = classification[metric].std(ddof=1)
        for metric in ("r2", "rmse"):
            row[f"{metric}_mean"] = regression[metric].mean()
            row[f"{metric}_std"] = regression[metric].std(ddof=1)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
