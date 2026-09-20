#!/usr/bin/env python3
"""Summarize original versus fixed-one-hot Data B fold metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = (
    "DESC_SVM",
    "ECFP_GB",
    "GCN",
    "GIN",
    "GAT",
    "CHIENN",
    "RXNFP",
    "MFP_MLP",
    "CHIMERA",
)
TASK_METRICS = {
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


def fold_path(results: Path, model: str, task: str, onehot: bool) -> Path:
    family = "CHIMERA" if model == "CHIMERA" else f"BASELINE/{model}"
    if onehot:
        return results / "CONDITION_ENCODING" / family / f"dataB_{task}" / "fold_metrics.csv"
    return results / family / f"dataB_{task}" / "fold_metrics.csv"


def load_folds(path: Path, task: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing fold metrics: {path}")
    metrics = TASK_METRICS[task]
    frame = pd.read_csv(path)
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


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=project / "results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: RESULTS_ROOT/CONDITION_ENCODING)",
    )
    args = parser.parse_args()
    results = args.results_root.resolve()
    output = (args.output_dir or results / "CONDITION_ENCODING").resolve()

    summary_rows: list[dict] = []
    difference_rows: list[dict] = []
    for model in MODELS:
        by_encoding: dict[str, dict[str, float]] = {}
        for encoding, onehot in (("Original scalar codes", False), ("Fixed one-hot", True)):
            values: dict[str, float] = {}
            summary: dict[str, float | int | str] = {
                "dataset": "DataB",
                "model": model,
                "encoding": encoding,
                "num_folds": 10,
            }
            for task, metrics in TASK_METRICS.items():
                folds = load_folds(fold_path(results, model, task, onehot), task)
                for metric in metrics:
                    mean = float(folds[metric].mean())
                    std = float(folds[metric].std(ddof=1))
                    values[metric] = mean
                    summary[f"{metric}_mean"] = mean
                    summary[f"{metric}_std"] = std
            by_encoding[encoding] = values
            summary_rows.append(summary)

        difference: dict[str, float | str] = {
            "dataset": "DataB",
            "model": model,
            "difference": "fixed_one_hot_minus_original_scalar_codes",
        }
        for metric in DISPLAY:
            delta = by_encoding["Fixed one-hot"][metric] - by_encoding["Original scalar codes"][metric]
            difference[f"{metric}_difference"] = delta
        difference_rows.append(difference)

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "condition_encoding_summary.csv"
    difference_path = output / "condition_encoding_difference.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(difference_rows).to_csv(difference_path, index=False)
    print(f"Saved: {summary_path}")
    print(f"Saved: {difference_path}")


if __name__ == "__main__":
    main()
