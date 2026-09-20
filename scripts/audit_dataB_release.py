#!/usr/bin/env python3
"""Audit the released Data B preprocessing and condition encodings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chimera.common.conditions import DATA_B_CATEGORICAL_KEYS, build_data_b_conditions


CONFIG_NAMES = (
    "dataB_classification.yaml",
    "dataB_regression.yaml",
    "dataB_classification_onehot.yaml",
    "dataB_regression_onehot.yaml",
)
EXPECTED_MODELS = {
    "CHIMERA", "DESC_SVM", "ECFP_GB", "GCN", "GIN",
    "GAT", "CHIENN", "RXNFP", "MFP_MLP",
}


def load_config(name: str) -> dict:
    return yaml.safe_load((PROJECT_ROOT / "configs" / name).read_text(encoding="utf-8"))


def main() -> None:
    configs = {name: load_config(name) for name in CONFIG_NAMES}
    for name, config in configs.items():
        assert "drop_zero_ddg" not in config["data"], f"{name} still defines drop_zero_ddg"

    data_dir = PROJECT_ROOT / configs["dataB_classification.yaml"]["data"]["data_dir"]
    frame = pd.read_excel(data_dir / "Ir-Ni reaction.xlsx")
    ddg = pd.to_numeric(frame["ddG"], errors="coerce")
    ee = pd.to_numeric(frame["ee"], errors="coerce")
    labels = (ee.abs() >= 90).astype(int)

    assert len(frame) == 886
    assert int((ddg == 0).sum()) == 0 and int(ddg.isna().sum()) == 0
    assert int((ee == 0).sum()) == 0 and int(ee.isna().sum()) == 0
    assert labels.value_counts().to_dict() == {0: 455, 1: 431}

    matrices: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}
    for name, config in configs.items():
        matrices[name], metadata[name] = build_data_b_conditions(frame, data_dir, config)

    scalar_expected = frame[
        ["tem", "Time", *DATA_B_CATEGORICAL_KEYS]
    ].to_numpy(dtype=np.float32)
    scalar_cls = matrices["dataB_classification.yaml"]
    scalar_reg = matrices["dataB_regression.yaml"]
    onehot_cls = matrices["dataB_classification_onehot.yaml"]
    onehot_reg = matrices["dataB_regression_onehot.yaml"]
    assert scalar_cls.shape == (886, 7) and np.array_equal(scalar_cls, scalar_expected)
    assert np.array_equal(scalar_cls, scalar_reg)
    assert onehot_cls.shape == (886, 155) and np.array_equal(onehot_cls, onehot_reg)

    cursor = 2
    category_counts: dict[str, int] = {}
    levels = metadata["dataB_classification_onehot.yaml"]["categorical_levels"]
    for key in DATA_B_CATEGORICAL_KEYS:
        width = len(levels[key])
        block = onehot_cls[:, cursor : cursor + width]
        assert np.array_equal(block.sum(axis=1), np.ones(len(frame), dtype=np.float32))
        category_counts[key] = width
        cursor += width
    assert cursor == 155

    results = PROJECT_ROOT / "results"
    condition_results = results / "CONDITION_ENCODING"
    summaries = pd.read_csv(condition_results / "condition_encoding_summary.csv")
    differences = pd.read_csv(condition_results / "condition_encoding_difference.csv")
    assert set(summaries["model"]) == EXPECTED_MODELS
    assert set(summaries["encoding"]) == {"Original scalar codes", "Fixed one-hot"}
    assert len(summaries) == len(EXPECTED_MODELS) * 2
    assert set(differences["model"]) == EXPECTED_MODELS and len(differences) == len(EXPECTED_MODELS)

    task_metrics = {
        "classification": ("recall_macro", "f1_macro", "auc"),
        "regression": ("r2", "rmse"),
    }
    saved_fold_rows = 0
    for model in sorted(EXPECTED_MODELS):
        family = Path("CHIMERA") if model == "CHIMERA" else Path("BASELINE") / model
        for encoding, root in (
            ("Original scalar codes", results),
            ("Fixed one-hot", condition_results),
        ):
            summary = summaries.loc[
                (summaries["model"] == model) & (summaries["encoding"] == encoding)
            ].iloc[0]
            for task, metrics in task_metrics.items():
                group = pd.read_csv(root / family / f"dataB_{task}" / "fold_metrics.csv")
                assert sorted(group["fold"].astype(int).tolist()) == list(range(1, 11))
                if encoding == "Fixed one-hot":
                    saved_fold_rows += len(group)
                for metric in metrics:
                    assert np.isclose(group[metric].mean(), summary[f"{metric}_mean"])
                    assert np.isclose(group[metric].std(ddof=1), summary[f"{metric}_std"])

        original = summaries.loc[
            (summaries["model"] == model) & (summaries["encoding"] == "Original scalar codes")
        ].iloc[0]
        onehot = summaries.loc[
            (summaries["model"] == model) & (summaries["encoding"] == "Fixed one-hot")
        ].iloc[0]
        difference = differences.loc[differences["model"] == model].iloc[0]
        for metrics in task_metrics.values():
            for metric in metrics:
                expected = onehot[f"{metric}_mean"] - original[f"{metric}_mean"]
                assert np.isclose(difference[f"{metric}_difference"], expected)

    report = {
        "status": "PASS",
        "raw_rows": int(len(frame)),
        "final_rows": int(len(frame)),
        "rows_removed": 0,
        "ddg_zero_rows": int((ddg == 0).sum()),
        "ddg_missing_rows": int(ddg.isna().sum()),
        "ee_zero_rows": int((ee == 0).sum()),
        "ee_missing_rows": int(ee.isna().sum()),
        "absolute_ee_distribution": {
            "min": float(ee.abs().min()),
            "q25": float(ee.abs().quantile(0.25)),
            "median": float(ee.abs().median()),
            "q75": float(ee.abs().quantile(0.75)),
            "max": float(ee.abs().max()),
        },
        "class_balance": {"low_ee": 455, "high_ee": 431},
        "scalar_condition_dim": int(scalar_cls.shape[1]),
        "onehot_condition_dim": int(onehot_cls.shape[1]),
        "onehot_category_counts": category_counts,
        "same_encoding_for_classification_and_regression": True,
        "saved_onehot_models": sorted(EXPECTED_MODELS),
        "saved_outer_folds_per_model": 10,
        "saved_fold_metric_rows": saved_fold_rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
