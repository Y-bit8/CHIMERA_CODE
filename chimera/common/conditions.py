from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_B_CATEGORICAL_KEYS = ("metal", "solvent", "additive", "gm", "elsi")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        raise KeyError(f"Missing Data B condition column: {column}")
    values = pd.to_numeric(df[column], errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Data B condition column {column!r} contains non-finite values")
    return values


def build_data_b_conditions(
    df: pd.DataFrame,
    data_dir: str | Path,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build Data B conditions without fitting on labels or held-out folds.

    ``scalar_codes`` reproduces the historical seven-scalar representation.
    ``fixed_one_hot`` retains temperature and time as continuous values and
    expands the five nominal variables using the released fixed codebook.
    """

    data_cfg = cfg.get("data", {})
    continuous = (
        ("temperature", data_cfg.get("temperature_column", "tem")),
        ("time", data_cfg.get("time_column", "Time")),
    )
    categorical = tuple(
        (key, data_cfg.get(f"{key}_column", key)) for key in DATA_B_CATEGORICAL_KEYS
    )
    encoding = str(data_cfg.get("condition_encoding", "scalar_codes")).lower()

    arrays = [_numeric_column(df, source)[:, None] for _, source in continuous]
    feature_names = [name for name, _ in continuous]
    metadata: dict[str, Any] = {
        "encoding": encoding,
        "continuous_columns": [source for _, source in continuous],
        "categorical_columns": [source for _, source in categorical],
    }

    if encoding == "scalar_codes":
        for key, source in categorical:
            arrays.append(_numeric_column(df, source)[:, None])
            feature_names.append(key)
    elif encoding == "fixed_one_hot":
        code_key_path = Path(data_dir) / data_cfg.get("condition_code_key", "DataB_code_key.csv")
        if not code_key_path.is_file():
            raise FileNotFoundError(f"Missing fixed Data B condition codebook: {code_key_path}")
        code_key = pd.read_csv(code_key_path, encoding="utf-8-sig")
        required = {"coded_column", "code", "original_value"}
        missing = sorted(required.difference(code_key.columns))
        if missing:
            raise ValueError(f"{code_key_path} is missing codebook columns: {missing}")

        categorical_levels: dict[str, list[dict[str, Any]]] = {}
        for key, source in categorical:
            rows = code_key.loc[code_key["coded_column"].astype(str) == key].copy()
            if rows.empty:
                raise ValueError(f"{code_key_path} has no entries for {key!r}")
            rows["code"] = pd.to_numeric(rows["code"], errors="raise")
            if rows["code"].duplicated().any():
                raise ValueError(f"{code_key_path} contains duplicate codes for {key!r}")
            rows = rows.sort_values("code")
            observed = _numeric_column(df, source)
            levels = rows["code"].to_numpy(dtype=np.float32)
            unknown = sorted(set(observed.tolist()).difference(levels.tolist()))
            if unknown:
                raise ValueError(f"Observed {source!r} codes are absent from the fixed codebook: {unknown}")

            one_hot = (observed[:, None] == levels[None, :]).astype(np.float32)
            if not np.all(one_hot.sum(axis=1) == 1.0):
                raise ValueError(f"Every {source!r} value must map to exactly one fixed category")
            arrays.append(one_hot)
            feature_names.extend(
                f"{key}={int(code) if float(code).is_integer() else code}" for code in levels
            )
            categorical_levels[key] = [
                {
                    "code": int(row.code) if float(row.code).is_integer() else float(row.code),
                    "original_value": str(row.original_value),
                }
                for row in rows.itertuples(index=False)
            ]

        metadata.update(
            {
                "code_key_file": str(code_key_path),
                "code_key_sha256": _sha256(code_key_path),
                "categorical_levels": categorical_levels,
            }
        )
    else:
        raise ValueError(
            "data.condition_encoding must be 'scalar_codes' or 'fixed_one_hot', "
            f"got {encoding!r}"
        )

    matrix = np.concatenate(arrays, axis=1).astype(np.float32, copy=False)
    metadata.update(
        {
            "condition_dim": int(matrix.shape[1]),
            "feature_names": feature_names,
        }
    )
    return matrix, metadata
