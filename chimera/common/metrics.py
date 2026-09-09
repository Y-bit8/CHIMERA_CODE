from __future__ import annotations

import math
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error


def safe_float(x: Any) -> float | None:
    try:
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    except Exception:
        return None


def regression_metrics(y_true, y_pred) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true) else None
    mae = mean_absolute_error(y_true, y_pred) if len(y_true) else None
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else None
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0 else None
    return {"rmse": safe_float(rmse), "mae": safe_float(mae), "r2": safe_float(r2), "corr": safe_float(corr)}


def classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float | None]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    out: dict[str, float | None] = {"accuracy": None, "precision_macro": None, "recall_macro": None, "f1_macro": None, "auc": None}
    if len(y_true):
        out["accuracy"] = safe_float(accuracy_score(y_true, y_pred))
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        out.update({"precision_macro": safe_float(p), "recall_macro": safe_float(r), "f1_macro": safe_float(f1)})
        if y_score is not None:
            try:
                score = np.asarray(y_score)
                if score.ndim == 2 and score.shape[1] == 2:
                    out["auc"] = safe_float(roc_auc_score(y_true, score[:, 1]))
                elif score.ndim == 2 and score.shape[1] > 2:
                    out["auc"] = safe_float(roc_auc_score(y_true, score, multi_class="ovr"))
            except Exception:
                out["auc"] = None
    return out


def summarize_fold_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    summary: dict[str, Any] = {"num_folds": len(rows)}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if row.get(key) is not None]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return summary
