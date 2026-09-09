from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from chimera.common.splits import component_split_groups

from .packages import pyg_data_generation


def _sanitize_mol(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    if mol is None:
        return None
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        mol,
        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_KEKULIZE
        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        catchErrors=True,
    )
    return mol


def _fingerprint_concat(row: pd.Series, smiles_cols: list[str], n_bits: int = 1024) -> np.ndarray:
    parts = []
    for col in smiles_cols:
        mol = _sanitize_mol(row[col]) if col in row else None
        arr = np.zeros((n_bits,), dtype=np.int8)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, n_bits)
            DataStructs.ConvertToNumpyArray(fp, arr)
        parts.append(arr)
    return np.concatenate(parts).astype(np.float32)


def _as_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _prepare_data_a_frame(df: pd.DataFrame, cfg: dict[str, Any], task_type: str) -> pd.DataFrame:
    data_cfg = cfg.get("data", {})
    aliases = {
        "ligand": data_cfg.get("ligand_column", "Ni_ligan_Smiles"),
        "product": data_cfg.get("product_column", "product"),
        "R1": data_cfg.get("r1_column", "R1"),
        "R2": data_cfg.get("r2_column", "R2"),
    }
    missing = [src for src in aliases.values() if src not in df.columns]
    if missing:
        raise KeyError(f"Data A is missing required column(s): {missing}")
    for dst, src in aliases.items():
        df[dst] = df[src]

    essential = ["ligand", "product", "R1", "R2"]
    df = df.dropna(subset=essential).copy()
    for col in essential:
        df = df[df[col].astype(str).str.strip().ne("")]
    df = df.reset_index(drop=True)

                                                                      
    temp_col = data_cfg.get("temperature_column", "Temp")
    time_col = data_cfg.get("time_column", "Time[h]")
    df["tem"] = _as_numeric(df[temp_col], 0.0) if temp_col in df.columns else 0
    df["Time"] = _as_numeric(df[time_col], 0.0) if time_col in df.columns else 0

                                                                                 
                                                                         
    if data_cfg.get("metal_column") and data_cfg["metal_column"] in df.columns:
        df["metal"] = pd.factorize(df[data_cfg["metal_column"]].fillna("NA").astype(str))[0]
    else:
        df["metal"] = 0
    df["solvent"] = 0
    df["additive"] = 0
    df["gm"] = 0
    df["elsi"] = 0

    if task_type == "classification":
        ee_col = data_cfg.get("ee_column", "Yield (optical)")
        if ee_col not in df.columns:
            raise KeyError(f"Data A classification ee column not found: {ee_col}")
        ee = pd.to_numeric(df[ee_col], errors="coerce")
        df = df.loc[ee.notna()].copy()
        ee = ee.loc[df.index]
        if data_cfg.get("use_absolute_ee", True):
            ee = ee.abs()
        df["ee"] = ee.astype(float)
        threshold = float(data_cfg.get("ee_threshold", 90.0))
        df["target_label"] = (df["ee"] >= threshold).astype(int)
        df["label"] = df["target_label"]
    else:
        target_col = data_cfg.get("target_column", "ddG")
        if target_col not in df.columns:
            raise KeyError(f"Data A regression target column not found: {target_col}")
        target = pd.to_numeric(df[target_col], errors="coerce")
        df = df.loc[target.notna()].copy()
        target = target.loc[df.index]
        if data_cfg.get("drop_zero_ddg", False):
            keep = target != 0
            df = df.loc[keep].copy()
            target = target.loc[keep]
        df["target_value"] = target.astype(float)
        df["label"] = 0

    return df.reset_index(drop=True)


def build_dataset(data_dir: str | Path, cfg: dict[str, Any]):
    data_dir = Path(data_dir)
    data_file = data_dir / cfg.get("data", {}).get("reaction_file", "data3647.xlsx")
    df = pd.read_excel(data_file, header=0)
    df = _prepare_data_a_frame(df, cfg, task_type="classification")

    smiles_cols = cfg.get("data", {}).get("fingerprint_smiles_cols", ["product", "ligand"])
    n_bits = int(cfg.get("data", {}).get("fingerprint_bits", 1024))
    fps = [_fingerprint_concat(row, smiles_cols, n_bits) for _, row in df.iterrows()]

    temp = df["tem"]
    time = df["Time"]
    metal = df["metal"]
    solvent = df["solvent"]
    additive = df["additive"]
    gm = df["gm"]
    elsi = df["elsi"]
    label = df["target_label"]

                                                                             
                                                                             
    dataset = pyg_data_generation(df, temp, time, metal, solvent, additive, gm, elsi, label, fps, fps)
    metadata = {
        "_split_groups": component_split_groups(df),
        "dataset": "Data A",
        "num_rows": int(len(df)),
        "data_file": str(data_file),
        "target": "ee >= threshold",
        "fingerprint_smiles_cols": smiles_cols,
        "fingerprint_dim": int(len(fps[0])) if fps else 0,
        "smiles_aliases": {
            "ligand": cfg.get("data", {}).get("ligand_column", "Ni_ligan_Smiles"),
            "product": cfg.get("data", {}).get("product_column", "product"),
            "R1": cfg.get("data", {}).get("r1_column", "R1"),
            "R2": cfg.get("data", {}).get("r2_column", "R2"),
        },
    }
    return dataset, metadata
