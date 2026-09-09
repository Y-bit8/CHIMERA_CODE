from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from chimera.common.splits import component_split_groups

from .packages import pyg_data_generation

GRAPH_FEATURE_FILES = [
    "Ir-Ni L Atom.csv", "Ir-Ni L Motif.csv",
    "Ir-Ni P Atom.csv", "Ir-Ni P Motif.csv",
    "Ir-Ni R1 Atom.csv", "Ir-Ni R1 Motif.csv",
    "Ir-Ni R2 Atom.csv", "Ir-Ni R2 Motif.csv",
]


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


def build_dataset(data_dir: str | Path, cfg: dict[str, Any]):
    data_dir = Path(data_dir)
    data_file = data_dir / cfg.get("data", {}).get("reaction_file", "Ir-Ni reaction.xlsx")
    df = pd.read_excel(data_file, header=0)
    if cfg.get("data", {}).get("drop_zero_ddg", True) and "ddG" in df.columns:
        df = df[df["ddG"] != 0].reset_index(drop=True)
    target_col = cfg.get("data", {}).get("target_column", "ddG")
    df["target_value"] = df[target_col].astype(float)
    df["label"] = 0
    df["bondary"] = df[cfg.get("data", {}).get("boundary_column", "tem")]
    column_index = df.columns.get_loc("bondary")
    feature_frames = [pd.read_csv(data_dir / name, header=0) for name in GRAPH_FEATURE_FILES]
    df = pd.concat([df] + feature_frames, axis=1).reset_index(drop=True)
    smiles_cols = cfg.get("data", {}).get("fingerprint_smiles_cols", ["ligand", "product"])
    fps = [_fingerprint_concat(row, smiles_cols, int(cfg.get("data", {}).get("fingerprint_bits", 1024))) for _, row in df.iterrows()]
    temp = df[cfg.get("data", {}).get("temperature_column", "tem")]
    time = df[cfg.get("data", {}).get("time_column", "Time")]
    metal = df[cfg.get("data", {}).get("metal_column", "metal")]
    solvent = df[cfg.get("data", {}).get("solvent_column", "solvent")]
    additive = df[cfg.get("data", {}).get("additive_column", "additive")]
    gm = df[cfg.get("data", {}).get("gm_column", "gm")]
    elsi = df[cfg.get("data", {}).get("elsi_column", "elsi")]
    label = df["label"]
    add_fea1 = df.iloc[:, column_index + 1 :].values
    dataset = pyg_data_generation(df, temp, time, metal, solvent, additive, gm, elsi, label, fps, add_fea1)
    metadata = {
        "_split_groups": component_split_groups(df),
        "num_rows": int(len(df)),
        "data_file": str(data_file),
        "fingerprint_smiles_cols": smiles_cols,
        "columns": list(map(str, df.columns[:40])),
    }
    return dataset, metadata
