from __future__ import annotations

from typing import Any

import pandas as pd
from rdkit import Chem


COMPONENT_COLUMNS = {
    "catalyst": "ligand",
    "product": "product",
    "reactant1": "R1",
    "reactant2": "R2",
}


def canonical_component(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return "__MISSING__"
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return text
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def component_split_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    missing = [column for column in COMPONENT_COLUMNS.values() if column not in df.columns]
    if missing:
        raise KeyError(f"Cannot build component split groups; missing columns: {missing}")
    return {
        key: [canonical_component(value) for value in df[column].tolist()]
        for key, column in COMPONENT_COLUMNS.items()
    }
