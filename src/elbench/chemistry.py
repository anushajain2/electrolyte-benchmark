"""Reviewed molecular identity mapping and fixed RDKit descriptors."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# Fixed, documented descriptor set. Order is part of the benchmark definition.
SOLVENT_DESCRIPTORS = [
    "mw",  # molecular weight
    "logp",  # Crippen logP
    "tpsa",  # topological polar surface area
    "hba",  # H-bond acceptors (Lipinski)
    "hbd",  # H-bond donors (Lipinski)
    "rotb",  # rotatable bonds
    "rings",  # ring count
    "fsp3",  # fraction sp3 carbons
    "heavy",  # heavy atom count
    "frac_halogen",  # (F+Cl+Br+I) / heavy atoms
]
ANION_DESCRIPTORS = ["mw", "heavy", "n_F", "tpsa", "rings", "rotb"]


def load_species(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ["cation_stoich", "anion_stoich"]:
        df[col] = pd.to_numeric(df[col].replace("", np.nan))
    if df.duplicated(["dataset_id", "raw_name"]).any():
        raise ValueError("species mapping has duplicate (dataset_id, raw_name) rows")
    # Same canonical id must always resolve to the same structure.
    struct = df[df.review_status == "verified"].groupby("canonical_id")[["smiles", "anion_smiles", "cation_smiles"]].nunique()
    if (struct > 1).any().any():
        raise ValueError("canonical id maps to more than one structure")
    return df


@lru_cache(maxsize=None)
def mol(smiles: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"unparseable SMILES {smiles!r}")
    return m


@lru_cache(maxsize=None)
def canonical_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(mol(smiles))


@lru_cache(maxsize=None)
def molar_mass(smiles: str) -> float:
    return float(Descriptors.MolWt(mol(smiles)))


def salt_molar_mass(row: pd.Series) -> float:
    return row["cation_stoich"] * molar_mass(row["cation_smiles"]) + row["anion_stoich"] * molar_mass(row["anion_smiles"])


@lru_cache(maxsize=None)
def solvent_descriptors(smiles: str) -> tuple[float, ...]:
    m = mol(smiles)
    heavy = m.GetNumHeavyAtoms()
    halogens = sum(1 for a in m.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I"))
    vals = {
        "mw": Descriptors.MolWt(m),
        "logp": Crippen.MolLogP(m),
        "tpsa": rdMolDescriptors.CalcTPSA(m),
        "hba": Lipinski.NumHAcceptors(m),
        "hbd": Lipinski.NumHDonors(m),
        "rotb": rdMolDescriptors.CalcNumRotatableBonds(m),
        "rings": rdMolDescriptors.CalcNumRings(m),
        "fsp3": rdMolDescriptors.CalcFractionCSP3(m),
        "heavy": heavy,
        "frac_halogen": halogens / heavy if heavy else 0.0,
    }
    return tuple(float(vals[k]) for k in SOLVENT_DESCRIPTORS)


@lru_cache(maxsize=None)
def anion_descriptors(smiles: str) -> tuple[float, ...]:
    m = mol(smiles)
    vals = {
        "mw": Descriptors.MolWt(m),
        "heavy": m.GetNumHeavyAtoms(),
        "n_F": sum(1 for a in m.GetAtoms() if a.GetSymbol() == "F"),
        "tpsa": rdMolDescriptors.CalcTPSA(m),
        "rings": rdMolDescriptors.CalcNumRings(m),
        "rotb": rdMolDescriptors.CalcNumRotatableBonds(m),
    }
    return tuple(float(vals[k]) for k in ANION_DESCRIPTORS)


def descriptor_table(species: pd.DataFrame) -> pd.DataFrame:
    """Per canonical component: descriptors from SMILES only (no target information)."""
    rows = []
    for _, r in species[species.review_status == "verified"].drop_duplicates("canonical_id").iterrows():
        if r.role == "solvent":
            d = dict(zip(SOLVENT_DESCRIPTORS, solvent_descriptors(r.smiles)))
            rows.append({"canonical_id": r.canonical_id, "role": "solvent", "smiles": canonical_smiles(r.smiles), **{f"solv_{k}": v for k, v in d.items()}})
        else:
            d = dict(zip(ANION_DESCRIPTORS, anion_descriptors(r.anion_smiles)))
            rows.append({"canonical_id": r.canonical_id, "role": "salt", "smiles": canonical_smiles(r.anion_smiles), **{f"anion_{k}": v for k, v in d.items()}})
    return pd.DataFrame(rows)
