"""Shared pytest fixtures.

All fixtures live here so individual test modules stay focused on the
behaviour they are exercising. The :func:`project_root` fixture locates the
repo root regardless of where pytest is invoked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btk_aidd.config import Config, load_config
from btk_aidd.data.ligands import LigandPreparer, PreparedLigand


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Repository root (``btk-aidd/``)."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def default_config(project_root: Path) -> Config:
    """Load the packaged default config exactly as production would."""
    return load_config(project_root / "config" / "default.yaml")


@pytest.fixture(scope="session")
def fixtures_dir(project_root: Path) -> Path:
    return project_root / "data" / "fixtures"


@pytest.fixture()
def simple_smiles_set() -> list[tuple[str, str, int]]:
    """A handful of drug SMILES with names and binary activity labels."""
    return [
        ("ACT1", "CC(=O)OC1=CC=CC=C1C(=O)O", 1),  # aspirin
        ("ACT2", "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", 1),  # erlotinib
        ("ACT3", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 1),  # ibuprofen
        ("ACT4", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 1),  # caffeine
        ("DEC1", "CCO", 0),
        ("DEC2", "CCCCCCCCO", 0),
        ("DEC3", "c1ccccc1", 0),
        ("DEC4", "CC(=O)N", 0),
    ]


@pytest.fixture()
def prepared_ligands(default_config: Config) -> list[PreparedLigand]:
    """Return a small list of MMFF-minimised ligands."""
    preparer = LigandPreparer(default_config.ligand_prep)
    items = [
        ("LIG_A", "c1ccccc1"),
        ("LIG_B", "CCO"),
        ("LIG_C", "CC(=O)O"),
        ("LIG_D", "COc1ccccc1"),
    ]
    return preparer.prepare_many(items)


@pytest.fixture()
def cached_scores_csv(tmp_path: Path) -> Path:
    """Write a tiny cached docking CSV for docking-engine tests."""
    df = pd.DataFrame(
        {
            "name": ["LIG_A", "LIG_B", "LIG_C", "LIG_D"],
            "affinity_kcal_mol": [-9.2, -6.5, -7.8, -5.9],
        }
    )
    path = tmp_path / "cache.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def synthetic_receptor_pdb(tmp_path: Path) -> Path:
    """Write a minimal valid PDB with a reference ligand ('LIG') at origin."""
    pdb = (
        "HEADER    TEST\n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O\n"
        "ATOM      5  CB  ALA A   1       1.988  -0.773  -1.199  1.00  0.00           C\n"
        "HETATM  100  C1  LIG A   1       5.000   5.000   5.000  1.00  0.00           C\n"
        "HETATM  101  C2  LIG A   1       5.500   5.500   5.500  1.00  0.00           C\n"
        "HETATM  102  N1  LIG A   1       6.000   5.000   5.500  1.00  0.00           N\n"
        "HETATM  103  O1  LIG A   1       6.500   5.500   5.000  1.00  0.00           O\n"
        "HETATM  104  O   HOH A   2      10.000  10.000  10.000  1.00  0.00           O\n"
        "END\n"
    )
    path = tmp_path / "test_receptor.pdb"
    path.write_text(pdb)
    return path


@pytest.fixture()
def binary_rng() -> np.random.Generator:
    return np.random.default_rng(42)
