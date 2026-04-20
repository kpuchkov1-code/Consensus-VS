"""Tests for the data layer (ChEMBL cache, decoys, ligand prep, receptor)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

from btk_aidd.config import PropertyWindows
from btk_aidd.data.chembl import _to_nanomolar, load_cached, save_cache
from btk_aidd.data.decoys import DecoyGenerator, DecoyGeneratorConfig, compute_properties
from btk_aidd.data.ligands import LigandPreparer
from btk_aidd.data.receptor import pocket_from_reference_ligand

# -------------------------------------------------------------------- ChEMBL


def test_load_cached_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "molecule_chembl_id": "CHEMBL1",
                "canonical_smiles": "CCO",
                "standard_type": "IC50",
                "standard_value_nM": 50.0,
                "pchembl_value": 7.3,
                "target_chembl_id": "CHEMBL5251",
            },
        ]
    )
    path = tmp_path / "a.csv"
    save_cache(df, path)
    loaded = load_cached(path)
    assert list(loaded.columns)[:6] == [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_value_nM",
        "pchembl_value",
        "target_chembl_id",
    ]
    assert len(loaded) == 1


def test_load_cached_rejects_missing_columns(tmp_path: Path) -> None:
    df = pd.DataFrame([{"molecule_chembl_id": "X", "canonical_smiles": "C"}])
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_cached(path)


@pytest.mark.parametrize(
    ("value", "units", "expected"),
    [
        (100.0, "nM", 100.0),
        (0.5, "uM", 500.0),
        (2.0, "mM", 2_000_000.0),
        (500.0, "pM", 0.5),
        (1e-9, "M", 1.0),
        (1.0, None, None),
        (1.0, "g/L", None),
    ],
)
def test_to_nanomolar(value: float, units: str | None, expected: float | None) -> None:
    assert _to_nanomolar(value, units) == expected


# ---------------------------------------------------------------------- decoys


def test_compute_properties_from_aspirin() -> None:
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert mol is not None
    props = compute_properties(mol)
    assert 150 < props.mw < 200
    assert props.hba >= 3
    assert props.hbd == 1
    assert props.charge == 0


def test_decoy_generator_respects_similarity_cutoff() -> None:
    cfg = DecoyGeneratorConfig(
        windows=PropertyWindows(mw=500, logp=5, hba=10, hbd=10, rotatable=10, charge=2),
        similarity_cutoff=0.01,  # extremely strict → nothing survives
        random_seed=0,
    )
    gen = DecoyGenerator(cfg)
    # aspirin as sole active, with aspirin itself in candidate pool
    picked = gen.generate(
        actives_smiles=["CC(=O)OC1=CC=CC=C1C(=O)O"],
        candidate_smiles=["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO"],
        count=5,
    )
    assert "CC(=O)OC1=CC=CC=C1C(=O)O" not in picked


def test_decoy_generator_honours_property_windows() -> None:
    cfg = DecoyGeneratorConfig(
        windows=PropertyWindows(mw=10, logp=0.1, hba=0, hbd=0, rotatable=0, charge=0),
        similarity_cutoff=1.0,  # disable similarity filter for this test
        random_seed=0,
    )
    gen = DecoyGenerator(cfg)
    picked = gen.generate(
        actives_smiles=["CCO"],
        candidate_smiles=["CCCCCCCC"],  # MW way off
        count=5,
    )
    assert picked == []


def test_decoy_generator_accepts_close_match() -> None:
    cfg = DecoyGeneratorConfig(
        windows=PropertyWindows(mw=50, logp=2.0, hba=3, hbd=3, rotatable=3, charge=1),
        similarity_cutoff=0.9,  # permissive
        random_seed=0,
    )
    gen = DecoyGenerator(cfg)
    picked = gen.generate(
        actives_smiles=["c1ccccc1"],
        candidate_smiles=["c1ccncc1", "CC", "CCCCCC"],
        count=5,
    )
    assert len(picked) >= 1


# -------------------------------------------------------------------- ligands


def test_ligand_preparer_produces_3d(default_config) -> None:
    p = LigandPreparer(default_config.ligand_prep)
    result = p.prepare("x", "c1ccccc1")
    assert result is not None
    assert result.embed_success
    assert result.mol.GetNumConformers() >= 1
    assert result.mol.GetNumHeavyAtoms() == 6


def test_ligand_preparer_handles_bad_smiles(default_config) -> None:
    p = LigandPreparer(default_config.ligand_prep)
    assert p.prepare("x", "this_is_not_smiles") is None


def test_ligand_preparer_batch_drops_failures(default_config) -> None:
    p = LigandPreparer(default_config.ligand_prep)
    out = p.prepare_many([("x", "c1ccccc1"), ("y", "!!!"), ("z", "CCO")])
    assert len(out) == 2
    assert {lig.name for lig in out} == {"x", "z"}


# -------------------------------------------------------------------- receptor


def test_pocket_extraction(synthetic_receptor_pdb: Path) -> None:
    pocket = pocket_from_reference_ligand(
        synthetic_receptor_pdb,
        reference_ligand_resname="LIG",
        box_padding=2.0,
    )
    assert pocket.atom_coords.shape == (4, 3)
    # centroid should be near (5.75, 5.25, 5.25) for our four LIG atoms
    cx, cy, cz = pocket.centroid
    assert abs(cx - 5.75) < 0.1
    assert abs(cy - 5.25) < 0.1
    assert abs(cz - 5.25) < 0.1
    # box size = (max - min) + 2 * padding; for our atoms dims ~1.5 + 4 = ~5.5
    assert 5.0 < pocket.box.size[0] < 6.0


def test_pocket_extraction_raises_when_resname_missing(synthetic_receptor_pdb: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        pocket_from_reference_ligand(synthetic_receptor_pdb, "ZZZ")
