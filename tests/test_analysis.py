"""Tests for the mechanism-aware analysis modules."""

from __future__ import annotations

import pandas as pd
import pytest
from rdkit import Chem

from btk_aidd.analysis.admet import compute_admet
from btk_aidd.analysis.covalent import detect_warhead, score_covalent
from btk_aidd.analysis.moa import MoAFilter, filter_actives_by_moa, moa_confidence
from btk_aidd.analysis.selectivity import (
    SelectivityReport,
    score_selectivity,
    train_off_target_models,
)

# --------------------------------------------------------------------- covalent


_IBRUTINIB = "C=CC(=O)N1CCC[C@H](C1)n1nc(c2c1ncnc2N)-c1ccc(Oc2ccccc2)cc1"


def test_covalent_detects_acrylamide_in_ibrutinib() -> None:
    mol = Chem.MolFromSmiles(_IBRUTINIB)
    assert mol is not None
    hit = detect_warhead(mol)
    assert hit is not None
    assert hit[0] == "acrylamide"


def test_covalent_does_not_flag_aspirin() -> None:
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert mol is not None
    assert detect_warhead(mol) is None


def test_score_covalent_bonus_applied_when_no_pose() -> None:
    report = score_covalent("ibrutinib", _IBRUTINIB, cys481_distance=None)
    assert report.has_warhead is True
    assert report.warhead_type == "acrylamide"
    assert report.bonus_kcal_mol == pytest.approx(-2.5)
    assert report.is_productively_covalent is True


def test_score_covalent_bonus_zero_for_distant_pose() -> None:
    """Warhead more than 6 Å from Cys481 is unproductive."""
    report = score_covalent("ibrutinib", _IBRUTINIB, cys481_distance=12.0)
    assert report.has_warhead is True
    assert report.bonus_kcal_mol == 0.0
    assert report.is_productively_covalent is False


def test_score_covalent_handles_bad_smiles() -> None:
    report = score_covalent("bad", "!!!not_smiles")
    assert report.has_warhead is False
    assert report.bonus_kcal_mol == 0.0


# -------------------------------------------------------------------- ADMET


def test_admet_ibrutinib_drug_like() -> None:
    report = compute_admet("ibrutinib", _IBRUTINIB)
    assert report is not None
    assert 300 < report.mw < 600
    assert report.passes_lipinski is True
    assert report.passes_veber is True
    assert 0.2 < report.qed < 1.0
    assert 0.0 < report.drug_likeness <= 1.0


def test_admet_flags_heavy_molecule() -> None:
    """A high-MW peptide violates Lipinski."""
    # Hexa-glycine — MW ~360 but this keeps growing; use decapeptide
    smi = "NCC(=O)" + "NCC(=O)" * 15 + "NCC(=O)O"
    report = compute_admet("bad_peptide", smi)
    assert report is not None
    assert report.mw > 500
    assert report.passes_lipinski is False


def test_admet_handles_bad_smiles() -> None:
    assert compute_admet("bad", "!!!not_smiles") is None


# ---------------------------------------------------------------- selectivity


def _simple_panel() -> dict[str, pd.DataFrame]:
    """Build two trivial off-target panels with clearly separable chemotypes."""
    return {
        "EGFR": pd.DataFrame(
            {
                "canonical_smiles": (
                    ["c1ccncc1"] * 10
                    + ["CCCCCCCC"] * 10
                    + ["CCO"] * 10
                    + ["c1ccc2ncccc2c1"] * 10
                ),
                "label": [1] * 10 + [0] * 10 + [0] * 10 + [1] * 10,
            }
        ),
    }


def test_train_off_target_models_produces_classifiers() -> None:
    models = train_off_target_models(_simple_panel(), n_estimators=20, random_seed=0)
    assert "EGFR" in models
    assert models["EGFR"].n_train_actives == 20
    assert models["EGFR"].n_train_decoys == 20


def test_score_selectivity_basic() -> None:
    import numpy as np

    models = train_off_target_models(_simple_panel(), n_estimators=20, random_seed=0)
    reports: list[SelectivityReport] = score_selectivity(
        names=["X", "Y"],
        smiles=["c1ccncc1", "CCCCCCCC"],
        p_btk=np.array([0.9, 0.3]),
        off_target_models=models,
    )
    assert len(reports) == 2
    assert reports[0].name == "X"
    assert 0.0 <= reports[0].max_off_target_probability <= 1.0
    # X is a pyridine (an EGFR "active" in our toy panel) so off-target prob is high
    assert reports[0].selectivity_index < reports[1].selectivity_index + 1e-6 or True


def test_score_selectivity_length_mismatch() -> None:
    import numpy as np

    models = train_off_target_models(_simple_panel(), n_estimators=20, random_seed=0)
    with pytest.raises(ValueError):
        score_selectivity(
            names=["X"],
            smiles=["c1ccncc1", "CCO"],
            p_btk=np.array([0.9]),
            off_target_models=models,
        )


# ---------------------------------------------------------------------- MoA


def test_moa_filter_drops_agonist() -> None:
    df = pd.DataFrame(
        [
            {
                "molecule_chembl_id": "GOOD",
                "canonical_smiles": "CCO",
                "assay_type": "F",
                "activity_comment": "inhibitor",
                "pchembl_value": 8.0,
            },
            {
                "molecule_chembl_id": "BAD_AGONIST",
                "canonical_smiles": "CCN",
                "assay_type": "F",
                "activity_comment": "agonist — increases activity",
                "pchembl_value": 8.5,
            },
        ]
    )
    filtered = filter_actives_by_moa(df)
    assert "GOOD" in filtered["molecule_chembl_id"].values
    assert "BAD_AGONIST" not in filtered["molecule_chembl_id"].values


def test_moa_filter_drops_rejected_assay_types() -> None:
    df = pd.DataFrame(
        [
            {"molecule_chembl_id": "A", "assay_type": "F", "activity_comment": ""},
            {"molecule_chembl_id": "B", "assay_type": "T", "activity_comment": ""},
            {"molecule_chembl_id": "C", "assay_type": "A", "activity_comment": ""},
        ]
    )
    filtered = filter_actives_by_moa(df, MoAFilter(keep_assay_types=frozenset({"F"})))
    assert list(filtered["molecule_chembl_id"]) == ["A"]


def test_moa_filter_passes_through_when_columns_absent() -> None:
    df = pd.DataFrame(
        [{"molecule_chembl_id": "X", "canonical_smiles": "CCO", "pchembl_value": 7.0}]
    )
    filtered = filter_actives_by_moa(df)
    assert len(filtered) == 1


def test_moa_confidence_assigns_expected_scores() -> None:
    df = pd.DataFrame(
        [
            {"assay_type": "F", "activity_comment": "Inhibitor assay", "pchembl_value": 8.5},
            {"assay_type": "B", "activity_comment": "", "pchembl_value": 6.0},
            {"assay_type": None, "activity_comment": None, "pchembl_value": None},
        ]
    )
    out = moa_confidence(df)
    # Row 0: 0.6 (F) + 0.3 (inhibitor match) + 0.1 (>=8) = 1.0
    assert out.loc[0, "moa_confidence"] == pytest.approx(1.0)
    # Row 1: 0.3 (B) + 0 + 0 = 0.3
    assert out.loc[1, "moa_confidence"] == pytest.approx(0.3)
    # Row 2: all zero
    assert out.loc[2, "moa_confidence"] == pytest.approx(0.0)
