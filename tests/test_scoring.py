"""Tests for the physics, ML, and consensus scorers."""

from __future__ import annotations

import numpy as np
import pytest

from btk_aidd.config import ConsensusWeights, MLScoringConfig, PhysicsScoringConfig
from btk_aidd.data.ligands import PreparedLigand
from btk_aidd.scoring.consensus import consensus_score
from btk_aidd.scoring.ml import MLRescorer
from btk_aidd.scoring.physics import PhysicsRescorer

# -------------------------------------------------------------------- physics


def test_physics_rescorer_returns_finite(prepared_ligands: list[PreparedLigand]) -> None:
    rescorer = PhysicsRescorer(PhysicsScoringConfig())
    for ligand in prepared_ligands:
        result = rescorer.score(ligand)
        assert np.isfinite(result.total)
        # components sum to total
        assert np.isclose(
            result.total,
            result.hydrophobic + result.hbond + result.strain + result.size,
        )


def test_physics_rescorer_hydrophobic_sign(prepared_ligands: list[PreparedLigand]) -> None:
    """LogP > 0 should produce a negative (stabilising) hydrophobic term."""
    rescorer = PhysicsRescorer(PhysicsScoringConfig())
    for ligand in prepared_ligands:
        result = rescorer.score(ligand)
        # Hansch coefficient is -0.70 * logP. If logP positive, hydrophobic <= 0.
        from rdkit.Chem import Crippen

        logp = Crippen.MolLogP(ligand.mol)
        if logp > 0:
            assert result.hydrophobic <= 0


# ------------------------------------------------------------------------ ML


def test_ml_rescorer_perfect_separation() -> None:
    """With perfectly separable chemotypes, the classifier should score ~1.0 AUC."""
    names = [f"A{i}" for i in range(15)] + [f"D{i}" for i in range(15)]
    # Actives are all heavy aromatic nitrogen heterocycles; decoys are aliphatic chains.
    actives = ["c1ccncc1"] * 15
    decoys = ["CCCCCCCC"] * 15
    labels = np.array([1] * 15 + [0] * 15, dtype=int)

    rescorer = MLRescorer(MLScoringConfig(n_estimators=50, random_seed=0))
    split = rescorer.fit_predict(names, actives + decoys, labels)

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(split.test_labels, split.test_probabilities)
    assert auc >= 0.95


def test_ml_rescorer_length_mismatch() -> None:
    rescorer = MLRescorer(MLScoringConfig())
    with pytest.raises(ValueError):
        rescorer.fit_predict(["a"], ["CCO", "c1ccccc1"], np.array([1, 0]))


def test_ml_rescorer_requires_enough_data() -> None:
    rescorer = MLRescorer(MLScoringConfig())
    with pytest.raises(ValueError, match="Too few compounds"):
        rescorer.fit_predict(
            names=["a", "b"],
            smiles=["CCO", "c1ccccc1"],
            labels=np.array([1, 0]),
        )


# -------------------------------------------------------------------- consensus


def test_consensus_z_normalises() -> None:
    names = ["a", "b", "c", "d"]
    docking = np.array([-5.0, -9.0, -7.0, -6.0])
    physics = np.array([-2.0, -6.0, -4.0, -3.0])
    ml = np.array([-0.1, -0.9, -0.5, -0.3])

    result = consensus_score(
        names,
        docking,
        physics,
        ml,
        weights=ConsensusWeights(docking=1.0, physics=1.0, ml=1.0),
    ).frame

    # z-scores should have mean ~0 and std ~1 per scorer
    for col in ("docking_z", "physics_z", "ml_z"):
        assert abs(result[col].mean()) < 1e-9
        assert abs(result[col].std(ddof=0) - 1.0) < 1e-9


def test_consensus_weights_applied() -> None:
    names = ["a", "b", "c", "d"]
    docking = np.array([-5.0, -9.0, -7.0, -6.0])
    physics = np.zeros(4)  # no variance → z-score zero
    ml = np.zeros(4)

    result = consensus_score(
        names,
        docking,
        physics,
        ml,
        weights=ConsensusWeights(docking=0.5, physics=1.0, ml=1.0),
    ).frame

    # Because physics / ml have zero variance their z-scores are zero, so
    # consensus = 0.5 * docking_z
    expected = 0.5 * result["docking_z"].to_numpy()
    np.testing.assert_allclose(result["consensus"].to_numpy(), expected)


def test_consensus_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        consensus_score(
            ["a", "b"],
            np.array([-5.0, -9.0]),
            np.array([-5.0]),
            np.array([-5.0, -9.0]),
            weights=ConsensusWeights(),
        )
