"""Consensus scoring: z-normalised weighted combination of individual scorers.

All individual scores are assumed to follow the "lower = better" convention
(matching AutoDock Vina affinity sign). This module:

1. Z-normalises each scorer independently,
2. Multiplies by the configured non-negative weight,
3. Sums to produce a single consensus score per ligand.

The weights do not have to sum to 1; they are passed through unchanged so that
ablation studies (e.g. weight = 0) are trivial.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from btk_aidd.config import ConsensusWeights
from btk_aidd.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ConsensusResult:
    """Per-ligand scores assembled into a single DataFrame."""

    frame: pd.DataFrame


def _z_normalise(values: np.ndarray) -> np.ndarray:
    """Return mean-centred, std-scaled values. Zero-std arrays are returned unchanged."""
    std = float(values.std())
    if std == 0.0 or not np.isfinite(std):
        return values - float(values.mean())
    return (values - float(values.mean())) / std


def consensus_score(
    names: list[str],
    docking: np.ndarray,
    physics: np.ndarray,
    ml: np.ndarray,
    weights: ConsensusWeights,
) -> ConsensusResult:
    """Combine three per-ligand score vectors into a consensus ranking.

    Args:
        names: Ligand identifiers, aligned with the three score vectors.
        docking: Vina affinities (lower = better).
        physics: Physics-rescorer totals (lower = better).
        ml: Sign-flipped ML probabilities (lower = better).
        weights: Non-negative weights for the three components.

    Returns:
        A :class:`ConsensusResult` whose ``frame`` has columns:
        ``name, docking, physics, ml, docking_z, physics_z, ml_z, consensus``.

    Raises:
        ValueError: If the input arrays disagree on length.
    """
    if not (len(names) == len(docking) == len(physics) == len(ml)):
        raise ValueError(
            f"length mismatch: names={len(names)}, docking={len(docking)}, "
            f"physics={len(physics)}, ml={len(ml)}"
        )

    dz = _z_normalise(np.asarray(docking, dtype=float))
    pz = _z_normalise(np.asarray(physics, dtype=float))
    mz = _z_normalise(np.asarray(ml, dtype=float))

    w = weights
    consensus = w.docking * dz + w.physics * pz + w.ml * mz

    frame = pd.DataFrame(
        {
            "name": names,
            "docking": np.asarray(docking, dtype=float),
            "physics": np.asarray(physics, dtype=float),
            "ml": np.asarray(ml, dtype=float),
            "docking_z": dz,
            "physics_z": pz,
            "ml_z": mz,
            "consensus": consensus,
        }
    )
    logger.info(
        "Built consensus table for %d ligands (weights d=%.2f, p=%.2f, m=%.2f)",
        len(names),
        w.docking,
        w.physics,
        w.ml,
    )
    return ConsensusResult(frame=frame)
