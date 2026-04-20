"""Cached docking engine.

Reads Vina-style affinity scores from a CSV and returns them verbatim. This is
the default engine for CI, unit tests, and for the ``--fast`` pipeline mode
where re-docking every round would be wasteful.

CSV schema (header required)::

    name,affinity_kcal_mol[,pose_sdf_path]

``pose_sdf_path`` is optional; if absent, :class:`DockingResult.pose_sdf_path`
is ``None``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from btk_aidd.data.ligands import PreparedLigand
from btk_aidd.data.receptor import DockingBox
from btk_aidd.docking.engine import DockingEngine, DockingResult
from btk_aidd.logger import get_logger

logger = get_logger(__name__)


class CachedEngine(DockingEngine):
    """Look up docking affinities from a pre-computed CSV."""

    def __init__(self, cache_csv: str | Path) -> None:
        path = Path(cache_csv)
        if not path.exists():
            raise FileNotFoundError(f"Docking cache not found: {path}")
        df = pd.read_csv(path)
        required = {"name", "affinity_kcal_mol"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Cache {path} missing columns: {sorted(missing)}")
        self._scores: dict[str, float] = dict(
            zip(df["name"].astype(str), df["affinity_kcal_mol"].astype(float), strict=True)
        )
        self._pose_paths: dict[str, str | None] = {}
        if "pose_sdf_path" in df.columns:
            for name, pose in zip(df["name"].astype(str), df["pose_sdf_path"], strict=True):
                if isinstance(pose, str) and pose.strip():
                    self._pose_paths[name] = pose
        logger.info("CachedEngine loaded %d scores from %s", len(self._scores), path)

    def dock(
        self,
        ligand: PreparedLigand,
        receptor_pdbqt: str | Path,  # unused
        box: DockingBox,  # unused
    ) -> DockingResult:
        del receptor_pdbqt, box  # explicitly ignored
        score = self._scores.get(ligand.name)
        if score is None or math.isnan(score):
            return DockingResult(
                name=ligand.name,
                affinity_kcal_mol=float("nan"),
                pose_sdf_path=None,
                success=False,
            )
        pose = self._pose_paths.get(ligand.name)
        return DockingResult(
            name=ligand.name,
            affinity_kcal_mol=float(score),
            pose_sdf_path=Path(pose) if pose else None,
            success=True,
        )

    @property
    def size(self) -> int:
        """Number of cached ligands — useful for tests and diagnostics."""
        return len(self._scores)
