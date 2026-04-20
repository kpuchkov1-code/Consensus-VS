"""Abstract docking-engine interface.

Every concrete engine (Vina, Smina, cached) implements :class:`DockingEngine`.
The pipeline only depends on this ABC, which keeps the engine choice a
pure configuration concern and makes the scoring layer testable without an
installed docking binary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from btk_aidd.data.ligands import PreparedLigand
from btk_aidd.data.receptor import DockingBox


@dataclass(frozen=True)
class DockingResult:
    """One ligand's docking outcome.

    Attributes:
        name: Stable ligand identifier.
        affinity_kcal_mol: Best-pose predicted binding affinity. More negative
            is better-binding. ``nan`` indicates docking failure.
        pose_sdf_path: Path to an SDF containing the top-ranked pose(s), or
            ``None`` if poses were not written (e.g. cached engine).
        success: Whether docking completed and produced at least one pose.
    """

    name: str
    affinity_kcal_mol: float
    pose_sdf_path: Path | None
    success: bool


class DockingEngine(ABC):
    """Base class for all docking engines."""

    @abstractmethod
    def dock(
        self,
        ligand: PreparedLigand,
        receptor_pdbqt: str | Path,
        box: DockingBox,
    ) -> DockingResult:
        """Dock a single prepared ligand against a receptor.

        Args:
            ligand: Output of :class:`btk_aidd.data.ligands.LigandPreparer`.
            receptor_pdbqt: Path to the receptor in PDBQT (or PDB for engines
                that accept it).
            box: Axis-aligned box defining the search space.

        Returns:
            A :class:`DockingResult`. On failure, ``success`` is False and
            ``affinity_kcal_mol`` is NaN.
        """

    def dock_many(
        self,
        ligands: list[PreparedLigand],
        receptor_pdbqt: str | Path,
        box: DockingBox,
    ) -> list[DockingResult]:
        """Dock a batch sequentially. Subclasses may override for parallelism."""
        return [self.dock(lig, receptor_pdbqt, box) for lig in ligands]
