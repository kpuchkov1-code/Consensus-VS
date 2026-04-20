"""Receptor preparation and pocket definition.

This module handles the protein side of the pipeline: parsing a PDB file,
stripping waters / cofactors, extracting the reference ligand's binding-site
centroid, and computing a docking box that encloses the pocket with a
configurable padding.

PDB writing is done through Biopython only, so the module has no OpenMM /
PDBFixer requirement. For production use (full hydrogen network, missing
side-chain reconstruction) the user can pre-process the PDB with PDBFixer
externally; the defaults here are adequate for Vina docking since Vina itself
rebuilds the polar-hydrogen network on PDBQT conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser, Select

from btk_aidd.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DockingBox:
    """Axis-aligned bounding box for AutoDock Vina."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))


@dataclass(frozen=True)
class PocketDefinition:
    """Pocket geometry derived from a reference co-crystallised ligand."""

    centroid: tuple[float, float, float]
    atom_coords: np.ndarray  # shape (n_atoms, 3)
    box: DockingBox


class _KeepProteinSelect(Select):
    """Biopython selector that keeps only standard protein residues + optionally the reference ligand."""

    def __init__(self, keep_ligand_resname: str | None) -> None:
        self._keep_ligand = keep_ligand_resname

    def accept_residue(self, residue: object) -> int:  # type: ignore[override]
        # ``residue.id`` is a tuple: (hetero_flag, resseq, icode). Standard residues
        # have hetero_flag == ' '. We reject waters and all other hetatm unless
        # the residue name matches the reference ligand we want to keep.
        hetero_flag, _, _ = residue.id  # type: ignore[attr-defined]
        resname: str = residue.resname.strip()  # type: ignore[attr-defined]
        if hetero_flag == " ":
            return 1
        if resname == "HOH" or resname == "WAT":
            return 0
        if self._keep_ligand is not None and resname == self._keep_ligand.upper():
            return 1
        return 0


class ReceptorPreparer:
    """Clean a PDB file for docking and extract the ligand pocket."""

    def __init__(self, pdb_id: str, reference_ligand_resname: str) -> None:
        self._pdb_id = pdb_id
        self._ref_resname = reference_ligand_resname.upper()

    def clean(self, input_pdb: str | Path, output_pdb: str | Path) -> Path:
        """Strip waters / non-reference HETATM and write a cleaned PDB.

        Args:
            input_pdb: Path to the raw PDB (downloaded from RCSB).
            output_pdb: Destination path for the cleaned PDB.

        Returns:
            The output path.
        """
        in_path = Path(input_pdb)
        out_path = Path(output_pdb)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(self._pdb_id, str(in_path))

        io = PDBIO()
        io.set_structure(structure)
        io.save(str(out_path), select=_KeepProteinSelect(self._ref_resname))
        logger.info("Wrote cleaned receptor -> %s", out_path)
        return out_path


def pocket_from_reference_ligand(
    pdb_path: str | Path,
    reference_ligand_resname: str,
    box_padding: float = 8.0,
) -> PocketDefinition:
    """Extract reference-ligand atoms and build an enclosing docking box.

    Args:
        pdb_path: PDB containing the reference ligand as a HETATM residue.
        reference_ligand_resname: Three-letter residue name of the reference
            ligand (e.g. ``"1E8"`` for ibrutinib in 4OT6).
        box_padding: Angstroms added to every side of the ligand bounding box.

    Returns:
        A :class:`PocketDefinition` with centroid, atom coords, and
        :class:`DockingBox`.

    Raises:
        ValueError: If no atoms are found for the reference residue.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("_", str(pdb_path))

    coords: list[list[float]] = []
    target = reference_ligand_resname.upper()
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.resname.strip().upper() != target:
                    continue
                for atom in residue:
                    coords.append(list(atom.coord))

    if not coords:
        raise ValueError(
            f"Reference ligand '{reference_ligand_resname}' not found in {pdb_path}"
        )

    coord_array = np.asarray(coords, dtype=float)
    centroid = coord_array.mean(axis=0)
    mins = coord_array.min(axis=0) - box_padding
    maxs = coord_array.max(axis=0) + box_padding
    size = maxs - mins

    return PocketDefinition(
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        atom_coords=coord_array,
        box=DockingBox(
            center=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            size=(float(size[0]), float(size[1]), float(size[2])),
        ),
    )
