"""AutoDock Vina docking engine.

Uses the Python ``vina`` package when available. This is an *optional*
dependency; the ``CachedEngine`` is the default for test and CI environments
where installing Vina's C++ core would be heavyweight. The interface is
deliberately small: the engine takes a prepared ligand + receptor PDBQT +
search box and returns the best-pose affinity in kcal/mol.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from btk_aidd.data.ligands import PreparedLigand
from btk_aidd.data.receptor import DockingBox
from btk_aidd.docking.engine import DockingEngine, DockingResult
from btk_aidd.logger import get_logger

logger = get_logger(__name__)


class VinaEngine(DockingEngine):
    """AutoDock Vina wrapper."""

    def __init__(
        self,
        exhaustiveness: int = 16,
        num_poses: int = 5,
        cpu: int | None = None,
        seed: int = 42,
    ) -> None:
        try:
            from vina import Vina  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "The 'vina' Python package is required for VinaEngine. "
                "Install with: pip install vina"
            ) from exc

        self._Vina = Vina
        self._exhaustiveness = exhaustiveness
        self._num_poses = num_poses
        self._cpu = cpu or max(1, (os.cpu_count() or 2) - 1)
        self._seed = seed

    def dock(
        self,
        ligand: PreparedLigand,
        receptor_pdbqt: str | Path,
        box: DockingBox,
    ) -> DockingResult:
        receptor_path = Path(receptor_pdbqt)
        if not receptor_path.exists():
            return DockingResult(
                name=ligand.name,
                affinity_kcal_mol=float("nan"),
                pose_sdf_path=None,
                success=False,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ligand_pdbqt = tmp / f"{_safe(ligand.name)}.pdbqt"
            out_pdbqt = tmp / f"{_safe(ligand.name)}_out.pdbqt"

            if not _write_ligand_pdbqt(ligand, ligand_pdbqt):
                return DockingResult(
                    name=ligand.name,
                    affinity_kcal_mol=float("nan"),
                    pose_sdf_path=None,
                    success=False,
                )

            try:
                v = self._Vina(sf_name="vina", cpu=self._cpu, seed=self._seed)
                v.set_receptor(str(receptor_path))
                v.set_ligand_from_file(str(ligand_pdbqt))
                v.compute_vina_maps(center=list(box.center), box_size=list(box.size))
                v.dock(exhaustiveness=self._exhaustiveness, n_poses=self._num_poses)
                v.write_poses(str(out_pdbqt), n_poses=self._num_poses, overwrite=True)
                energies = v.energies(n_poses=self._num_poses)
            except (RuntimeError, ValueError) as exc:
                logger.warning("Vina failed for %s: %s", ligand.name, exc)
                return DockingResult(
                    name=ligand.name,
                    affinity_kcal_mol=float("nan"),
                    pose_sdf_path=None,
                    success=False,
                )

            if not energies:
                return DockingResult(
                    name=ligand.name,
                    affinity_kcal_mol=float("nan"),
                    pose_sdf_path=None,
                    success=False,
                )

            best_affinity = float(energies[0][0])
            return DockingResult(
                name=ligand.name,
                affinity_kcal_mol=best_affinity,
                pose_sdf_path=None,  # callers can re-generate SDF from PDBQT if needed
                success=True,
            )


def _safe(name: str) -> str:
    """Sanitise a name for use in temp filenames."""
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in name)


def _write_ligand_pdbqt(ligand: PreparedLigand, path: Path) -> bool:
    """Minimal SDF->PDBQT conversion using RDKit + manual PDB→PDBQT.

    This avoids a Meeko dependency at the cost of skipping advanced Gasteiger
    charge assignment. For production use, replace with Meeko's
    ``MoleculePreparation``; for benchmark virtual screening on standard drug-
    like molecules the RDKit-derived charges are adequate.
    """
    try:
        # Ensure hydrogens and 3D coords are present.
        mol = ligand.mol
        if mol.GetNumConformers() == 0:
            return False
        AllChem.ComputeGasteigerCharges(mol)
        pdb_block = Chem.MolToPDBBlock(mol)
    except (RuntimeError, ValueError):
        return False

    # Vina's PDBQT format is PDB + per-atom charge + autodock atom type.
    # For virtual screening we accept PDB atom types as a pragmatic fallback
    # when Meeko is not installed; users wanting rigorous scoring should use
    # Meeko's MoleculePreparation.write_pdbqt_string.
    lines_out: list[str] = []
    for line in pdb_block.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            # Pad PDB line to 80 cols and tack on occupancy/bfactor/charge/type.
            atom_symbol = line[76:78].strip() or line[12:14].strip()
            at_type = _autodock_atom_type(atom_symbol)
            padded = line.ljust(66) + "0.00      " + at_type.rjust(2)
            lines_out.append(padded)
        elif line.startswith(("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")) or line.startswith(("END", "TER")):
            lines_out.append(line)
    # Bracket ligand atoms in a single ROOT for rigid docking fallback.
    if not any(line.startswith("ROOT") for line in lines_out):
        atom_lines = [lo for lo in lines_out if lo.startswith(("ATOM", "HETATM"))]
        other = [lo for lo in lines_out if lo.startswith(("END", "TER"))]
        rotatable = _count_rotatable(ligand.mol)
        lines_out = ["ROOT", *atom_lines, "ENDROOT", f"TORSDOF {rotatable}", *other]

    path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return True


def _autodock_atom_type(symbol: str) -> str:
    """Map an element symbol to an AutoDock Vina atom type."""
    mapping = {
        "C": "C",
        "N": "NA",
        "O": "OA",
        "S": "SA",
        "H": "HD",
        "F": "F",
        "Cl": "Cl",
        "Br": "Br",
        "I": "I",
        "P": "P",
    }
    return mapping.get(symbol, symbol or "C")


def _count_rotatable(mol: Chem.Mol) -> int:
    try:
        from rdkit.Chem import Lipinski

        return int(Lipinski.NumRotatableBonds(mol))
    except ImportError:  # pragma: no cover
        return 0
