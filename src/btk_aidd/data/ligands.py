"""Ligand preparation: SMILES -> embedded, MMFF-minimised 3D conformer.

RDKit's ETKDG (Experimental Torsion Distance Geometry) is used for initial
embedding, followed by MMFF94 force-field optimisation. All operations are
seeded for reproducibility.

The :class:`LigandPreparer` is intentionally stateless; it only holds the
configuration so that the same preparer can be reused for actives and decoys
without shared mutable state leaking between batches.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem

from btk_aidd.config import LigandPrepConfig
from btk_aidd.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreparedLigand:
    """A 3D-prepared ligand ready for docking or scoring."""

    name: str
    smiles: str
    mol: Chem.Mol
    mmff_energy_kcal_mol: float
    embed_success: bool
    minimise_success: bool


class LigandPreparer:
    """Convert SMILES into MMFF-minimised 3D RDKit molecules."""

    def __init__(self, config: LigandPrepConfig) -> None:
        self._cfg = config

    def prepare(self, name: str, smiles: str) -> PreparedLigand | None:
        """Embed + optimise a single SMILES.

        Args:
            name: A stable identifier (e.g. ChEMBL ID, decoy index).
            smiles: SMILES string.

        Returns:
            A :class:`PreparedLigand` on success, or ``None`` if the SMILES
            could not be parsed or embedded.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Unparseable SMILES for %s: %s", name, smiles)
            return None

        mol = Chem.AddHs(mol)

        embed_success = self._embed(mol)
        if not embed_success:
            logger.warning("Embedding failed for %s (%s)", name, smiles)
            return PreparedLigand(
                name=name,
                smiles=smiles,
                mol=mol,
                mmff_energy_kcal_mol=float("nan"),
                embed_success=False,
                minimise_success=False,
            )

        energy, minimise_success = self._minimise(mol)
        return PreparedLigand(
            name=name,
            smiles=smiles,
            mol=mol,
            mmff_energy_kcal_mol=energy,
            embed_success=True,
            minimise_success=minimise_success,
        )

    def prepare_many(self, items: list[tuple[str, str]]) -> list[PreparedLigand]:
        """Prepare a batch of ``(name, smiles)`` tuples, skipping failures."""
        prepared: list[PreparedLigand] = []
        for name, smi in items:
            result = self.prepare(name, smi)
            if result is not None and result.embed_success:
                prepared.append(result)
        logger.info("Prepared %d / %d ligands", len(prepared), len(items))
        return prepared

    # ------------------------------------------------------------------ helpers

    def _embed(self, mol: Chem.Mol) -> bool:
        params = AllChem.ETKDGv3()
        params.randomSeed = self._cfg.embed_seed
        params.useSmallRingTorsions = True
        try:
            cid = AllChem.EmbedMolecule(mol, params)
        except (RuntimeError, ValueError):
            return False
        if cid < 0:
            # fallback: relax geometry constraints
            params.useRandomCoords = True
            try:
                cid = AllChem.EmbedMolecule(mol, params)
            except (RuntimeError, ValueError):
                return False
        return cid >= 0

    def _minimise(self, mol: Chem.Mol) -> tuple[float, bool]:
        """Run MMFF94 minimisation and return ``(energy_kcal_mol, converged)``."""
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=self._cfg.mmff_variant)
        except (ValueError, RuntimeError):
            return float("nan"), False
        if props is None:
            return float("nan"), False

        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        if ff is None:
            return float("nan"), False

        converged_code = ff.Minimize(maxIts=self._cfg.max_mmff_iters)
        # RDKit returns 0 on convergence, 1 otherwise.
        return float(ff.CalcEnergy()), converged_code == 0
