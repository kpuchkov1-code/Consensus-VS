"""Property-matched decoy generator (DUD-E style).

Given a set of active SMILES, produce a pool of decoys that are

1. **Chemically similar** in bulk properties (MW, logP, HBA, HBD, rotatable
   bonds, formal charge) to at least one active within configurable absolute
   windows, and
2. **Topologically dissimilar** from every active (Tanimoto below a cutoff on
   Morgan fingerprints).

The decoy candidate pool is supplied as a list of SMILES (e.g. a random sample
from ChEMBL or a local ZINC subset). The generator is deterministic under a
fixed random seed.

This follows the published DUD-E recipe (Mysinger *et al.*, *J. Med. Chem.*
2012, 55, 14) at a high level; it is intentionally more permissive so that a
few hundred actives can yield thousands of decoys without requiring the full
ZINC benchmark download.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator

from btk_aidd.config import PropertyWindows
from btk_aidd.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LigandProperties:
    """Bulk physicochemical descriptors of a small molecule."""

    mw: float
    logp: float
    hba: int
    hbd: int
    rotatable: int
    charge: int


def compute_properties(mol: Chem.Mol) -> LigandProperties:
    """Compute DUD-E-style matching descriptors from an RDKit mol.

    Args:
        mol: A sanitised RDKit molecule.

    Returns:
        A :class:`LigandProperties` instance.
    """
    return LigandProperties(
        mw=float(Descriptors.MolWt(mol)),
        logp=float(Crippen.MolLogP(mol)),
        hba=int(Lipinski.NumHAcceptors(mol)),
        hbd=int(Lipinski.NumHDonors(mol)),
        rotatable=int(Lipinski.NumRotatableBonds(mol)),
        charge=int(Chem.GetFormalCharge(mol)),
    )


def _morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> DataStructs.ExplicitBitVect:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprint(mol)


@dataclass(frozen=True)
class DecoyGeneratorConfig:
    """Runtime parameters for :class:`DecoyGenerator`.

    These are populated from :class:`btk_aidd.config.DecoysConfig` by the
    pipeline; tests may instantiate directly.
    """

    windows: PropertyWindows
    similarity_cutoff: float
    random_seed: int
    fp_radius: int = 2
    fp_n_bits: int = 2048


class DecoyGenerator:
    """Selects property-matched, topology-dissimilar decoys from a candidate pool."""

    def __init__(self, config: DecoyGeneratorConfig) -> None:
        self._cfg = config
        self._rng = np.random.default_rng(config.random_seed)

    def generate(
        self,
        actives_smiles: Sequence[str],
        candidate_smiles: Iterable[str],
        count: int,
    ) -> list[str]:
        """Pick up to ``count`` decoy SMILES from ``candidate_smiles``.

        Args:
            actives_smiles: SMILES of active compounds (defines match windows).
            candidate_smiles: Candidate pool (e.g. random ChEMBL sample).
            count: Maximum number of decoys to return.

        Returns:
            A deterministic, de-duplicated list of decoy SMILES.
        """
        active_mols = [self._parse(s) for s in actives_smiles]
        active_mols = [m for m in active_mols if m is not None]
        if not active_mols:
            raise ValueError("No parseable actives supplied")

        active_props = [compute_properties(m) for m in active_mols]
        active_fps = [_morgan_fp(m, self._cfg.fp_radius, self._cfg.fp_n_bits) for m in active_mols]

        collected: list[str] = []
        seen: set[str] = set()
        for cand_smi in candidate_smiles:
            if len(collected) >= count:
                break
            cand_mol = self._parse(cand_smi)
            if cand_mol is None:
                continue
            cand_smi_canon = Chem.MolToSmiles(cand_mol)
            if cand_smi_canon in seen:
                continue
            seen.add(cand_smi_canon)

            cand_props = compute_properties(cand_mol)
            if not self._matches_any(cand_props, active_props):
                continue

            cand_fp = _morgan_fp(cand_mol, self._cfg.fp_radius, self._cfg.fp_n_bits)
            max_sim = max(
                (DataStructs.TanimotoSimilarity(cand_fp, af) for af in active_fps),
                default=0.0,
            )
            if max_sim >= self._cfg.similarity_cutoff:
                continue

            collected.append(cand_smi_canon)

        logger.info("Selected %d / %d requested decoys", len(collected), count)
        return collected

    # ------------------------------------------------------------------ helpers

    def _matches_any(
        self,
        cand: LigandProperties,
        actives: Sequence[LigandProperties],
    ) -> bool:
        """Return True iff ``cand`` is within all property windows of at least one active."""
        w = self._cfg.windows
        for act in actives:
            if (
                abs(cand.mw - act.mw) <= w.mw
                and abs(cand.logp - act.logp) <= w.logp
                and abs(cand.hba - act.hba) <= w.hba
                and abs(cand.hbd - act.hbd) <= w.hbd
                and abs(cand.rotatable - act.rotatable) <= w.rotatable
                and abs(cand.charge - act.charge) <= w.charge
            ):
                return True
        return False

    @staticmethod
    def _parse(smiles: str) -> Chem.Mol | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
            return None
        return mol
