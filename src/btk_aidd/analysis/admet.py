"""ADMET / drug-likeness analysis.

Every high-ranked docking hit must still clear a drug-likeness gate before
it is worth synthesising. This module computes, per ligand, the set of
rules and scores that medicinal chemists actually look at:

=========================  ================================================
rule / score                what it checks
=========================  ================================================
Lipinski "Rule of 5"        MW ≤ 500, logP ≤ 5, HBA ≤ 10, HBD ≤ 5
Veber's rules               rotatable bonds ≤ 10, TPSA ≤ 140 Å²
Ghose filter                MW 160–480, logP −0.4 to +5.6, atoms 20–70
QED                         Bickerton 2012 quantitative drug-likeness [0,1]
SA score                    Ertl–Schuffenhauer synthetic accessibility [1,10]
PAINS filter                Pan-assay interference alerts (Baell 2010)
Brenk alerts                structural alerts (toxic / reactive groups)
=========================  ================================================

References:
    * Lipinski, C. A. *et al.* Experimental and computational approaches to
      estimate solubility and permeability. *Adv. Drug Del. Rev.* 1997, 23.
    * Veber, D. F. *et al.* Molecular properties that influence the oral
      bioavailability of drug candidates. *J. Med. Chem.* 2002, 45, 2615.
    * Bickerton, G. R. *et al.* Quantifying the chemical beauty of drugs.
      *Nat. Chem.* 2012, 4, 90-98.
    * Baell, J. B.; Holloway, G. A. New substructure filters for removal of
      pan assay interference compounds (PAINS). *J. Med. Chem.* 2010, 53.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from rdkit import Chem, RDConfig
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from btk_aidd.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ADMETReport:
    """Per-ligand drug-likeness profile."""

    name: str
    # raw descriptors
    mw: float
    logp: float
    hba: int
    hbd: int
    rotatable: int
    tpsa: float
    heavy_atoms: int
    # rules
    lipinski_violations: int
    passes_lipinski: bool
    passes_veber: bool
    passes_ghose: bool
    # scores
    qed: float  # 0-1, higher = more drug-like
    sa_score: float  # 1-10, lower = easier to synthesise
    # alerts
    pains_alerts: tuple[str, ...]
    brenk_alerts: tuple[str, ...]
    # summary
    drug_likeness: float  # consolidated 0-1, see ``overall_drug_likeness``


def compute_admet(name: str, smiles: str) -> ADMETReport | None:
    """Compute the full ADMET report for one ligand.

    Args:
        name: Stable identifier.
        smiles: SMILES string.

    Returns:
        An :class:`ADMETReport`, or ``None`` if the SMILES is unparseable.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Unparseable SMILES for %s: %s", name, smiles)
        return None

    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    rotatable = int(Lipinski.NumRotatableBonds(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    heavy = int(Descriptors.HeavyAtomCount(mol))

    # ---- rules ----
    lipinski_breaches = sum(
        [
            mw > 500,
            logp > 5,
            hba > 10,
            hbd > 5,
        ]
    )
    passes_lipinski = lipinski_breaches <= 1  # at most one breach allowed
    passes_veber = rotatable <= 10 and tpsa <= 140.0
    passes_ghose = (160 <= mw <= 480) and (-0.4 <= logp <= 5.6) and (20 <= heavy <= 70)

    # ---- scores ----
    try:
        qed_value = float(QED.qed(mol))
    except (ValueError, RuntimeError):
        qed_value = 0.0

    sa_value = _synthetic_accessibility(mol)

    # ---- alerts ----
    pains = _run_filter_catalog(mol, FilterCatalogParams.FilterCatalogs.PAINS)
    brenk = _run_filter_catalog(mol, FilterCatalogParams.FilterCatalogs.BRENK)

    drug_likeness = _overall_drug_likeness(
        passes_lipinski=passes_lipinski,
        passes_veber=passes_veber,
        qed=qed_value,
        sa_score=sa_value,
        pains_hits=len(pains),
    )

    return ADMETReport(
        name=name,
        mw=mw,
        logp=logp,
        hba=hba,
        hbd=hbd,
        rotatable=rotatable,
        tpsa=tpsa,
        heavy_atoms=heavy,
        lipinski_violations=int(lipinski_breaches),
        passes_lipinski=bool(passes_lipinski),
        passes_veber=bool(passes_veber),
        passes_ghose=bool(passes_ghose),
        qed=qed_value,
        sa_score=sa_value,
        pains_alerts=tuple(pains),
        brenk_alerts=tuple(brenk),
        drug_likeness=drug_likeness,
    )


def compute_many(items: list[tuple[str, str]]) -> list[ADMETReport]:
    """Batch-compute ADMET reports for a list of ``(name, smiles)`` pairs."""
    reports: list[ADMETReport] = []
    for name, smi in items:
        r = compute_admet(name, smi)
        if r is not None:
            reports.append(r)
    return reports


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _overall_drug_likeness(
    *,
    passes_lipinski: bool,
    passes_veber: bool,
    qed: float,
    sa_score: float,
    pains_hits: int,
) -> float:
    """Fold five signals into a 0-1 drug-likeness summary.

    Weights reflect medicinal-chemistry intuition:

    * QED (0.4) — the most information-dense descriptor.
    * Lipinski pass (0.2) — coarse oral-bioavailability gate.
    * Veber pass (0.15) — permeability-relevant.
    * Synthetic accessibility (0.15) — harder synthesis discounts the score
      (normalised so SA=3 → 1.0 and SA=8 → 0.0).
    * PAINS penalty (0.1) — subtracted linearly up to two hits.

    Returns a float clipped to [0, 1].
    """
    sa_component = max(0.0, min(1.0, (8.0 - sa_score) / 5.0))  # 3→1.0, 8→0.0
    pains_component = max(0.0, 1.0 - 0.5 * pains_hits)

    score = (
        0.40 * qed
        + 0.20 * (1.0 if passes_lipinski else 0.0)
        + 0.15 * (1.0 if passes_veber else 0.0)
        + 0.15 * sa_component
        + 0.10 * pains_component
    )
    return max(0.0, min(1.0, score))


class _SAScorer:
    """Lazy loader for the Ertl-Schuffenhauer SA-score model.

    RDKit ships ``sascorer.py`` as a contrib module rather than a first-class
    import, so we load it the first time it's needed and cache the resulting
    scorer. The ``calculate_score`` function is the public entry point.
    """

    _module: ClassVar[object | None] = None

    @classmethod
    def score(cls, mol: Chem.Mol) -> float:
        if cls._module is None:
            import importlib.util
            import sys
            from pathlib import Path

            contrib_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
            if not contrib_path.exists():
                return 5.0  # graceful fallback ≈ "medium difficulty"
            spec = importlib.util.spec_from_file_location("sascorer", contrib_path)
            if spec is None or spec.loader is None:
                return 5.0
            module = importlib.util.module_from_spec(spec)
            sys.modules["sascorer"] = module
            spec.loader.exec_module(module)
            cls._module = module
        try:
            return float(cls._module.calculateScore(mol))  # type: ignore[attr-defined]
        except (ValueError, RuntimeError):
            return 5.0


def _synthetic_accessibility(mol: Chem.Mol) -> float:
    """Return the Ertl-Schuffenhauer SA score (1 = trivial, 10 = very hard)."""
    return _SAScorer.score(mol)


def _run_filter_catalog(
    mol: Chem.Mol,
    catalog: FilterCatalogParams.FilterCatalogs,
) -> list[str]:
    """Return substructure alerts for ``mol`` from the given RDKit filter catalog."""
    params = FilterCatalogParams()
    params.AddCatalog(catalog)
    fc = FilterCatalog(params)
    matches: list[str] = []
    for entry in fc.GetMatches(mol):
        matches.append(entry.GetDescription())
    return matches
