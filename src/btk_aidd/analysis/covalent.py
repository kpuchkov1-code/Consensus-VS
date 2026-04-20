"""Covalent binding analysis for BTK (Cys481 warhead detection).

BTK's first-generation approved inhibitors (ibrutinib, acalabrutinib,
zanubrutinib) all bind covalently to **Cys481** in the ATP pocket. A
pipeline that ignores covalent chemistry systematically mis-ranks
irreversible binders: their affinity is driven by the covalent bond, not
by the non-covalent scoring function.

This module addresses that gap with two linked checks:

1. **Warhead detection** — SMARTS patterns for electrophilic groups that
   can form a Michael adduct or direct S-alkylation with a cysteine thiol.
2. **Cys481 proximity** — if a docked pose is available, the distance
   between the ligand's reactive atom and Cys481's sulfur is computed;
   warheads within ~6 Å of the sulfur are flagged as productively
   positioned.

The output is a :class:`CovalentReport` combining a binary flag, the
identified warhead class, and a bonus score (kcal/mol-like units) that
downstream consensus scoring can weight in.

References:
    * Lagoutte, R.; Patouret, R.; Winssinger, N. Covalent inhibitors: an
      opportunity for rational target selectivity. *Curr. Opin. Chem.
      Biol.* 2017, 39, 54-63.
    * Pan, Z. *et al.* Discovery of selective irreversible inhibitors for
      Bruton's tyrosine kinase. *ChemMedChem* 2007, 2, 58-61.
    * Liu, Q. *et al.* Developing irreversible inhibitors of the protein
      kinase cysteinome. *Chem. Biol.* 2013, 20, 146-159.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from btk_aidd.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Warhead library
# ---------------------------------------------------------------------------
# Each entry maps a human-readable name to a SMARTS pattern that matches
# the electrophilic substructure. Ordering matters: more specific warheads
# are listed before their more generic parents so that `_first_match()`
# returns the most informative label.

_WARHEADS: tuple[tuple[str, str], ...] = (
    # Classic Michael acceptors attached to an amide N — the "ibrutinib" warhead
    ("acrylamide", "[NX3;H0,H1][CX3](=O)[CX3]=[CX3]"),
    # Propargyl / alkyne amide (used in some 2nd-gen BTK inhibitors)
    ("propiolamide", "[NX3;H0,H1][CX3](=O)[CX2]#[CX2]"),
    # α-halo carbonyls (chloroacetamide, bromoacetamide, iodoacetamide)
    ("chloroacetamide", "[NX3;H0,H1][CX3](=O)[CH2][Cl,Br,I]"),
    # Vinyl sulfone / sulfonamide
    ("vinyl_sulfone", "[SX4](=O)(=O)[CX3]=[CX3]"),
    # Maleimide
    ("maleimide", "O=C1C=CC(=O)N1"),
    # Activated acyl (acyl fluoride / sulfonyl fluoride)
    ("acyl_fluoride", "[CX3](=O)F"),
    ("sulfonyl_fluoride", "[SX4](=O)(=O)F"),
    # Epoxide (direct alkylator)
    ("epoxide", "C1OC1"),
    # Boronic acid / ester (reversible covalent)
    ("boronic_acid", "[BX3]([OX2H])[OX2H]"),
    # General Michael acceptor as a catch-all (α,β-unsaturated carbonyl)
    ("michael_acceptor", "[CX3]=[CX3][CX3]=O"),
)


@dataclass(frozen=True)
class CovalentReport:
    """Per-ligand covalent-binding assessment."""

    name: str
    has_warhead: bool
    warhead_type: str | None
    warhead_smarts: str | None
    cys481_distance_angstroms: float | None  # None when no pose is available
    bonus_kcal_mol: float

    @property
    def is_productively_covalent(self) -> bool:
        """True when the warhead is within reactive range of Cys481.

        Returns True if a warhead is present AND either (a) the Cys481
        distance is within 6.0 Å, or (b) the distance is unknown (no pose).
        The latter is the "benefit of the doubt" path used in cached-mode
        runs, where the actual pose has not been computed.
        """
        if not self.has_warhead:
            return False
        if self.cys481_distance_angstroms is None:
            return True
        return self.cys481_distance_angstroms <= 6.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_warhead(mol: Chem.Mol) -> tuple[str, str] | None:
    """Return ``(warhead_name, warhead_smarts)`` or ``None`` if no warhead is found.

    Args:
        mol: A sanitised RDKit molecule.

    Returns:
        A tuple of the most-specific matched warhead class and its SMARTS
        pattern, or ``None`` if no warhead is present.
    """
    for name, smarts in _WARHEADS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return name, smarts
    return None


def score_covalent(
    name: str,
    smiles: str,
    cys481_distance: float | None = None,
    bonus_if_productive: float = -2.5,
) -> CovalentReport:
    """Build a :class:`CovalentReport` for a single ligand.

    Args:
        name: Stable identifier.
        smiles: SMILES of the ligand.
        cys481_distance: Optional distance (Å) from the warhead reactive atom
            to the Cys481 sulfur in a docked pose. Pass ``None`` in cached
            mode.
        bonus_if_productive: Score offset (kcal/mol-like units) applied when
            the ligand has a warhead positioned for Cys481 attack. More
            negative = stronger pull toward the top of the ranking. The
            default −2.5 kcal/mol is roughly the effective additional
            affinity conferred by a 5-day t½ on-target residence time at
            body temperature (Copeland 2006 estimate).

    Returns:
        A :class:`CovalentReport`. Molecules without a warhead return a
        zero bonus.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return CovalentReport(
            name=name,
            has_warhead=False,
            warhead_type=None,
            warhead_smarts=None,
            cys481_distance_angstroms=cys481_distance,
            bonus_kcal_mol=0.0,
        )

    match = detect_warhead(mol)
    if match is None:
        return CovalentReport(
            name=name,
            has_warhead=False,
            warhead_type=None,
            warhead_smarts=None,
            cys481_distance_angstroms=cys481_distance,
            bonus_kcal_mol=0.0,
        )

    warhead_name, warhead_smarts = match
    report = CovalentReport(
        name=name,
        has_warhead=True,
        warhead_type=warhead_name,
        warhead_smarts=warhead_smarts,
        cys481_distance_angstroms=cys481_distance,
        bonus_kcal_mol=bonus_if_productive if cys481_distance is None else 0.0,
    )
    if cys481_distance is not None:
        productive = cys481_distance <= 6.0
        report = CovalentReport(
            name=name,
            has_warhead=True,
            warhead_type=warhead_name,
            warhead_smarts=warhead_smarts,
            cys481_distance_angstroms=cys481_distance,
            bonus_kcal_mol=bonus_if_productive if productive else 0.0,
        )
    if report.bonus_kcal_mol != 0.0:
        logger.debug(
            "Covalent warhead %s on %s (distance=%s) -> bonus %.2f kcal/mol",
            warhead_name,
            name,
            cys481_distance,
            report.bonus_kcal_mol,
        )
    return report


def score_many(
    items: list[tuple[str, str]],
    cys481_distances: dict[str, float] | None = None,
    bonus_if_productive: float = -2.5,
) -> list[CovalentReport]:
    """Batch-score ``(name, smiles)`` pairs."""
    distances = cys481_distances or {}
    return [
        score_covalent(
            name=name,
            smiles=smi,
            cys481_distance=distances.get(name),
            bonus_if_productive=bonus_if_productive,
        )
        for name, smi in items
    ]
