"""Mode-of-action filtering.

A docking pipeline that blindly trusts every ChEMBL "active" mis-ranks
compounds that bind BTK but do **not** inhibit it — e.g. allosteric
activators, cryptic-site stabilisers, or screening artefacts. ChEMBL tags
each measurement with an *assay type* (B = binding, F = functional,
A = ADME, T = toxicity) and a human-readable ``activity_comment``.

This module extracts that information and returns:

1. **A cleaned training set** that preserves only measurements consistent
   with an inhibition mechanism of action.
2. **A per-compound MoA-confidence score** reflecting how strongly the
   underlying evidence supports "this molecule inhibits BTK" (as opposed
   to "this molecule binds BTK somehow").

The confidence score is used downstream to discount the ML rescorer's
output for compounds trained on weak evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from btk_aidd.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MoAFilter:
    """Threshold parameters for mode-of-action filtering."""

    keep_assay_types: frozenset[str] = frozenset({"B", "F"})
    reject_comments: frozenset[str] = frozenset(
        {
            "not active",
            "inconclusive",
            "no data",
            "insufficient data",
            "unreliable",
        }
    )
    # When an activity_comment mentions an agonist / activator mechanism,
    # we drop the row even if the raw potency is strong.
    reject_keywords: frozenset[str] = frozenset(
        {
            "agonist",
            "activator",
            "enhancer",
            "potentiator",
        }
    )


def filter_actives_by_moa(
    df: pd.DataFrame,
    moa: MoAFilter | None = None,
) -> pd.DataFrame:
    """Return the subset of ``df`` consistent with BTK inhibition.

    The function is tolerant: if a ``assay_type`` or ``activity_comment``
    column is missing (e.g. the cached fixture CSV), it simply returns the
    input unchanged and logs a warning. This keeps the pipeline backwards-
    compatible with v0.1 fixtures.

    Args:
        df: ChEMBL-schema DataFrame (as produced by
            :func:`btk_aidd.data.chembl.load_cached`).
        moa: Optional :class:`MoAFilter` override.

    Returns:
        A filtered DataFrame. Rows with no ``assay_type`` / ``activity_comment``
        columns are kept (treated as "unknown but not rejected").
    """
    moa = moa or MoAFilter()
    total = len(df)
    if df.empty:
        return df

    work = df.copy()
    filters_applied = []

    if "assay_type" in work.columns:
        before = len(work)
        work = work[work["assay_type"].isin(moa.keep_assay_types) | work["assay_type"].isna()]
        filters_applied.append(("assay_type", before - len(work)))

    if "activity_comment" in work.columns:
        before = len(work)
        comments = work["activity_comment"].fillna("").str.lower()
        reject_mask = comments.apply(
            lambda s: (s in moa.reject_comments)
            or any(kw in s for kw in moa.reject_keywords)
        )
        work = work[~reject_mask]
        filters_applied.append(("activity_comment", before - len(work)))

    if not filters_applied:
        logger.warning(
            "MoA filter: no 'assay_type' or 'activity_comment' columns found — "
            "returning input unchanged"
        )
        return work

    work = work.reset_index(drop=True)
    summary = ", ".join(f"{k} dropped {n}" for k, n in filters_applied)
    logger.info("MoA filter: %d -> %d rows (%s)", total, len(work), summary)
    return work


def moa_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a ``moa_confidence`` column to an actives DataFrame.

    The confidence is a simple additive score:

    * 0.6 if ``assay_type == 'F'`` (functional / enzymatic assay)
    * 0.3 if ``assay_type == 'B'`` (binding assay only)
    * 0.0 otherwise

    * +0.3 if the ``activity_comment`` explicitly mentions "inhibitor",
      "inhibition", or "antagonist".
    * +0.1 per measurement if ``pchembl_value`` >= 8 (strong potency).

    Clipped to ``[0, 1]``.

    Missing columns are treated as zero contribution (the absence of
    evidence is not evidence of absence).
    """
    if df.empty:
        return df.assign(moa_confidence=pd.Series(dtype=float))

    work = df.copy()
    scores = pd.Series(0.0, index=work.index)

    if "assay_type" in work.columns:
        scores = scores + work["assay_type"].map({"F": 0.6, "B": 0.3}).fillna(0.0)

    if "activity_comment" in work.columns:
        comments = work["activity_comment"].fillna("").str.lower()
        inhibitor_mentioned = comments.str.contains(
            "inhibitor|inhibition|antagonist", regex=True, na=False
        )
        scores = scores + 0.3 * inhibitor_mentioned.astype(float)

    if "pchembl_value" in work.columns:
        strong = (work["pchembl_value"].fillna(0.0) >= 8.0).astype(float)
        scores = scores + 0.1 * strong

    work["moa_confidence"] = scores.clip(lower=0.0, upper=1.0)
    return work
