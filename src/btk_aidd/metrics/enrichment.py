"""Virtual-screening evaluation metrics: ROC-AUC and enrichment factors.

Scores follow the "lower = better" convention. Internally we pass the negated
scores to scikit-learn so that higher values indicate actives (as scikit-learn
expects for ``roc_auc_score``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from btk_aidd.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ScorerReport:
    """Per-scorer evaluation summary."""

    name: str
    roc_auc: float
    enrichment: dict[float, float]  # fraction -> EF
    fpr: np.ndarray
    tpr: np.ndarray
    n_actives: int
    n_decoys: int


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC with the score sign flipped ("lower = better" convention).

    Args:
        labels: Binary ground truth (1 = active, 0 = decoy).
        scores: Predicted scores, lower values = more likely active.

    Returns:
        ROC-AUC in [0, 1]. Returns ``nan`` if only one class is present.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, -scores))


def enrichment_factor(labels: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    """Enrichment factor at ``fraction`` of the ranked database.

    EF@X% is the ratio of the active rate in the top X% to the overall active
    rate. An EF of 1.0 is random; >1 indicates early enrichment.

    Args:
        labels: Binary ground truth (1 = active, 0 = decoy).
        scores: Predicted scores, lower values = more likely active.
        fraction: Top fraction of ranked compounds to evaluate (0, 1].

    Returns:
        Enrichment factor. Returns ``nan`` if the overall active rate is 0.

    Raises:
        ValueError: If ``fraction`` is outside (0, 1].
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)
    if n == 0:
        return float("nan")

    overall_rate = float(labels.mean())
    if overall_rate == 0.0:
        return float("nan")

    top_k = max(1, int(np.ceil(fraction * n)))
    # "Lower = better" -> sort ascending, take first top_k
    order = np.argsort(scores, kind="stable")
    top_hits = int(labels[order[:top_k]].sum())
    top_rate = top_hits / top_k
    return top_rate / overall_rate


def scorer_report(
    name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    enrichment_fractions: list[float],
) -> ScorerReport:
    """Compile a full :class:`ScorerReport` for one scorer."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    auc = roc_auc(labels, scores)
    ef = {f: enrichment_factor(labels, scores, f) for f in enrichment_fractions}
    fpr, tpr, _ = roc_curve(labels, -scores)
    return ScorerReport(
        name=name,
        roc_auc=auc,
        enrichment=ef,
        fpr=fpr,
        tpr=tpr,
        n_actives=int(labels.sum()),
        n_decoys=int(len(labels) - labels.sum()),
    )
