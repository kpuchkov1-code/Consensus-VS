"""Matplotlib / seaborn visualisations for the evaluation stage.

All plotting functions take a destination path and return it so that calling
code can reliably reference the written file. None of the plotting functions
call :func:`plt.show` — this module is designed for headless execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw

from btk_aidd.logger import get_logger
from btk_aidd.metrics.enrichment import ScorerReport

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", context="notebook")


def plot_roc_curves(
    reports: Sequence[ScorerReport],
    dest: str | Path,
) -> Path:
    """Plot overlaid ROC curves for multiple scorers.

    Args:
        reports: A sequence of :class:`ScorerReport` from the metrics module.
        dest: Destination PNG path.

    Returns:
        The destination path.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for report in reports:
        ax.plot(
            report.fpr,
            report.tpr,
            label=f"{report.name} (AUC = {report.roc_auc:.3f})",
            linewidth=2,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — BTK virtual screening")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=True)
    return _save(fig, dest)


def plot_enrichment_bars(
    reports: Sequence[ScorerReport],
    fractions: Sequence[float],
    dest: str | Path,
) -> Path:
    """Grouped bar chart of enrichment factors.

    Args:
        reports: Scorer reports to compare.
        fractions: Fractions used (e.g. ``[0.01, 0.05, 0.10]``).
        dest: Destination PNG path.
    """
    rows: list[dict[str, float | str]] = []
    for report in reports:
        for f in fractions:
            rows.append(
                {
                    "scorer": report.name,
                    "fraction": f"EF@{round(f * 100)}%",
                    "enrichment": float(report.enrichment.get(f, float("nan"))),
                }
            )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=df,
        x="fraction",
        y="enrichment",
        hue="scorer",
        ax=ax,
    )
    ax.axhline(1.0, linestyle="--", color="grey", linewidth=1)
    ax.set_ylabel("Enrichment factor (× random)")
    ax.set_xlabel("Top fraction")
    ax.set_title("Enrichment factor — BTK virtual screening")
    ax.legend(title="Scorer", loc="upper right", frameon=True)
    return _save(fig, dest)


def plot_top_hits_grid(
    smiles_by_name: dict[str, str],
    ranked_names: Sequence[str],
    dest: str | Path,
    top_n: int = 10,
) -> Path:
    """Render a 2D structure grid of the top-ranked ligands.

    Args:
        smiles_by_name: Mapping from ligand name -> SMILES.
        ranked_names: Ligand names sorted from best to worst.
        dest: Destination PNG path.
        top_n: Number of structures to include.
    """
    names = list(ranked_names)[:top_n]
    mols: list[Chem.Mol] = []
    legends: list[str] = []
    for name in names:
        smi = smiles_by_name.get(name)
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mols.append(mol)
        legends.append(name)

    if not mols:
        logger.warning("No renderable ligands for top-hits grid")
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No ligands", ha="center", va="center")
        ax.axis("off")
        return _save(fig, dest)

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(5, len(mols)),
        subImgSize=(300, 300),
        legends=legends,
    )

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest_path)
    logger.info("Wrote top-hits grid to %s", dest_path)
    return dest_path


def _save(fig: plt.Figure, dest: str | Path) -> Path:
    """Tight-layout + save helper with directory creation."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote figure to %s", dest_path)
    return dest_path
