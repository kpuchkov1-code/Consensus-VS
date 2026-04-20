"""End-to-end pipeline orchestrator.

Glues the data, docking, scoring, analysis, metric, and viz modules into a
single ``run_pipeline`` entry point. The orchestrator holds no state of its
own; it pulls configuration from :class:`btk_aidd.config.Config` and writes
all outputs under ``config.runtime.output_dir``.

Execution stages::

    load_inputs -> MoA filter -> prepare_ligands -> dock -> rescore_physics
                -> rescore_ml -> covalent -> ADMET -> selectivity
                -> consensus  -> evaluate -> visualise

Each stage is a small private function so unit tests can exercise them in
isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from btk_aidd.analysis import (
    compute_admet_many,
    filter_actives_by_moa,
    score_covalent_many,
    score_selectivity,
    train_off_target_models,
)
from btk_aidd.analysis.moa import MoAFilter
from btk_aidd.config import Config
from btk_aidd.data.ligands import LigandPreparer, PreparedLigand
from btk_aidd.docking.engine import DockingResult
from btk_aidd.docking.factory import build_engine
from btk_aidd.logger import configure, get_logger
from btk_aidd.metrics.enrichment import ScorerReport, scorer_report
from btk_aidd.scoring.consensus import consensus_score
from btk_aidd.scoring.ml import MLRescorer
from btk_aidd.scoring.physics import PhysicsRescorer
from btk_aidd.viz.plots import plot_enrichment_bars, plot_roc_curves, plot_top_hits_grid

logger = get_logger(__name__)


@dataclass(frozen=True)
class LigandRow:
    """A single input molecule (active or decoy)."""

    name: str
    smiles: str
    label: int  # 1 = active, 0 = decoy


@dataclass(frozen=True)
class PipelineOutputs:
    """Artefacts produced by a pipeline run."""

    scores_csv: Path
    roc_png: Path
    enrichment_png: Path
    top_hits_png: Path
    reports: list[ScorerReport]


def load_ligand_table(
    actives_csv: str | Path,
    decoys_csv: str | Path,
    fast_mode: bool,
    fast_actives: int,
    fast_decoys: int,
    rng_seed: int,
    moa_filter: MoAFilter | None = None,
) -> list[LigandRow]:
    """Load actives and decoys into a single list of :class:`LigandRow`.

    Args:
        actives_csv: CSV with at least ``molecule_chembl_id`` and
            ``canonical_smiles`` columns.
        decoys_csv: CSV with at least ``name`` and ``canonical_smiles`` columns.
        fast_mode: If True, subsample deterministically to the sizes below.
        fast_actives: Max actives in fast mode.
        fast_decoys: Max decoys in fast mode.
        rng_seed: Seed for the subsampling.
        moa_filter: Optional MoA filter applied to actives before sampling.
            Rows inconsistent with an inhibition mechanism are dropped.

    Returns:
        List of :class:`LigandRow`, actives first then decoys.
    """
    actives = pd.read_csv(actives_csv)
    decoys = pd.read_csv(decoys_csv)

    required_actives = {"molecule_chembl_id", "canonical_smiles"}
    if not required_actives.issubset(actives.columns):
        missing = sorted(required_actives - set(actives.columns))
        raise ValueError(f"Actives CSV missing columns: {missing}")

    required_decoys = {"name", "canonical_smiles"}
    if not required_decoys.issubset(decoys.columns):
        missing = sorted(required_decoys - set(decoys.columns))
        raise ValueError(f"Decoys CSV missing columns: {missing}")

    if moa_filter is not None:
        actives = filter_actives_by_moa(actives, moa_filter)

    if fast_mode:
        actives = actives.sample(
            n=min(fast_actives, len(actives)),
            random_state=rng_seed,
        ).reset_index(drop=True)
        decoys = decoys.sample(
            n=min(fast_decoys, len(decoys)),
            random_state=rng_seed,
        ).reset_index(drop=True)

    rows: list[LigandRow] = []
    for _, row in actives.iterrows():
        rows.append(
            LigandRow(
                name=str(row["molecule_chembl_id"]),
                smiles=str(row["canonical_smiles"]),
                label=1,
            )
        )
    for _, row in decoys.iterrows():
        rows.append(
            LigandRow(
                name=str(row["name"]),
                smiles=str(row["canonical_smiles"]),
                label=0,
            )
        )
    logger.info(
        "Loaded %d actives + %d decoys = %d total",
        int(actives.shape[0]),
        int(decoys.shape[0]),
        len(rows),
    )
    return rows


def run_pipeline(
    config: Config,
    actives_csv: str | Path,
    decoys_csv: str | Path,
    project_root: Path | None = None,
) -> PipelineOutputs:
    """Run the full pipeline end-to-end.

    Args:
        config: Validated :class:`Config`.
        actives_csv: Path to actives CSV.
        decoys_csv: Path to decoys CSV.
        project_root: Directory used to resolve relative config paths. If
            ``None``, the current working directory is used.

    Returns:
        A :class:`PipelineOutputs` pointing at every artefact written.
    """
    configure(config.runtime.log_level)
    project_root = project_root or Path.cwd()
    output_dir = _resolve(project_root, config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    moa_filter = (
        MoAFilter(keep_assay_types=frozenset(config.analysis.moa.keep_assay_types))
        if config.analysis.moa.enabled
        else None
    )
    rows = load_ligand_table(
        actives_csv=actives_csv,
        decoys_csv=decoys_csv,
        fast_mode=(config.runtime.mode == "fast"),
        fast_actives=config.runtime.fast_actives,
        fast_decoys=config.runtime.fast_decoys,
        rng_seed=config.data.random_seed,
        moa_filter=moa_filter,
    )

    prepared = _prepare(rows, config)

    if len(prepared) < 4:
        raise RuntimeError(
            f"Only {len(prepared)} ligands prepared; need >=4 to train/test. "
            "Check input SMILES or reduce filtering."
        )

    dock_results = _dock(prepared, config, project_root)
    prepared, dock_results, kept_labels = _filter_by_docking(prepared, dock_results, rows)

    physics_totals = _rescore_physics(prepared, config)
    ml_split = _rescore_ml(prepared, kept_labels, config)

    # Align everything to the ML test-set (that is the evaluation universe).
    test_names = ml_split.test_names
    test_indices = [prepared_index_by_name(prepared, n) for n in test_names]
    test_docking = np.array(
        [dock_results[i].affinity_kcal_mol for i in test_indices], dtype=float
    )
    test_physics = np.array([physics_totals[i] for i in test_indices], dtype=float)
    test_ml = ml_split.test_scores
    test_ml_prob = ml_split.test_probabilities
    test_labels = ml_split.test_labels
    test_smiles = [prepared[i].smiles for i in test_indices]

    consensus = consensus_score(
        names=test_names,
        docking=test_docking,
        physics=test_physics,
        ml=test_ml,
        weights=config.scoring.consensus.weights,
    ).frame
    consensus.insert(1, "label", test_labels)

    # ------------------------------------------------------------------ mechanism
    consensus = _annotate_covalent(consensus, test_names, test_smiles, config)
    consensus = _annotate_admet(consensus, test_names, test_smiles, config)
    consensus = _annotate_selectivity(
        consensus,
        test_names,
        test_smiles,
        test_ml_prob,
        config,
        project_root,
    )
    if "covalent_bonus" in consensus.columns:
        consensus["consensus_covalent"] = (
            consensus["consensus"] + consensus["covalent_bonus"]
        )

    scores_csv = output_dir / "scores.csv"
    consensus.to_csv(scores_csv, index=False)
    logger.info("Wrote scores -> %s", scores_csv)

    reports = [
        scorer_report(
            "Docking",
            test_labels,
            test_docking,
            config.evaluation.enrichment_fractions,
        ),
        scorer_report(
            "Physics",
            test_labels,
            test_physics,
            config.evaluation.enrichment_fractions,
        ),
        scorer_report(
            "ML",
            test_labels,
            test_ml,
            config.evaluation.enrichment_fractions,
        ),
        scorer_report(
            "Consensus",
            test_labels,
            consensus["consensus"].to_numpy(),
            config.evaluation.enrichment_fractions,
        ),
    ]

    roc_png = plot_roc_curves(reports, output_dir / "roc.png")
    enrichment_png = plot_enrichment_bars(
        reports, config.evaluation.enrichment_fractions, output_dir / "enrichment.png"
    )

    smiles_by_name = {row.name: row.smiles for row in rows}
    ranked_by_consensus = (
        consensus.sort_values("consensus", ascending=True)["name"].astype(str).tolist()
    )
    actives_only = [n for n in ranked_by_consensus if int(_label_of(n, consensus)) == 1]
    top_hits_png = plot_top_hits_grid(
        smiles_by_name=smiles_by_name,
        ranked_names=actives_only,
        dest=output_dir / "top_hits.png",
        top_n=config.evaluation.top_n_hits,
    )

    return PipelineOutputs(
        scores_csv=scores_csv,
        roc_png=roc_png,
        enrichment_png=enrichment_png,
        top_hits_png=top_hits_png,
        reports=reports,
    )


# ---------------------------------------------------------------------- helpers


def _prepare(rows: list[LigandRow], config: Config) -> list[PreparedLigand]:
    preparer = LigandPreparer(config.ligand_prep)
    prepared: list[PreparedLigand] = []
    for row in rows:
        result = preparer.prepare(row.name, row.smiles)
        if result is None or not result.embed_success:
            logger.warning("Skipping ligand %s (embed failed)", row.name)
            continue
        prepared.append(result)
    logger.info("Prepared %d / %d ligands", len(prepared), len(rows))
    return prepared


def _dock(
    ligands: list[PreparedLigand],
    config: Config,
    project_root: Path,
) -> list[DockingResult]:
    engine = build_engine(config.docking, project_root=project_root)
    # Cached engine ignores receptor path + box; Vina engine needs a real PDBQT.
    receptor_path = _resolve(project_root, "data/processed/receptor.pdbqt")
    box_placeholder = _default_box()
    results = engine.dock_many(ligands, receptor_path, box_placeholder)
    logger.info("Docking produced %d results", len(results))
    return results


def _filter_by_docking(
    prepared: list[PreparedLigand],
    dock_results: list[DockingResult],
    rows: list[LigandRow],
) -> tuple[list[PreparedLigand], list[DockingResult], np.ndarray]:
    """Drop ligands whose docking failed (NaN affinity)."""
    label_by_name = {row.name: row.label for row in rows}
    kept: list[int] = [
        i
        for i, result in enumerate(dock_results)
        if result.success and not np.isnan(result.affinity_kcal_mol)
    ]
    prepared_kept = [prepared[i] for i in kept]
    results_kept = [dock_results[i] for i in kept]
    labels_kept = np.array([label_by_name[prepared[i].name] for i in kept], dtype=int)
    dropped = len(prepared) - len(kept)
    if dropped > 0:
        logger.info("Dropped %d ligands with failed docking", dropped)
    return prepared_kept, results_kept, labels_kept


def _rescore_physics(ligands: list[PreparedLigand], config: Config) -> list[float]:
    rescorer = PhysicsRescorer(config.scoring.physics)
    return [rescorer.score(lig).total for lig in ligands]


def _rescore_ml(ligands: list[PreparedLigand], labels: np.ndarray, config: Config):
    rescorer = MLRescorer(config.scoring.ml)
    names = [lig.name for lig in ligands]
    smiles = [lig.smiles for lig in ligands]
    return rescorer.fit_predict(names, smiles, labels)


def prepared_index_by_name(ligands: list[PreparedLigand], name: str) -> int:
    """O(n) lookup; tiny lists so not worth a dict cache."""
    for i, lig in enumerate(ligands):
        if lig.name == name:
            return i
    raise KeyError(name)


def _label_of(name: str, frame: pd.DataFrame) -> int:
    return int(frame.loc[frame["name"].astype(str) == name, "label"].iloc[0])


def _resolve(root: Path, relative: str | Path) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else (root / p)


def _default_box():
    """Placeholder box for cached-engine mode (engine ignores it)."""
    from btk_aidd.data.receptor import DockingBox

    return DockingBox(center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0))


# ------------------------------------------------------------------ analysis


def _annotate_covalent(
    frame: pd.DataFrame,
    names: list[str],
    smiles: list[str],
    config: Config,
) -> pd.DataFrame:
    if not config.analysis.covalent.enabled:
        return frame
    reports = score_covalent_many(
        list(zip(names, smiles, strict=True)),
        bonus_if_productive=config.analysis.covalent.bonus_kcal_mol,
    )
    cov_df = pd.DataFrame(
        {
            "name": [r.name for r in reports],
            "has_warhead": [r.has_warhead for r in reports],
            "warhead_type": [r.warhead_type or "" for r in reports],
            "covalent_bonus": [r.bonus_kcal_mol for r in reports],
        }
    )
    return frame.merge(cov_df, on="name", how="left")


def _annotate_admet(
    frame: pd.DataFrame,
    names: list[str],
    smiles: list[str],
    config: Config,
) -> pd.DataFrame:
    if not config.analysis.admet.enabled:
        return frame
    reports = compute_admet_many(list(zip(names, smiles, strict=True)))
    admet_df = pd.DataFrame(
        {
            "name": [r.name for r in reports],
            "mw": [r.mw for r in reports],
            "logp": [r.logp for r in reports],
            "qed": [r.qed for r in reports],
            "sa_score": [r.sa_score for r in reports],
            "passes_lipinski": [r.passes_lipinski for r in reports],
            "passes_veber": [r.passes_veber for r in reports],
            "pains_alert_count": [len(r.pains_alerts) for r in reports],
            "drug_likeness": [r.drug_likeness for r in reports],
        }
    )
    return frame.merge(admet_df, on="name", how="left")


def _annotate_selectivity(
    frame: pd.DataFrame,
    names: list[str],
    smiles: list[str],
    p_btk: np.ndarray,
    config: Config,
    project_root: Path,
) -> pd.DataFrame:
    if not config.analysis.selectivity.enabled:
        return frame
    panels: dict[str, pd.DataFrame] = {}
    for kinase in config.analysis.selectivity.off_targets:
        panel_path = Path(
            config.analysis.selectivity.panel_glob.format(kinase=kinase)
        )
        if not panel_path.is_absolute():
            panel_path = project_root / panel_path
        if not panel_path.exists():
            logger.warning(
                "Off-target panel missing for %s at %s — skipping that kinase",
                kinase,
                panel_path,
            )
            continue
        panels[kinase] = pd.read_csv(panel_path)
    if not panels:
        logger.warning(
            "Selectivity enabled but no panels loaded — skipping selectivity scoring"
        )
        return frame

    models = train_off_target_models(
        panels,
        morgan_radius=config.analysis.selectivity.morgan_radius,
        morgan_n_bits=config.analysis.selectivity.morgan_n_bits,
        n_estimators=config.analysis.selectivity.n_estimators,
        random_seed=config.analysis.selectivity.random_seed,
    )
    # MLRescorer returns negated probabilities as scores; recover positive P(BTK).
    reports = score_selectivity(
        names=names,
        smiles=smiles,
        p_btk=p_btk,
        off_target_models=models,
        morgan_radius=config.analysis.selectivity.morgan_radius,
        morgan_n_bits=config.analysis.selectivity.morgan_n_bits,
    )
    sel_df = pd.DataFrame(
        {
            "name": [r.name for r in reports],
            "p_btk": [r.p_btk for r in reports],
            "max_off_target": [r.max_off_target for r in reports],
            "max_off_target_prob": [r.max_off_target_probability for r in reports],
            "selectivity_index": [r.selectivity_index for r in reports],
        }
    )
    return frame.merge(sel_df, on="name", how="left")
