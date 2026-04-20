"""Integration test: run the full pipeline on fixture data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from btk_aidd.config import Config
from btk_aidd.pipeline import load_ligand_table, run_pipeline


def test_load_ligand_table_fast_mode(fixtures_dir: Path) -> None:
    rows = load_ligand_table(
        actives_csv=fixtures_dir / "btk_actives.csv",
        decoys_csv=fixtures_dir / "btk_decoys.csv",
        fast_mode=True,
        fast_actives=5,
        fast_decoys=5,
        rng_seed=1,
    )
    # 5 actives + 5 decoys
    assert sum(1 for r in rows if r.label == 1) == 5
    assert sum(1 for r in rows if r.label == 0) == 5


def test_load_ligand_table_rejects_missing_columns(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("wrong,schema\nX,Y\n")
    (tmp_path / "d.csv").write_text("name,canonical_smiles\nDEC_0,CCO\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_ligand_table(
            actives_csv=tmp_path / "a.csv",
            decoys_csv=tmp_path / "d.csv",
            fast_mode=False,
            fast_actives=0,
            fast_decoys=0,
            rng_seed=0,
        )


def test_pipeline_end_to_end(
    default_config: Config, fixtures_dir: Path, tmp_path: Path, project_root: Path
) -> None:
    cfg = default_config.model_dump()
    cfg["runtime"]["output_dir"] = str(tmp_path / "results")
    cfg["runtime"]["fast_actives"] = 15
    cfg["runtime"]["fast_decoys"] = 15
    cfg["scoring"]["ml"]["n_estimators"] = 50
    cfg_typed = Config.model_validate(cfg)

    outputs = run_pipeline(
        config=cfg_typed,
        actives_csv=fixtures_dir / "btk_actives.csv",
        decoys_csv=fixtures_dir / "btk_decoys.csv",
        project_root=project_root,
    )

    assert outputs.scores_csv.exists()
    assert outputs.roc_png.exists()
    assert outputs.enrichment_png.exists()
    assert outputs.top_hits_png.exists()

    scores = pd.read_csv(outputs.scores_csv)
    expected_columns = {
        "name",
        "label",
        "docking",
        "physics",
        "ml",
        "docking_z",
        "physics_z",
        "ml_z",
        "consensus",
    }
    assert expected_columns.issubset(scores.columns)
    assert len(scores) > 0

    # All four reports should exist and have sensible AUCs.
    assert len(outputs.reports) == 4
    for report in outputs.reports:
        assert 0.0 <= report.roc_auc <= 1.0
        assert report.n_actives + report.n_decoys == len(scores)
