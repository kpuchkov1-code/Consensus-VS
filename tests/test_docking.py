"""Tests for docking engines."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from btk_aidd.config import DockingConfig
from btk_aidd.data.ligands import PreparedLigand
from btk_aidd.data.receptor import DockingBox
from btk_aidd.docking.cached_engine import CachedEngine
from btk_aidd.docking.factory import build_engine

_BOX = DockingBox(center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0))


def test_cached_engine_returns_known_score(
    cached_scores_csv: Path, prepared_ligands: list[PreparedLigand]
) -> None:
    engine = CachedEngine(cached_scores_csv)
    target = next(lig for lig in prepared_ligands if lig.name == "LIG_A")
    result = engine.dock(target, "ignored", _BOX)
    assert result.success is True
    assert math.isclose(result.affinity_kcal_mol, -9.2, rel_tol=1e-6)


def test_cached_engine_unknown_ligand_fails_gracefully(
    cached_scores_csv: Path, prepared_ligands: list[PreparedLigand]
) -> None:
    engine = CachedEngine(cached_scores_csv)
    # Forge a ligand whose name is missing from the cache.
    forged = PreparedLigand(
        name="NOT_IN_CACHE",
        smiles="CCO",
        mol=prepared_ligands[0].mol,
        mmff_energy_kcal_mol=0.0,
        embed_success=True,
        minimise_success=True,
    )
    result = engine.dock(forged, "ignored", _BOX)
    assert result.success is False
    assert math.isnan(result.affinity_kcal_mol)


def test_cached_engine_dock_many(
    cached_scores_csv: Path, prepared_ligands: list[PreparedLigand]
) -> None:
    engine = CachedEngine(cached_scores_csv)
    results = engine.dock_many(prepared_ligands, "ignored", _BOX)
    assert len(results) == len(prepared_ligands)
    assert all(r.success for r in results)


def test_cached_engine_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CachedEngine(tmp_path / "missing.csv")


def test_cached_engine_rejects_bad_schema(tmp_path: Path) -> None:
    df = pd.DataFrame({"ligand": ["x"], "score": [-7.0]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        CachedEngine(path)


def test_build_engine_factory_cached(
    cached_scores_csv: Path,
) -> None:
    cfg = DockingConfig(
        engine="cached",
        cached_scores_csv=str(cached_scores_csv),
    )
    engine = build_engine(cfg)
    assert isinstance(engine, CachedEngine)
    assert engine.size == 4


def test_build_engine_factory_unknown_raises() -> None:
    cfg = DockingConfig(engine="cached")  # valid for construction
    with pytest.raises(ValueError, match="Unknown docking engine"):
        # Manually force the ``engine`` past validation to exercise the factory guard.
        object.__setattr__(cfg, "engine", "potato")  # intentional: test the factory's defensive branch
        build_engine(cfg)
