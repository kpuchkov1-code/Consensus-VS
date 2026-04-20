"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from btk_aidd.config import Config, load_config


def test_default_config_loads(project_root: Path) -> None:
    cfg = load_config(project_root / "config" / "default.yaml")
    assert cfg.target.name == "BTK"
    assert cfg.target.chembl_id == "CHEMBL5251"
    assert cfg.target.pdb_id == "4OT6"
    assert cfg.runtime.mode in {"fast", "full"}


def test_config_is_frozen(default_config: Config) -> None:
    with pytest.raises(ValidationError):
        default_config.target.name = "Other"  # type: ignore[misc]


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_unknown_docking_engine_rejected(tmp_path: Path, default_config: Config) -> None:
    cfg = default_config.model_dump()
    cfg["docking"]["engine"] = "homebrew"  # not allowed
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValidationError):
        load_config(path)


def test_enrichment_fraction_range_validated(tmp_path: Path, default_config: Config) -> None:
    cfg = default_config.model_dump()
    cfg["evaluation"]["enrichment_fractions"] = [0.0, 0.1]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValidationError):
        load_config(path)


def test_bad_mmff_variant_rejected(tmp_path: Path, default_config: Config) -> None:
    cfg = default_config.model_dump()
    cfg["ligand_prep"]["mmff_variant"] = "UFF"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValidationError):
        load_config(path)
