"""Smoke tests for the click CLI."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from btk_aidd.cli import cli


def test_validate_default_config(project_root: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["validate", "--config", str(project_root / "config" / "default.yaml")],
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["target"]["name"] == "BTK"


def test_validate_rejects_bad_config(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "target:\n"
        "  name: BTK\n"
        "  chembl_id: CHEMBL5251\n"
        "  pdb_id: 4OT6\n"
        "  reference_ligand_resname: 1E8\n"
        "docking:\n"
        "  engine: homebrew\n"  # invalid
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "--config", str(bad)])
    assert result.exit_code != 0


def test_run_smoke(project_root: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--actives",
            str(project_root / "data" / "fixtures" / "btk_actives.csv"),
            "--decoys",
            str(project_root / "data" / "fixtures" / "btk_decoys.csv"),
            "--mode",
            "fast",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "=== Results ===" in result.output
    assert (tmp_path / "out" / "scores.csv").exists()
    assert (tmp_path / "out" / "roc.png").exists()


def test_help_shows_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for name in ("run", "fetch", "validate"):
        assert name in result.output
