"""Command-line interface for the BTK-AIDD pipeline.

Entry point installed as ``btk-aidd``. Three subcommands:

* ``run``        end-to-end pipeline on actives + decoys CSVs
* ``fetch``      live ChEMBL fetch (requires chembl-webresource-client)
* ``validate``   validate a config file and print the resolved values

All subcommands take an optional ``--config`` path; otherwise the packaged
``config/default.yaml`` is loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from btk_aidd.config import Config, load_config
from btk_aidd.logger import configure, get_logger
from btk_aidd.pipeline import run_pipeline

logger = get_logger(__name__)


def _load(config_path: str | None) -> Config:
    return load_config(Path(config_path)) if config_path else load_config()


@click.group(help="BTK-AIDD: hybrid AI + physics virtual screening pipeline.")
@click.version_option()
def cli() -> None:  # pragma: no cover - thin click glue
    pass


@cli.command("validate", help="Validate a config file and print the parsed values.")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None)
def validate_cmd(config_path: str | None) -> None:
    cfg = _load(config_path)
    click.echo(json.dumps(cfg.model_dump(), indent=2, default=str))


@cli.command("run", help="Run the end-to-end pipeline.")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option(
    "--actives",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to actives CSV (columns: molecule_chembl_id, canonical_smiles).",
)
@click.option(
    "--decoys",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to decoys CSV (columns: name, canonical_smiles).",
)
@click.option(
    "--mode",
    type=click.Choice(["fast", "full"]),
    default=None,
    help="Override runtime.mode from config.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Override runtime.output_dir from config.",
)
def run_cmd(
    config_path: str | None,
    actives: str,
    decoys: str,
    mode: str | None,
    output: str | None,
) -> None:
    cfg = _load(config_path)
    if mode is not None or output is not None:
        cfg_dump = cfg.model_dump()
        if mode is not None:
            cfg_dump["runtime"]["mode"] = mode
        if output is not None:
            cfg_dump["runtime"]["output_dir"] = output
        cfg = Config.model_validate(cfg_dump)

    configure(cfg.runtime.log_level)
    logger.info(
        "Starting pipeline (target=%s, mode=%s, engine=%s)",
        cfg.target.name,
        cfg.runtime.mode,
        cfg.docking.engine,
    )
    outputs = run_pipeline(cfg, actives_csv=actives, decoys_csv=decoys)

    click.echo("\n=== Results ===")
    for r in outputs.reports:
        ef_str = ", ".join(
            f"EF@{round(f * 100)}%={v:.2f}" for f, v in r.enrichment.items()
        )
        click.echo(f"{r.name:<10} AUC={r.roc_auc:.3f}  {ef_str}")
    click.echo(f"\nArtefacts written to {outputs.scores_csv.parent}/")
    click.echo(f"  scores:     {outputs.scores_csv}")
    click.echo(f"  ROC plot:   {outputs.roc_png}")
    click.echo(f"  EF plot:    {outputs.enrichment_png}")
    click.echo(f"  top hits:   {outputs.top_hits_png}")


@cli.command("fetch", help="Fetch BTK actives live from ChEMBL (optional dependency).")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    required=True,
    help="Destination CSV for fetched actives.",
)
def fetch_cmd(config_path: str | None, output: str) -> None:
    from btk_aidd.data.chembl import ChEMBLQuery, fetch_live, save_cache

    cfg = _load(config_path)
    query = ChEMBLQuery(
        target_chembl_id=cfg.target.chembl_id,
        activity_types=tuple(cfg.data.actives.activity_types),
        min_pchembl=cfg.data.actives.min_pchembl,
        max_records=cfg.data.actives.max_actives * 10,
    )
    df = fetch_live(query)
    save_cache(df, output)
    click.echo(f"Fetched {len(df)} activities -> {output}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli())  # type: ignore[misc]
