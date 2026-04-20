"""ChEMBL bioactivity fetcher.

Supports two modes:

* **live**  – requires the optional ``chembl-webresource-client`` package and
  network access. Queries the ChEMBL web services for activities against the
  configured target and returns a cleaned :class:`pandas.DataFrame`.
* **cached** – loads a previously-fetched CSV from disk. This is the default
  path for reproducible runs and CI, where hitting the live API would be slow
  or flaky.

The returned DataFrame schema is stable across both modes:

    ======================  ==============  ================================
    column                  dtype           description
    ======================  ==============  ================================
    molecule_chembl_id      str             ChEMBL molecule accession
    canonical_smiles        str             canonical SMILES
    standard_type           str             IC50 / Ki / Kd / ...
    standard_value_nM       float           concentration in nanomolar
    pchembl_value           float           -log10(M) of standard_value
    target_chembl_id        str             target accession (e.g. CHEMBL5251)
    ======================  ==============  ================================
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from btk_aidd.logger import get_logger

logger = get_logger(__name__)

_COLUMNS: tuple[str, ...] = (
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_value_nM",
    "pchembl_value",
    "target_chembl_id",
)


@dataclass(frozen=True)
class ChEMBLQuery:
    """Parameters for a ChEMBL bioactivity query."""

    target_chembl_id: str
    activity_types: tuple[str, ...]
    min_pchembl: float
    max_records: int = 10_000


def fetch_live(query: ChEMBLQuery) -> pd.DataFrame:
    """Query ChEMBL for activities against ``query.target_chembl_id``.

    Args:
        query: Parameters specifying target, activity types, and filters.

    Returns:
        A DataFrame conforming to the schema documented in this module.

    Raises:
        ImportError: If the optional ``chembl-webresource-client`` dependency
            is not installed.
    """
    try:
        from chembl_webresource_client.new_client import new_client  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "chembl-webresource-client is required for live ChEMBL queries. "
            "Install with: pip install chembl-webresource-client"
        ) from exc

    logger.info(
        "Fetching %s activities for target %s (min pChEMBL %.1f)",
        ", ".join(query.activity_types),
        query.target_chembl_id,
        query.min_pchembl,
    )

    activities = new_client.activity  # type: ignore[attr-defined]
    resultset = activities.filter(
        target_chembl_id=query.target_chembl_id,
        standard_type__in=list(query.activity_types),
        pchembl_value__gte=query.min_pchembl,
        standard_relation="=",
    ).only(
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "target_chembl_id",
    )

    records: list[dict[str, Any]] = []
    for row in _bounded(resultset, query.max_records):
        smiles = row.get("canonical_smiles")
        value = row.get("standard_value")
        pchembl = row.get("pchembl_value")
        units = row.get("standard_units")
        if not smiles or value is None or pchembl is None:
            continue
        value_nm = _to_nanomolar(float(value), units)
        if value_nm is None:
            continue
        records.append(
            {
                "molecule_chembl_id": row.get("molecule_chembl_id"),
                "canonical_smiles": smiles,
                "standard_type": row.get("standard_type"),
                "standard_value_nM": value_nm,
                "pchembl_value": float(pchembl),
                "target_chembl_id": row.get("target_chembl_id"),
            }
        )

    df = pd.DataFrame.from_records(records, columns=list(_COLUMNS))
    logger.info("Fetched %d activity rows", len(df))
    return _deduplicate(df)


def load_cached(path: str | Path) -> pd.DataFrame:
    """Load a previously-saved activity CSV.

    The CSV must contain at least the columns listed in :data:`_COLUMNS`.
    Extra columns are preserved; missing columns raise a ``ValueError``.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cached ChEMBL CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in _COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Cached CSV {csv_path} missing required columns: {missing}")
    logger.info("Loaded %d activity rows from %s", len(df), csv_path)
    return _deduplicate(df)


def save_cache(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a fetched DataFrame to CSV (atomic write)."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(dst)
    logger.info("Saved %d activity rows to %s", len(df), dst)


def _to_nanomolar(value: float, units: str | None) -> float | None:
    """Convert a ChEMBL ``standard_value`` + units to nanomolar.

    Returns ``None`` for unsupported / ambiguous units so the caller can skip.
    """
    if units is None:
        return None
    u = units.strip().lower()
    if u in {"nm", "nanomolar"}:
        return value
    if u in {"um", "\u00b5m", "micromolar"}:
        return value * 1_000.0
    if u in {"mm", "millimolar"}:
        return value * 1_000_000.0
    if u in {"pm", "picomolar"}:
        return value / 1_000.0
    if u == "m":
        return value * 1e9
    return None


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate molecule rows to the strongest measured activity."""
    if df.empty:
        return df
    deduped = (
        df.sort_values("pchembl_value", ascending=False)
        .drop_duplicates(subset=["molecule_chembl_id"], keep="first")
        .reset_index(drop=True)
    )
    dropped = len(df) - len(deduped)
    if dropped > 0:
        logger.info("Deduplicated %d rows -> %d molecules", len(df), len(deduped))
    return deduped


def _bounded(iterable: Iterable[dict[str, Any]], limit: int) -> Iterable[dict[str, Any]]:
    """Yield at most ``limit`` items from ``iterable``.

    A tiny helper so the live fetch doesn't materialise tens of thousands of
    rows when the caller only needs a few hundred.
    """
    for i, item in enumerate(iterable):
        if i >= limit:
            break
        yield item
