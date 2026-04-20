"""Docking engine factory — one place to build an engine from :class:`DockingConfig`."""

from __future__ import annotations

from pathlib import Path

from btk_aidd.config import DockingConfig
from btk_aidd.docking.cached_engine import CachedEngine
from btk_aidd.docking.engine import DockingEngine
from btk_aidd.logger import get_logger

logger = get_logger(__name__)


def build_engine(config: DockingConfig, project_root: Path | None = None) -> DockingEngine:
    """Construct the docking engine selected by ``config.engine``.

    Args:
        config: Validated :class:`DockingConfig`.
        project_root: Optional project root used to resolve relative paths
            (e.g. ``cached_scores_csv``).

    Returns:
        A ready-to-use :class:`DockingEngine`.

    Raises:
        ImportError: If the ``vina`` engine is requested but the optional
            ``vina`` package is not installed.
    """
    if config.engine == "cached":
        cache_path = Path(config.cached_scores_csv)
        if not cache_path.is_absolute() and project_root is not None:
            cache_path = project_root / cache_path
        logger.info("Building CachedEngine from %s", cache_path)
        return CachedEngine(cache_path)

    if config.engine == "vina":
        from btk_aidd.docking.vina_engine import VinaEngine

        logger.info(
            "Building VinaEngine (exhaustiveness=%d, num_poses=%d)",
            config.exhaustiveness,
            config.num_poses,
        )
        return VinaEngine(exhaustiveness=config.exhaustiveness, num_poses=config.num_poses)

    raise ValueError(f"Unknown docking engine: {config.engine!r}")
