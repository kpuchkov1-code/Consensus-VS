"""Lightweight project-wide logger setup.

Uses the stdlib :mod:`logging` module with a single StreamHandler so that the
pipeline integrates cleanly with external tooling (CI, Jupyter, subprocess
capture) without imposing a heavyweight logging framework.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_configured: bool = False


def configure(level: str = "INFO") -> None:
    """Install the root handler and set the global level.

    Calling ``configure`` multiple times is safe: only the first call attaches
    a handler; subsequent calls just update the level.

    Args:
        level: A standard logging level name (``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``, ``"ERROR"``).
    """
    global _configured
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    if not _configured:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
        _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, ensuring the root is configured."""
    if not _configured:
        configure()
    return logging.getLogger(name)
