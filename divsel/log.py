"""Verbosity control for divsel.

Three levels, selected with ``--verbose``:

``quiet``   warnings and errors only.
``normal``  one line per stage (scan, describe, select, write) plus the final
            summary.  This is what you want in a SLURM log.
``high``    everything: per-batch progress and timings, per-file scan results,
            the species union and resulting descriptor length, cache hit/miss
            paths, per-batch quotas, pool size and evictions, k-means cluster
            bookkeeping, and the distance diagnostics.

The original scripts used bare ``print()`` everywhere, including a per-structure
debug dump from inside each worker process.  Everything of that kind belongs at
``high``.  Library code should never print; it calls :func:`get_logger`.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

Verbosity = Literal["quiet", "normal", "high"]

VERBOSITY_LEVELS: dict[str, int] = {
    "quiet": logging.WARNING,
    "normal": logging.INFO,
    "high": logging.DEBUG,
}

ROOT_NAME = "divsel"
_configured = False


def setup_logging(verbose: str = "normal", *, stream=None) -> logging.Logger:
    """Configure the ``divsel`` logger hierarchy.  Idempotent.

    Only the CLI (or a user of the Python API who wants log output) should call
    this.  Importing divsel does not configure logging.
    """
    global _configured

    if verbose not in VERBOSITY_LEVELS:
        raise ValueError(
            f"unknown verbosity {verbose!r}; expected one of {sorted(VERBOSITY_LEVELS)}"
        )

    logger = logging.getLogger(ROOT_NAME)
    logger.setLevel(VERBOSITY_LEVELS[verbose])

    if not _configured:
        handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
        # Timestamps only at 'high' -- at 'normal' they are noise, and the old
        # scripts' datetime.now() prefixes made the log harder to skim.
        if verbose == "high":
            fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
            datefmt = "%H:%M:%S"
        else:
            fmt = "%(levelname)-7s %(message)s"
            datefmt = None
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logger.addHandler(handler)
        logger.propagate = False
        _configured = True
    else:
        for handler in logger.handlers:
            if verbose == "high":
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
                    )
                )
            else:
                handler.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a submodule, e.g. ``get_logger(__name__)``."""
    if name.startswith(ROOT_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")


def is_high(logger: logging.Logger | None = None) -> bool:
    """True when 'high' verbosity is active.

    Use to guard the construction of expensive debug messages, not to decide
    whether to log at all.
    """
    logger = logger or logging.getLogger(ROOT_NAME)
    return logger.isEnabledFor(logging.DEBUG)
