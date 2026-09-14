"""Run manifest: everything needed to reproduce a selection.

The original scripts printed timestamps and nothing else -- the SOAP
parameters, the seed, k, and which frames were chosen under them were never
recorded together, so a selection could not be reproduced or audited after the
fact.  This module writes one JSON file that pins all of it.
"""

from __future__ import annotations

import getpass
import json
import platform
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import descriptor_id, format_descriptor_diff
from .errors import DescriptorMismatchError

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "new_run_id",
    "environment_info",
    "write_manifest",
    "read_manifest",
    "validate_compatible",
]


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _version(module_name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(module_name)
    except Exception:
        try:
            mod = __import__(module_name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None


def environment_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "pandas": _version("pandas"),
        "scikit_learn": _version("scikit-learn"),
        "ase": _version("ase"),
        "dscribe": _version("dscribe"),
        "divsel": _version("divsel"),
    }


def run_info(argv: list[str] | None = None) -> dict:
    return {
        "run_id": new_run_id(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "user": _safe_user(),
        "cwd": str(Path.cwd()),
        "argv": list(argv if argv is not None else sys.argv),
    }


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - no passwd entry in some containers
        return "unknown"


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False, default=str))
    return path


def read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def validate_compatible(
    seed_fields: Mapping[str, Any],
    this_fields: Mapping[str, Any],
    *,
    seed_label: str = "seed",
) -> None:
    """Raise :class:`DescriptorMismatchError` unless the identities agree."""
    a, b = descriptor_id(seed_fields), descriptor_id(this_fields)
    if a != b:
        raise DescriptorMismatchError(
            format_descriptor_diff(
                seed_fields, this_fields, seed_id=a, this_id=b, seed_label=seed_label
            )
        )
