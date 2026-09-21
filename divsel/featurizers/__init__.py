"""Featurizer registry.

Backends are imported lazily so that a heavy optional dependency (torch,
fairchem) is never pulled in just because the module was imported.  Adding a
backend is one entry in ``_REGISTRY`` plus a class satisfying
:class:`divsel.featurizers.base.Featurizer`.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import Featurizer

__all__ = ["Featurizer", "get_featurizer", "featurizer_from_spec", "available"]


def _load_soap() -> type[Featurizer]:
    from .soap import SoapFeaturizer

    return SoapFeaturizer


def _load_uma() -> type[Featurizer]:
    from .uma import UMAFeaturizer

    return UMAFeaturizer


def _load_table() -> type[Featurizer]:
    from .table import TableFeaturizer

    return TableFeaturizer


def _load_mace() -> type[Featurizer]:
    from .mace import MACEFeaturizer

    return MACEFeaturizer


_REGISTRY: dict[str, Callable[[], type[Featurizer]]] = {
    "soap": _load_soap,
    "table": _load_table,
    "uma": _load_uma,
    "mace": _load_mace,
}

#: backends that actually produce vectors today. "table" serves precomputed
#: ones rather than computing them, but it is every bit as usable.
IMPLEMENTED = ("soap", "table")


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_featurizer(name: str, **kwargs: Any) -> Featurizer:
    """Construct a featurizer by registry name."""
    try:
        loader = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown featurizer {name!r}; available: {available()} "
            f"(implemented: {list(IMPLEMENTED)})"
        ) from None
    return loader()(**kwargs)


def featurizer_from_spec(spec: dict) -> Featurizer:
    """Rebuild a featurizer from :meth:`Featurizer.spec`, e.g. inside a worker."""
    cls = _REGISTRY[spec["name"]]()
    from_spec = getattr(cls, "from_spec_kwargs", None)
    if from_spec is not None:
        return from_spec(**spec.get("kwargs", {}))
    return cls(**spec.get("kwargs", {}))
