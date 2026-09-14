"""divsel -- diversity selection of atomic structures.

Pick a diverse subset of structures using SOAP descriptors and either k-means
or farthest-point sampling, with streaming for datasets that do not fit in
memory and a seed-set mechanism for continuing across active-learning rounds.

The central idea is one function::

    from divsel import select_diverse
    result = select_diverse(X_candidates, n_select=200, X_seed=X_already_have,
                            method="kmeans")

``X_seed`` means "descriptors I already hold: stay away from these, and never
return them".  Both ``--selected_images`` (previous rounds) and the streaming
carry-forward (earlier batches of this run) go through it, unchanged.

Attributes are resolved lazily so that ``import divsel`` -- and in particular
``divsel --help`` -- does not pay for ase, sklearn or dscribe.
"""

from __future__ import annotations

__version__ = "0.1.0"

_LAZY = {
    "select_diverse": ("divsel.selection", "select_diverse"),
    "SelectionResult": ("divsel.selection", "SelectionResult"),
    "min_sqdist_to_set": ("divsel.selection", "min_sqdist_to_set"),
    "fps_order_scores": ("divsel.selection", "fps_order_scores"),
    "run_selection": ("divsel.streaming", "run_selection"),
    "run_describe": ("divsel.streaming", "run_describe"),
    "gather_frames": ("divsel.gather", "gather"),
    "make_figures": ("divsel.plotting", "make_figures"),
    "ensure_cell": ("divsel.box", "ensure_cell"),
    "scan_sources": ("divsel.frames", "scan_sources"),
    "get_featurizer": ("divsel.featurizers", "get_featurizer"),
    "SoapConfig": ("divsel.config", "SoapConfig"),
    "BoxConfig": ("divsel.config", "BoxConfig"),
    "SamplingConfig": ("divsel.config", "SamplingConfig"),
    "StreamConfig": ("divsel.config", "StreamConfig"),
    "SelectConfig": ("divsel.config", "SelectConfig"),
    "descriptor_id": ("divsel.config", "descriptor_id"),
    "DivselError": ("divsel.errors", "DivselError"),
    "DescriptorMismatchError": ("divsel.errors", "DescriptorMismatchError"),
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'divsel' has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
