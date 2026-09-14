"""MACE latent features -- PLACEHOLDER.

Not implemented in 0.1.0, and there is currently no plan to build it: MACE was
deliberately dropped from the sibling ``latent_features`` project in favour of
UMA alone.  This stub is kept so the registry, the CLI choices and the
descriptor identity already have a slot for it, and so that whoever adds it
starts from the contract in :mod:`divsel.featurizers.base` rather than from a
blank file.

If you do implement it, the same three constraints as UMA apply:
``parallel_mode = "in_process_batched"`` (no forking a CUDA context), a lazy
import (torch is slow to import and must not be pulled in by ``divsel --help``),
and ``identity_fields`` that pin the model, the layer tap and the pooling mode.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from ase import Atoms

from .base import Featurizer

__all__ = ["MACEFeaturizer"]

_NOT_IMPLEMENTED = (
    "the 'mace' featurizer is not implemented yet (divsel 0.1.0).\n"
    "It is registered as a placeholder only -- see divsel/featurizers/mace.py.\n"
    "For now use --featurizer soap."
)


class MACEFeaturizer(Featurizer):
    name = "mace"
    parallel_mode = "in_process_batched"

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def prepare(self, frame_index) -> None:  # pragma: no cover - unreachable
        raise NotImplementedError(_NOT_IMPLEMENTED)

    @property
    def feature_dim(self) -> int:  # pragma: no cover - unreachable
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def featurize(self, atoms_list: Sequence[Atoms]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def identity_fields(self) -> dict:  # pragma: no cover - unreachable
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def spec(self) -> dict:  # pragma: no cover - unreachable
        raise NotImplementedError(_NOT_IMPLEMENTED)
