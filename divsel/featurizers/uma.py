"""UMA latent features -- PLACEHOLDER.

Not implemented in 0.1.0.  This file exists to hold the contract and the
decisions that are expensive to rediscover, so that filling it in later is a
small change rather than a refactor of the streaming loop.

HOW TO IMPLEMENT
----------------
Delegate, do not duplicate.  A working extractor already exists at
``/projects/caiw/lipingliu/Package_development/latent_spaces_as_features``
(package ``latent_features``), with a ``BackendBase`` ABC and a ``UMABackend``
that already handles:

* the fairchem dual load path -- ``load_predict_unit(path)`` for a fine-tuned
  checkpoint, ``pretrained_mlip.get_predict_unit(name)`` for a pretrained model;
* forward-hook layer taps, with ``energy_block[0]`` (128-dim) the default;
* ``calc.reset()`` before extraction.  ASE caches calculator results, which
  silently defeats forward hooks -- after a relaxation the geometry matches the
  cached state, so no forward pass runs and the hook never fires.  Do not
  remove that call;
* per-atom axis normalization: the energy head gives ``(n_atoms, d)`` while
  backbone taps give ``(n_atoms, 1, d)``.

So ``UMAFeaturizer`` should be a thin adapter: build the backend in
:meth:`prepare`, extract per-atom features in :meth:`featurize`, mean-pool to
one vector per structure, return float32.

THINGS THAT WILL BITE
---------------------
* ``parallel_mode`` must be ``"in_process_batched"``.  A CUDA context cannot be
  forked across a ``multiprocessing`` Pool, so the streaming loop must not build
  a worker pool for this backend.  The base class default is ``process_pool``;
  the override below is deliberate.
* The fairchem import takes **over two minutes**.  It must stay inside the
  methods -- never at module scope -- or ``divsel --help`` becomes unusable.
* fairchem needs the ``fairchem-v3`` environment (python 3.12, ase 3.27,
  numpy 2.2), while dscribe here runs on python 3.10 / numpy 1.26.  That is why
  this is an optional extra, not a core dependency.
* **UMA inference is not bitwise deterministic**: repeating the identical
  forward pass on the identical structure shifts per-atom features by ~2.4e-7
  (measured on uma-s-1p1, CPU).  Features therefore reproduce to ~1e-6, never
  exactly.  Never assert tighter than that -- it tests the noise floor, not the
  code.  Note this does not weaken the FPS determinism guarantee, which is about
  the *selection given fixed features*.
* ``identity_fields`` must include the model name or checkpoint hash, the layer
  tap, and the pooling mode.  Two UMA runs with different taps are no more
  comparable than SOAP at two different cutoffs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from ase import Atoms

from .base import Featurizer

__all__ = ["UMAFeaturizer"]

_NOT_IMPLEMENTED = (
    "the 'uma' featurizer is not implemented yet (divsel 0.1.0).\n"
    "It is registered so the CLI and the descriptor identity already account for "
    "it, but no feature extraction code exists.\n"
    "To implement it, see the notes in divsel/featurizers/uma.py: wrap the "
    "existing `latent_features` package (latent_spaces_as_features) rather than "
    "reimplementing the fairchem load path and forward hooks.\n"
    "For now use --featurizer soap."
)


class UMAFeaturizer(Featurizer):
    name = "uma"
    # A CUDA context cannot be forked across a multiprocessing Pool.
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
