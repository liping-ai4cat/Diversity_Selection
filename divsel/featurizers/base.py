"""The featurizer contract.

SOAP is the only backend implemented in 0.1.0.  The contract below exists so
that adding a model backend (UMA, MACE) later does not require rewriting
:mod:`divsel.streaming`.  Three properties matter, and all three are chosen for
the benefit of a future GPU backend rather than for SOAP:

1. **`featurize` takes a list of Atoms, not one.**  SOAP would be perfectly
   happy per-structure, but a torch model wants a batch per forward pass.  A
   per-structure API would make the model path pathologically slow.
2. **`parallel_mode` is declared by the featurizer**, not assumed by the caller.
   SOAP wants ``process_pool``.  A torch model wants ``in_process_batched``,
   because a CUDA context cannot be forked across a multiprocessing Pool.
3. **`feature_dim` is known before the first frame is featurized.**  The
   descriptor store allocates one ``(N, d)`` memmap up front; SOAP derives d
   from the scan's species union, a model backend from its checkpoint.

Everything downstream -- L2 normalization, the seed-set primitive, streaming,
gather, the plots -- operates on an ``(n, d)`` float32 matrix and neither knows
nor cares where it came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from ase import Atoms

__all__ = ["Featurizer"]


class Featurizer(ABC):
    """Turns structures into one fixed-length vector each."""

    #: registry key, and the value recorded in ``descriptor_id``
    name: str = "base"

    #: "process_pool" (CPU, forkable) or "in_process_batched" (GPU / torch)
    parallel_mode: str = "process_pool"

    @abstractmethod
    def prepare(self, frame_index: Any) -> None:
        """Hook called once after the scan, before any featurization.

        SOAP uses it to resolve the species union into a fixed descriptor
        length.  Model backends typically ignore the frame index and load a
        checkpoint instead.  Must leave :attr:`feature_dim` well defined.
        """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Length of one feature vector.  Valid after :meth:`prepare`."""

    @abstractmethod
    def featurize(self, atoms_list: Sequence[Atoms]) -> np.ndarray:
        """Return ``(len(atoms_list), feature_dim)`` float32, row-aligned."""

    @abstractmethod
    def identity_fields(self) -> dict:
        """Everything that changes the meaning of a feature vector.

        Fed into :func:`divsel.config.descriptor_id`.  Omitting a field here
        means two incomparable feature sets can hash the same, which is exactly
        the silent-corruption failure this machinery exists to prevent.
        """

    @abstractmethod
    def spec(self) -> dict:
        """Picklable description, so worker processes can rebuild this object.

        Passing parameters rather than the object itself keeps the featurizer
        usable under both the fork and spawn start methods.
        """

    @property
    def neighbor_cutoff(self) -> float | None:
        """Radial cutoff of the local environment, or None if not applicable.

        This is what decides how much vacuum a boxed structure needs: putting a
        molecule in a box reproduces the non-periodic descriptor only when
        ``2 * vacuum >= cutoff``, otherwise the structure sees its own periodic
        images.  ``None`` means "no meaningful cutoff", and the boxing guard is
        then skipped.

        MUST be valid before :meth:`prepare` is called -- the boxing probe runs
        before the scan pass, so it cannot wait for the species union.
        """
        return None

    def check_supported(self, atoms: Atoms) -> str | None:
        """Return None if the structure can be featurized, else a reason string.

        The reason is recorded per frame in ``frames.csv``; the caller decides
        whether to skip or raise.
        """
        return None
