"""Give non-periodic structures a cell so the descriptor is well defined.

The key fact that makes this clean: when ``2 * vacuum >= r_cut``, periodic SOAP
on the boxed structure is numerically identical to non-periodic SOAP, because
every periodic image is farther than the cutoff from every atom.  So a mixed
dataset of bulk cells, slabs and molecules can run through a single
``periodic=True`` descriptor with one descriptor id.

Verified against dscribe 2.1.1 / ase 3.22.1:

* ``atoms.center(vacuum=v)`` sets each cell length to ``extent + 2v``, so the
  minimum image separation is exactly ``2v``.
* molecule in a vacuum-15 box, periodic SOAP vs non-periodic SOAP: maxdiff 1.6e-11.
* same molecule at vacuum 2 (2v = 4 < r_cut = 6): maxdiff 466.

Hence the vacuum requirement is a hard error -- but **only for a structure
that is actually boxed**.  A fully periodic cell never enters the boxing path
(:func:`needed_axes` returns all-false and :func:`ensure_cell` hands the
structure straight back as ``native``), so no amount of vacuum is required of
it and the guard must not fire on its account.  That is why the check lives
here, next to the periodicity decision, rather than as an up-front gate on the
configuration: only this module knows which structures get a box.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from .config import BoxConfig
from .errors import BoxError, VacuumTooSmallError

__all__ = ["ensure_cell", "needed_axes", "needs_boxing"]

_CARTESIAN = np.eye(3)


def _orthonormal_completion(
    fixed_vectors: list[np.ndarray], n_needed: int
) -> list[np.ndarray]:
    """Directions to pad along, orthogonal to the lattice vectors we keep.

    With two fixed vectors (a slab) this is just the plane normal, which is what
    a hexagonal surface needs -- padding along a Cartesian axis there would tilt
    the cell.  With one or zero fixed vectors we Gram-Schmidt Cartesian axes
    against what is already there, so the padded directions are mutually
    orthogonal and the cell cannot come out degenerate.
    """
    basis: list[np.ndarray] = []
    for v in fixed_vectors:
        w = np.asarray(v, dtype=float).copy()
        for b in basis:
            w -= (w @ b) * b
        norm = np.linalg.norm(w)
        if norm > 1e-9:
            basis.append(w / norm)

    out: list[np.ndarray] = []
    while len(out) < n_needed:
        best, best_leftover = None, -1.0
        for e in _CARTESIAN:
            w = e.copy()
            for b in basis + out:
                w -= (w @ b) * b
            leftover = np.linalg.norm(w)
            if leftover > best_leftover:
                best, best_leftover = w, leftover
        if best is None or best_leftover < 1e-9:  # pragma: no cover - 3 axes always suffice
            raise BoxError("could not construct an orthogonal padding direction")
        out.append(best / best_leftover)
    return out


def needed_axes(atoms: Atoms, cfg: BoxConfig) -> np.ndarray:
    """Which of the three lattice directions must be built or padded.

    The single source of truth for "is this structure non-periodic?".  An axis
    needs work when it is not periodic or its lattice vector is effectively
    absent.  ``mode="native"`` never touches anything; ``mode="box"`` rebuilds
    everything regardless of ``pbc``.

    Pure: takes no copy and mutates nothing, so callers can use it to ask the
    question without paying for the answer.
    """
    if cfg.mode == "native":
        return np.zeros(3, dtype=bool)
    if cfg.mode == "box":
        return np.ones(3, dtype=bool)
    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    return (~pbc) | (lengths < cfg.min_length_tol)


def needs_boxing(atoms: Atoms, cfg: BoxConfig) -> bool:
    """True if this structure would be modified by :func:`ensure_cell`."""
    return bool(needed_axes(atoms, cfg).any())


def ensure_cell(
    atoms: Atoms,
    cfg: BoxConfig,
    *,
    cutoff: float | None = None,
    where: str | None = None,
) -> tuple[Atoms, str]:
    """Return ``(atoms, cell_mode)`` with ``cell_mode`` in native/padded/boxed.

    ``native`` -- untouched.
    ``padded`` -- some directions were given vacuum; the rest kept their lattice
                  vectors.  This is the slab case: a rank-2 cell with
                  ``pbc=[True, True, False]`` must NOT be fully re-boxed, or the
                  in-plane periodicity is destroyed.
    ``boxed``  -- all three directions were built from scratch.

    ``cutoff`` is the descriptor's neighbour cutoff (see
    :attr:`divsel.featurizers.base.Featurizer.neighbor_cutoff`).  When given, a
    structure that is *actually being boxed* must satisfy ``2*vacuum >= cutoff``
    or :class:`~divsel.errors.VacuumTooSmallError` is raised, naming ``where``.
    A structure returned as ``native`` is never checked -- it keeps its own
    cell, so the vacuum has no bearing on it.
    """
    if cfg.mode == "native":
        return atoms, "native"

    cell = np.array(atoms.cell, dtype=float)
    lengths = np.asarray(atoms.cell.lengths(), dtype=float)

    needs = needed_axes(atoms, cfg)
    if not needs.any():
        # Fully periodic: hand it straight back, untouched. No box is built, so
        # the vacuum setting is irrelevant and is deliberately NOT checked.
        return atoms, "native"

    # From here on this structure really is getting a box, so -- and only so --
    # the vacuum has to be large enough for the descriptor's cutoff.
    if not cfg.vacuum_is_sufficient(cutoff):
        raise VacuumTooSmallError(cfg.vacuum_error_message(float(cutoff), where))

    out = atoms.copy()
    positions = out.get_positions()
    # NB: named padded_axes, not needed_axes -- the latter is the module-level
    # function above, and a local of that name would shadow it.
    padded_axes = [int(a) for a in np.flatnonzero(needs)]
    fixed_axes = [int(a) for a in np.flatnonzero(~needs)]

    directions = _orthonormal_completion([cell[a] for a in fixed_axes], len(padded_axes))

    if len(positions) == 0:
        extents = {ax: 0.0 for ax in padded_axes}
    else:
        extents = {
            ax: float(np.ptp(positions @ d))
            for ax, d in zip(padded_axes, directions)
        }

    if cfg.shape == "cubic" and len(padded_axes) == 3:
        # Orientation-independent box: the same side length on every axis.
        side = max(extents.values()) + 2.0 * cfg.vacuum
        for ax, d in zip(padded_axes, directions):
            cell[ax] = side * d
    else:
        for ax, d in zip(padded_axes, directions):
            cell[ax] = (extents[ax] + 2.0 * cfg.vacuum) * d

    volume = abs(float(np.linalg.det(cell)))
    if volume <= 0.0:  # pragma: no cover - guarded by the orthonormal completion
        raise BoxError(
            f"constructed a degenerate cell (volume {volume:g}) for a structure with "
            f"{len(out)} atoms; original cell lengths were {lengths}"
        )

    out.set_cell(cell)
    out.set_pbc(True)
    # Cosmetic under PBC, but it keeps the written .traj sane and stops atoms
    # from sitting exactly on a cell face where wrapping could bite.
    out.center(axis=tuple(padded_axes))

    return out, ("boxed" if needs.all() else "padded")
