"""Turning a batch of frames into a block of feature rows.

Two things here are corrections of real bugs in the original pipeline.

**Row provenance.**  The original iterated ``pool.imap_unordered`` and appended
rows in *completion* order while discarding the index it had computed, and that
index was a running counter of kept structures anyway -- shifted by every
species skip -- so it did not identify a source frame either.  Here every task
carries the row it must land in and the worker result is scattered by that row::

    for rows, block in pool.imap_unordered(_worker, tasks):
        Xb[list(rows)] = block          # row is authoritative

That is strictly better than switching to an ordered ``imap``: no head-of-line
blocking, and the row position is correct by construction rather than by
convention.

**Normalization.**  Per-structure L2, applied here, before anything is
persisted.  It is stateless, so batch 1, batch 50 and last year's seed set all
live in the same space -- which is what makes the seed mechanism meaningful at
all.  Euclidean distance then equals the normalized SOAP-kernel distance,
``d2 = 2(1 - cos)``, bounded in [0, 4].

The original instead fit a ``StandardScaler`` on the stacked all-rounds matrix.
That is impossible to stream, and it also meant the distance between two
candidates changed depending on how many previously-selected structures you
passed in.  It additionally divides each SOAP column by its own standard
deviation, which *upweights* near-zero-variance channels -- the rare-species
cross terms -- so the metric ends up dominated by the least informative columns.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Iterator, Sequence

import numpy as np
from ase import Atoms

from .box import ensure_cell
from .config import BoxConfig
from .errors import FeatureTableError, SpeciesError, VacuumTooSmallError
from .featurizers import Featurizer, featurizer_from_spec
from .featurizers.table import SRC_INFO_KEY
from .frames import FrameIndex, FrameRef
from .log import get_logger

logger = get_logger(__name__)

__all__ = ["l2_normalize", "describe_batch", "describe_atoms_list"]

# Worker-process globals.  Set once per worker by _worker_init.
_WORKER_FEATURIZER: Featurizer | None = None


def l2_normalize(X: np.ndarray, mode: str = "l2") -> np.ndarray:
    """Per-row L2 normalization (``mode='l2'``) or a no-op (``mode='none'``).

    Zero rows are left alone rather than turned into NaN; a genuinely zero
    feature vector means the structure had no atoms the descriptor recognises,
    which is a data problem worth seeing rather than a silent NaN.
    """
    if mode == "none":
        return X
    if mode != "l2":
        raise ValueError(f"unknown normalize mode {mode!r}")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    np.divide(X, norms, out=X, where=norms > 0)
    return X


def _worker_init(spec: dict) -> None:
    global _WORKER_FEATURIZER
    _WORKER_FEATURIZER = featurizer_from_spec(spec)


def _worker_compute(task: tuple[tuple[int, ...], list[Atoms]]):
    rows, atoms_list = task
    assert _WORKER_FEATURIZER is not None
    return rows, _WORKER_FEATURIZER.featurize(atoms_list)


def _chunks(
    items: list[tuple[int, Atoms]], chunk_size: int
) -> Iterator[tuple[tuple[int, ...], list[Atoms]]]:
    for start in range(0, len(items), chunk_size):
        block = items[start : start + chunk_size]
        yield tuple(r for r, _ in block), [a for _, a in block]


def _prepare_frames(
    refs: Sequence[FrameRef],
    frame_index: FrameIndex,
    featurizer: Featurizer,
    box_cfg: BoxConfig,
    on_unknown_species: str,
    on_small_vacuum: str = "error",
    on_invalid_features: str = "error",
) -> tuple[list[tuple[int, Atoms]], list[dict]]:
    """Read, box and screen a batch.  Returns (tasks, per-ref metadata).

    Metadata has one entry per *attempted* frame, in ref order, so skipped
    frames are recorded rather than merely counted.
    """
    meta: list[dict] = [None] * len(refs)  # type: ignore[list-item]
    ready: list[tuple[int, Atoms]] = []
    next_row = 0
    # Only structures that actually get boxed are held to 2*vacuum >= cutoff;
    # ensure_cell decides that per structure, we just supply the cutoff.
    cutoff = featurizer.neighbor_cutoff
    # A table backend looks vectors up by provenance and never touches
    # coordinates, so building a box for it would be pure waste.
    needs_geometry = getattr(featurizer, "needs_geometry", True)

    for slot, atoms in frame_index.iter_slots(refs):
        ref = refs[slot]
        record: dict[str, Any] = {
            "local_row": -1,
            "source_file": str(frame_index.file_path(ref.file_id)),
            "frame_index": int(ref.frame_index),
            "natoms": int(len(atoms)),
            "formula": atoms.get_chemical_formula(mode="hill"),
            "cell_mode": "native",
            "status": "ok",
        }
        where = (
            f"{record['source_file']} frame {ref.frame_index} "
            f"({record['formula']})"
        )
        # Stamp provenance before anything else looks at the structure: the
        # table backend joins on it, and it is the same string gather.py writes
        # into selected.traj, so a seed frame and a candidate frame key alike.
        atoms.info[SRC_INFO_KEY] = f"{record['source_file']}:{ref.frame_index}"

        if not needs_geometry:
            boxed, cell_mode = atoms, "native"
        else:
            try:
                boxed, cell_mode = ensure_cell(
                    atoms, box_cfg, cutoff=cutoff, where=where
                )
            except VacuumTooSmallError:
                # A configuration error, not a property of this frame: every
                # structure needing a box hits it. Abort unless told otherwise.
                if on_small_vacuum == "error":
                    raise
                record["status"] = (
                    f"small_vacuum:2*vac={2 * box_cfg.vacuum:g}<cutoff={cutoff:g}"
                )
                meta[slot] = record
                continue
            except Exception as exc:  # a broken cell is a per-frame problem
                # Broad on purpose -- one corrupt frame must not kill a long
                # run -- but say so, or a systematic fault looks like a clean
                # run that happened to describe nothing.
                logger.warning("could not build a cell for %s: %s", where, exc)
                record["status"] = f"box_error:{exc}"
                meta[slot] = record
                continue
        record["cell_mode"] = cell_mode

        reason = featurizer.check_supported(boxed)
        if reason is not None:
            if on_unknown_species == "error":
                raise SpeciesError(
                    f"{record['source_file']} frame {ref.frame_index} "
                    f"({record['formula']}): {reason}.\n"
                    "Either widen --species to include it, or pass "
                    "--on_unknown_species skip to record and skip such frames."
                )
            record["status"] = f"skipped_species:{reason.split(':', 1)[-1]}"
            meta[slot] = record
            continue

        # A frame the upstream run already dropped keeps that verdict and its
        # original reason; no policy flag applies, because reopening the
        # decision would select over a different set than the vectors cover.
        inherited = featurizer.inherited_status(boxed)
        if inherited is not None:
            record["status"] = inherited
            meta[slot] = record
            continue

        # Distinct from check_supported: the structure is fine, the backend
        # just has no vector for it. Only a table backend can answer here.
        unavailable = featurizer.check_available(boxed)
        if unavailable is not None:
            if on_invalid_features == "error":
                raise FeatureTableError(
                    f"{record['source_file']} frame {ref.frame_index} "
                    f"({record['formula']}): {unavailable}.\n"
                    "Pass --on_invalid_features skip to record and skip such "
                    "frames instead of aborting."
                )
            record["status"] = f"invalid_features:{unavailable}"
            meta[slot] = record
            continue

        record["local_row"] = next_row
        ready.append((next_row, boxed))
        next_row += 1
        meta[slot] = record

    return ready, meta


def describe_batch(
    refs: Sequence[FrameRef],
    frame_index: FrameIndex,
    featurizer: Featurizer,
    *,
    box_cfg: BoxConfig,
    normalize: str = "l2",
    n_procs: int = 1,
    chunk_size: int = 8,
    on_unknown_species: str = "error",
    on_small_vacuum: str = "error",
    on_invalid_features: str = "error",
    pool: Any = None,
) -> tuple[np.ndarray, list[dict]]:
    """Featurize one batch of frames.

    Returns ``(X, meta)`` where ``X`` is ``(n_ok, d)`` float32 (normalized) and
    ``meta`` has one record per attempted frame with ``local_row`` pointing into
    ``X`` (or -1 when the frame was skipped).
    """
    ready, meta = _prepare_frames(
        refs,
        frame_index,
        featurizer,
        box_cfg,
        on_unknown_species,
        on_small_vacuum,
        on_invalid_features,
    )
    d = featurizer.feature_dim
    X = np.empty((len(ready), d), dtype=np.float32)
    if not ready:
        return X, meta

    filled = np.zeros(len(ready), dtype=bool)

    if featurizer.parallel_mode == "process_pool" and n_procs > 1 and len(ready) > 1:
        # chunksize 8, not the original 100: featurization is seconds per
        # structure, so oversized chunks only create tail imbalance at the end
        # of a batch.
        tasks = _chunks(ready, chunk_size)
        own_pool = pool is None
        if own_pool:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(
                processes=min(n_procs, len(ready)),
                initializer=_worker_init,
                initargs=(featurizer.spec(),),
            )
        try:
            for rows, block in pool.imap_unordered(_worker_compute, tasks):
                idx = list(rows)
                X[idx] = block
                filled[idx] = True
        finally:
            if own_pool:
                pool.close()
                pool.join()
    else:
        if featurizer.parallel_mode != "process_pool" and n_procs > 1:
            logger.debug(
                "featurizer %s declares parallel_mode=%s; running in-process "
                "(a CUDA context cannot be forked across a Pool)",
                featurizer.name,
                featurizer.parallel_mode,
            )
        for rows, atoms_list in _chunks(ready, max(1, chunk_size)):
            idx = list(rows)
            X[idx] = featurizer.featurize(atoms_list)
            filled[idx] = True

    if not filled.all():  # pragma: no cover - would indicate a lost task
        missing = int((~filled).sum())
        raise RuntimeError(f"{missing} rows were never filled by the worker pool")

    l2_normalize(X, normalize)
    return X, meta


def describe_atoms_list(
    atoms_list: Sequence[Atoms],
    featurizer: Featurizer,
    *,
    box_cfg: BoxConfig,
    normalize: str = "l2",
    on_unknown_species: str = "skip",
) -> tuple[np.ndarray, list[str]]:
    """Featurize structures already in memory -- used for seed trajectories.

    Deliberately the same code path as the batch case (same boxing, same
    species screening, same normalization), so a seed set can never be
    described by a subtly different recipe than the candidates.  The original
    had two separate SOAP implementations for exactly this, which is how they
    drifted apart.
    """
    kept: list[Atoms] = []
    statuses: list[str] = []
    cutoff = featurizer.neighbor_cutoff
    needs_geometry = getattr(featurizer, "needs_geometry", True)
    for i, atoms in enumerate(atoms_list):
        where = f"seed structure {i} ({atoms.get_chemical_formula(mode='hill')})"
        if needs_geometry:
            boxed, _ = ensure_cell(atoms, box_cfg, cutoff=cutoff, where=where)
        else:
            boxed = atoms
        reason = featurizer.check_supported(boxed)
        if reason is not None:
            if on_unknown_species == "error":
                raise SpeciesError(f"seed structure {atoms.get_chemical_formula()}: {reason}")
            statuses.append(reason)
            continue

        # A seed the backend cannot serve is always fatal, whatever the
        # candidate policy says. Silently dropping one would quietly weaken the
        # constraint that previously-selected structures are never re-picked.
        unavailable = featurizer.check_available(boxed)
        if unavailable is not None:
            raise FeatureTableError(
                f"{where}: {unavailable}.\n"
                f"Seed structures are matched to feature rows by "
                f"atoms.info['{SRC_INFO_KEY}'] = "
                f"{atoms.info.get(SRC_INFO_KEY, '<missing>')!r}. Extract "
                "features over the frames these seeds came from, so the table "
                "covers the candidates and the seed set alike."
            )

        statuses.append("ok")
        kept.append(boxed)

    if not kept:
        return np.empty((0, featurizer.feature_dim), dtype=np.float32), statuses
    X = featurizer.featurize(kept).astype(np.float32, copy=False)
    l2_normalize(X, normalize)
    return X, statuses
