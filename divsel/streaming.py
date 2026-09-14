"""The batch loop: describe, carry forward, select.

Two decoupled loops.  Featurization is the expensive part and must stream;
selection runs against a bounded pool.

    pool_X, pool_rows = empty, []
    for b, batch in enumerate(index.batches(batch_size)):
        Xb, meta = describe_batch(...)          # peak RAM = batch_size * d
        store.append(Xb, meta)                  # persist everything
        seed = vstack_nonempty(X_user_seed, pool_X)     # <-- THE one mechanism
        picks = select_diverse(Xb, quota(b), X_seed=seed, method="fps")
        pool_X, pool_rows = extend with Xb[picks]
        if len(pool_X) > pool_cap: evict by seeded FPS
    final = select_diverse(pool_X, n_select, X_seed=X_user_seed, method=<user's>)

``--selected_images`` and the carry-forward meet on the ``vstack_nonempty``
line and are indistinguishable to :func:`divsel.selection.select_diverse`.
They differ only in policy, and that policy lives here:

* **user seed points are permanent** -- never returned, never reconsidered;
* **pool points are provisional** -- they are candidates for the final pass, so
  a bad early pick can still be discarded.

Do not "simplify" this by making each batch's picks a hard seed.  Batch 0 is
chosen against an empty seed, i.e. with zero global context; locking it in is
precisely the order bias the pool exists to remove.

What is guaranteed
------------------
* peak memory O((batch_size + |pool| + n_seed) * d);
* no seed structure is ever re-selected;
* exactly ``n_select`` outputs when enough candidates exist;
* with a single batch the result is the exact, un-pooled answer.

What is not
-----------
* Multi-batch is not the same as single-batch.  FPS's k-center
  2-approximation does not survive batching and there is no known bound for
  pooled streaming FPS.  With ``--shuffle`` and ``pool_alpha >= 4`` it is
  empirically close; that is an empirical claim, not a theorem.
* ``--method kmeans`` over several batches selects from an FPS-thinned pool,
  which is density-distorted -- FPS deliberately flattens density and k-means
  is a density method.  ``--pool_stage random`` is the density-unbiased
  alternative and is the better choice there.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np

from . import manifest as manifest_mod
from .box import needs_boxing
from .gather import gather as gather_frames
from .config import (
    BoxConfig,
    SamplingConfig,
    SelectConfig,
    SoapConfig,
    StreamConfig,
    descriptor_id,
    format_descriptor_diff,
    species_symbols,
)
from .descriptors import describe_atoms_list, describe_batch
from .errors import DescriptorMismatchError, DivselError, VacuumTooSmallError
from .featurizers import IMPLEMENTED, get_featurizer
from .frames import expand_patterns, peek_first_frame, scan_sources
from .log import get_logger
from .selection import fps_order_scores, select_diverse, vstack_nonempty
from .store import DescriptorStore, seed_cache_path

logger = get_logger(__name__)

__all__ = ["run_selection", "run_describe", "quota_for"]

_STAMP_FIELDS = "divsel_descriptor_fields"

#: frames per worker task; small enough to avoid tail imbalance at the
#: end of a batch, large enough that the per-task overhead is negligible
_CHUNK_SIZE = 8


def quota_for(batch: int, n_batches: int, pool_target: int) -> int:
    """Per-batch quota as an integer prefix split.

    Quotas sum to exactly ``pool_target`` with no floating-point drift, and no
    batch ever gets zero.
    """
    if n_batches <= 1:
        return int(pool_target)
    lo = (pool_target * batch) // n_batches
    hi = (pool_target * (batch + 1)) // n_batches
    return max(1, hi - lo)


# --------------------------------------------------------------------------
# setup shared by `describe` and `select`
# --------------------------------------------------------------------------


def _probe_boxing_requirement(
    patterns: Sequence[str],
    box_cfg: BoxConfig,
    cutoff: float | None,
    on_small_vacuum: str = "error",
) -> None:
    """Fail fast if the vacuum is too small *and* something will be boxed.

    The authoritative check is per structure, inside
    :func:`divsel.box.ensure_cell`.  This is only an early warning so the
    common case -- every frame in a file has the same periodicity -- reports
    before the scan rather than partway through the first batch.

    Costs nothing unless the vacuum is already known to be insufficient; only
    then does it read one frame per input file to find out whether that even
    matters.  A purely periodic dataset is never boxed and never blocked.
    """
    if cutoff is None or box_cfg.mode == "native":
        return
    if box_cfg.vacuum_is_sufficient(cutoff):
        return
    if on_small_vacuum == "skip":
        # The user has already said they would rather lose those structures
        # than change the flag; the per-structure path records each one.
        logger.warning(
            "2*vacuum (%g A) is below the cutoff (%g A); structures that need "
            "a box will be SKIPPED and recorded as small_vacuum in frames.csv",
            2 * box_cfg.vacuum,
            cutoff,
        )
        return

    if box_cfg.mode == "box":
        # Forced boxing applies to every structure regardless of its pbc, so
        # there is nothing to probe -- it is always a violation.
        raise VacuumTooSmallError(
            box_cfg.vacuum_error_message(
                float(cutoff), "--cell_mode box gives every structure a box, so this run"
            )
        )

    for path in expand_patterns(patterns):
        atoms = peek_first_frame(path)
        if atoms is None or not needs_boxing(atoms, box_cfg):
            continue
        raise VacuumTooSmallError(
            box_cfg.vacuum_error_message(
                float(cutoff),
                f"{path} frame 0 ({atoms.get_chemical_formula(mode='hill')})",
            )
        )

    logger.debug(
        "2*vacuum (%g A) is below the cutoff (%g A), but no input file starts "
        "with a structure that needs a box -- continuing; the per-structure "
        "check still applies to every frame",
        2 * box_cfg.vacuum,
        cutoff,
    )


class SeedSource(NamedTuple):
    """One ``--selected_images`` file: its frames and the stamps they carry.

    ``stamps`` maps a descriptor id to ``(fields, where)``, so the ordinary case
    -- every frame of the file written by one divsel run -- collapses to a
    single entry, and a file assembled by concatenating rounds by hand still
    reports each distinct identity in it.
    """

    path: str
    images: list
    stamps: dict[str, tuple[dict, str]]


def _read_seed_structures(paths: Sequence[str]) -> list[SeedSource]:
    """Load each previously-selected trajectory together with its stamp(s).

    Per source file, and per distinct stamp within a file -- *not* one stamp for
    the whole set.  Accumulating one trajectory per active-learning round is the
    normal way to use ``--selected_images``, and catching the round that was
    computed with different parameters is the entire reason the stamp exists;
    keeping only the first stamp seen would check the one file that never needs
    checking and wave through every later one.
    """
    from ase.io import read as ase_read

    out: list[SeedSource] = []
    for path in paths:
        got = ase_read(str(path), index=":")
        got = got if isinstance(got, list) else [got]
        stamps: dict[str, tuple[dict, str]] = {}
        for i, atoms in enumerate(got):
            raw = atoms.info.get(_STAMP_FIELDS)
            if not raw:
                continue
            try:
                fields = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except Exception:  # pragma: no cover - a corrupt stamp is not fatal
                continue
            stamps.setdefault(descriptor_id(fields), (fields, f"{path} frame {i}"))
        out.append(SeedSource(str(path), got, stamps))
    return out


def _seed_identity(sources: Sequence[SeedSource]) -> dict | None:
    """The one descriptor identity the whole seed set agrees on.

    ``None`` means no seed frame carried a stamp at all -- an unverifiable but
    permitted case, warned about by the caller.  Disagreement *within* the seed
    set, on the other hand, is fatal: descriptors from two parameter sets do not
    live in one metric space, so distances across them are meaningless and a
    selection made against their union would be wrong rather than approximate.
    """
    seen: dict[str, tuple[dict, str]] = {}
    for src in sources:
        for did, entry in src.stamps.items():
            seen.setdefault(did, entry)

    if not seen:
        return None
    if len(seen) > 1:
        # Report the two lowest ids so the message is deterministic; the diff
        # names the fields, the labels name the frames they came from.
        (id_a, (fields_a, where_a)), (id_b, (fields_b, where_b)) = sorted(seen.items())[:2]
        raise DescriptorMismatchError(
            format_descriptor_diff(
                fields_a,
                fields_b,
                seed_id=id_a,
                this_id=id_b,
                seed_label=where_a,
                this_label=where_b,
                header=(
                    f"the --selected_images files disagree with each other: "
                    f"{len(seen)} different descriptor parameter sets across "
                    f"{len(sources)} file(s)."
                ),
                advice=(
                    "Pass only previously-selected files computed with the same "
                    "parameters, or recompute the odd one out. Their feature "
                    "vectors are not in the same space, so seeding with both "
                    "would silently corrupt this round's selection."
                ),
            )
        )
    return next(iter(seen.values()))[0]


def _build_setup(
    patterns: Sequence[str],
    *,
    featurizer: str,
    soap_cfg: SoapConfig,
    sampling: SamplingConfig,
    box_cfg: BoxConfig,
    normalize: str,
    selected_images: Sequence[str],
    on_small_vacuum: str = "error",
    cache_root: str | Path | None = None,
) -> dict:
    if featurizer not in IMPLEMENTED:
        # Fail before the scan, not after it: a clear message beats a crash
        # twenty minutes into a run.
        get_featurizer(featurizer)  # raises NotImplementedError with guidance

    selected_sources: list[SeedSource] = []
    if selected_images:
        selected_sources = _read_seed_structures(selected_images)
        logger.info(
            "seed set: %d structure(s) from %d file(s)",
            sum(len(s.images) for s in selected_sources),
            len(selected_sources),
        )
    # Raises if the seed files disagree among themselves, before anything
    # expensive happens.
    seed_stamp = _seed_identity(selected_sources)

    # When continuing from a seed, the species set comes from the seed, not from
    # this round's data.  Re-deriving it would silently change the descriptor
    # length and make the two incomparable.
    species_from_seed = None
    if seed_stamp and seed_stamp.get("species_z"):
        species_from_seed = tuple(int(z) for z in seed_stamp["species_z"])
        if soap_cfg.species is None:
            soap_cfg = soap_cfg.with_species(species_from_seed)
            logger.info(
                "species taken from the seed set: %s",
                species_symbols(soap_cfg.species),
            )

    need_species = soap_cfg.species is None

    # Constructed before the scan so the boxing probe can read its cutoff;
    # prepare() still runs afterwards, once the species union is known.
    featurizer_obj = get_featurizer(featurizer, cfg=soap_cfg)
    _probe_boxing_requirement(
        patterns, box_cfg, featurizer_obj.neighbor_cutoff, on_small_vacuum
    )

    frame_index = scan_sources(
        patterns,
        stride=sampling.stride,
        max_frames=sampling.max_frames,
        shuffle=sampling.shuffle,
        seed=sampling.seed,
        collect_species=need_species,
    )

    featurizer_obj.prepare(frame_index)

    fields = dict(featurizer_obj.identity_fields())
    fields["featurizer"] = featurizer_obj.name
    fields["normalize"] = normalize
    did = descriptor_id(fields)
    logger.info(
        "descriptor: %s d=%d id=%s",
        featurizer_obj.name,
        featurizer_obj.feature_dim,
        did,
    )
    if need_species:
        logger.debug(
            "species union from scan: %s",
            species_symbols(soap_cfg.species or featurizer_obj.cfg.species),
        )

    if seed_stamp is not None:
        manifest_mod.validate_compatible(seed_stamp, fields, seed_label="seed")
    elif selected_sources:
        logger.warning(
            "seed structures carry no divsel stamp; recomputing their descriptors "
            "with the current parameters. Compatibility is unverified -- if they "
            "were produced with different parameters the selection will be wrong."
        )

    return {
        "frame_index": frame_index,
        "featurizer": featurizer_obj,
        "fields": fields,
        "descriptor_id": did,
        "selected_sources": selected_sources,
        "soap_cfg": featurizer_obj.cfg if hasattr(featurizer_obj, "cfg") else soap_cfg,
        "cache_root": cache_root,
    }


def _seed_descriptors(
    setup: dict, box_cfg: BoxConfig, normalize: str
) -> tuple[np.ndarray, list[int]]:
    """Describe the previously-selected structures, one cache entry per file.

    Returns ``(X, n_per_source)`` with ``sum(n_per_source) == len(X)``, so the
    manifest can say which file contributed which rows.

    Caching per *file* rather than per seed *set* is what makes an
    active-learning campaign cheap: a seed set that grows by one trajectory per
    round costs one new featurization, not a full recompute of every round so
    far.  :func:`divsel.store.seed_cache_path` is already keyed on
    (descriptor id, file fingerprint), so this needs no new bookkeeping.
    """
    featurizer = setup["featurizer"]
    sources: list[SeedSource] = setup["selected_sources"]
    d = featurizer.feature_dim
    if not sources:
        return np.empty((0, d), dtype=np.float32), []

    did = setup["descriptor_id"]
    cache_root = setup["cache_root"]

    blocks: list[np.ndarray] = []
    for src in sources:
        path = seed_cache_path(did, src.path, cache_root=cache_root)
        if path.exists():
            X = np.load(path)
            # Rows may legitimately be fewer than frames -- a structure whose
            # species the descriptor does not cover is skipped below -- so this
            # bounds the count rather than demanding equality. The cache key
            # already covers the parameters and the file's size and mtime.
            if X.ndim == 2 and X.shape[1] == d and X.shape[0] <= len(src.images):
                logger.debug("seed cache hit: %s (%d row(s))", path, X.shape[0])
                blocks.append(X.astype(np.float32, copy=False))
                continue
            logger.debug("seed cache entry at %s has the wrong shape, recomputing", path)

        X, statuses = describe_atoms_list(
            src.images,
            featurizer,
            box_cfg=box_cfg,
            normalize=normalize,
            on_unknown_species="skip",
        )
        n_skipped = sum(1 for s in statuses if s != "ok")
        if n_skipped:
            logger.warning(
                "%d seed structure(s) in %s could not be described and were ignored",
                n_skipped,
                src.path,
            )
        if len(X):
            # Nothing is cached for a file that yielded no rows: that is a
            # configuration problem, and recomputing it re-emits the warning
            # above instead of hiding it behind a cache hit.
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, X)
            logger.debug("seed cache write: %s (%d row(s))", path, len(X))
        blocks.append(X)

    n_per_source = [int(len(b)) for b in blocks]
    stacked = np.vstack(blocks) if any(n_per_source) else np.empty((0, d), np.float32)
    return stacked.astype(np.float32, copy=False), n_per_source


def _make_pool(featurizer, n_procs: int):
    if featurizer.parallel_mode != "process_pool" or n_procs <= 1:
        return None
    ctx = mp.get_context("spawn")
    from .descriptors import _worker_init

    return ctx.Pool(processes=n_procs, initializer=_worker_init, initargs=(featurizer.spec(),))


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------


def run_describe(
    patterns: Sequence[str],
    out_dir: str | Path,
    *,
    featurizer: str = "soap",
    soap_cfg: SoapConfig | None = None,
    box_cfg: BoxConfig | None = None,
    sampling: SamplingConfig | None = None,
    stream: StreamConfig | None = None,
    normalize: str = "l2",
    on_small_vacuum: str = "error",
    argv: list[str] | None = None,
) -> dict:
    """Compute and persist descriptors without selecting anything.

    ``out_dir`` is a *directory*; ``descriptors.npy``, ``frames.csv`` and
    ``run_manifest.json`` are written inside it.
    """
    return _run(
        patterns,
        out_dir,
        select_cfg=None,
        featurizer=featurizer,
        soap_cfg=soap_cfg or SoapConfig(),
        box_cfg=box_cfg or BoxConfig(),
        sampling=sampling or SamplingConfig(),
        stream=stream or StreamConfig(),
        normalize=normalize,
        on_small_vacuum=on_small_vacuum,
        selected_images=(),
        argv=argv,
    )


def run_selection(
    patterns: Sequence[str],
    out_dir: str | Path,
    select_cfg: SelectConfig,
    *,
    featurizer: str = "soap",
    soap_cfg: SoapConfig | None = None,
    box_cfg: BoxConfig | None = None,
    sampling: SamplingConfig | None = None,
    stream: StreamConfig | None = None,
    selected_images: Sequence[str] = (),
    cache_root: str | Path | None = None,
    on_small_vacuum: str = "error",
    argv: list[str] | None = None,
) -> dict:
    """Describe, select, and write every output.  Returns the manifest dict.

    ``out_dir`` is a *directory*: ``selected.traj``, ``selected.csv``,
    ``frames.csv``, ``descriptors.npy`` and ``run_manifest.json`` are written
    inside it.  (Contrast :func:`divsel.gather.gather`, whose ``out_traj`` is a
    single file.)

    ``selected_images`` is the structures already chosen in previous rounds --
    one path or several.  They seed the selection, so this round's picks are
    diverse with respect to them, and they are never re-selected.  Every file
    must have been described with the same parameters; that is checked against
    the stamp each frame carries.
    """
    return _run(
        patterns,
        out_dir,
        select_cfg=select_cfg,
        featurizer=featurizer,
        soap_cfg=soap_cfg or SoapConfig(),
        box_cfg=box_cfg or BoxConfig(),
        sampling=sampling or SamplingConfig(),
        stream=stream or StreamConfig(),
        normalize=select_cfg.normalize,
        on_small_vacuum=on_small_vacuum,
        selected_images=selected_images,
        cache_root=cache_root,
        argv=argv,
    )


def _run(
    patterns: Sequence[str],
    out_dir: str | Path,
    *,
    select_cfg: SelectConfig | None,
    featurizer: str,
    soap_cfg: SoapConfig,
    box_cfg: BoxConfig,
    sampling: SamplingConfig,
    stream: StreamConfig,
    normalize: str,
    selected_images: Sequence[str],
    on_small_vacuum: str = "error",
    cache_root: str | Path | None = None,
    argv: list[str] | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = manifest_mod.run_info(argv)

    setup = _build_setup(
        patterns,
        featurizer=featurizer,
        soap_cfg=soap_cfg,
        sampling=sampling,
        box_cfg=box_cfg,
        normalize=normalize,
        selected_images=selected_images,
        on_small_vacuum=on_small_vacuum,
        cache_root=cache_root,
    )
    frame_index = setup["frame_index"]
    featurizer_obj = setup["featurizer"]
    did = setup["descriptor_id"]

    X_user_seed, n_per_source = _seed_descriptors(setup, box_cfg, normalize)
    n_seed = int(len(X_user_seed))

    n_frames = len(frame_index)
    if n_frames == 0:
        raise DivselError("no frames to process after stride/max_frames")

    batch_size = stream.batch_size
    n_batches = frame_index.n_batches(batch_size)
    n_select = select_cfg.n_select if select_cfg else 0
    pool_cap = stream.resolved_pool_cap(n_select) if select_cfg else 0
    pool_target = min(stream.pool_alpha * n_select, n_frames) if select_cfg else 0

    logger.info(
        "describing %d frame(s) in %d batch(es) of up to %d with %d process(es)",
        n_frames,
        n_batches,
        batch_size if batch_size > 0 else n_frames,
        stream.n_procs,
    )

    store = DescriptorStore(
        out_dir,
        featurizer_obj.feature_dim,
        did,
        n_rows_max=n_frames if sampling.scan else None,
    )
    # More workers than task chunks just pays spawn cost for nothing.
    n_chunks = max(1, -(-n_frames // _CHUNK_SIZE))
    pool = _make_pool(featurizer_obj, min(stream.n_procs, n_chunks))
    rng = np.random.default_rng(sampling.seed)

    pool_X = np.empty((0, featurizer_obj.feature_dim), dtype=np.float32)
    pool_rows: list[int] = []
    quotas: list[int] = []
    single_batch_X: np.ndarray | None = None
    single_batch_base = 0
    n_evictions = 0

    try:
        for b, batch_refs in enumerate(frame_index.batches(batch_size)):
            base_row = store.n_rows
            Xb, meta = describe_batch(
                batch_refs,
                frame_index,
                featurizer_obj,
                box_cfg=box_cfg,
                normalize=normalize,
                n_procs=stream.n_procs,
                chunk_size=_CHUNK_SIZE,
                on_unknown_species=soap_cfg.on_unknown_species,
                on_small_vacuum=on_small_vacuum,
                pool=pool,
            )
            store.append(Xb, meta)
            logger.debug(
                "batch %d/%d: %d frame(s) -> %d row(s) (total %d)",
                b + 1,
                n_batches,
                len(batch_refs),
                len(Xb),
                store.n_rows,
            )

            if select_cfg is None or len(Xb) == 0:
                continue

            if n_batches == 1:
                # No streaming: keep the block and select from it directly, so
                # the answer is the exact un-pooled one for either method.
                single_batch_X = Xb
                single_batch_base = base_row
                continue

            quota = quota_for(b, n_batches, pool_target)
            quotas.append(int(quota))
            seed_for_batch = vstack_nonempty(X_user_seed, pool_X)

            if stream.pool_stage == "random":
                take = min(quota, len(Xb))
                picks = np.sort(rng.choice(len(Xb), size=take, replace=False))
            else:
                picks = select_diverse(
                    Xb,
                    quota,
                    X_seed=seed_for_batch,
                    method="fps",
                    random_state=select_cfg.random_state,
                    strict_determinism=select_cfg.strict_determinism,
                ).indices

            pool_X = np.vstack([pool_X, Xb[picks]]) if len(pool_X) else Xb[picks].copy()
            pool_rows.extend(base_row + int(i) for i in picks)

            if len(pool_X) > pool_cap:
                keep = select_diverse(
                    pool_X,
                    pool_cap,
                    X_seed=X_user_seed if n_seed else None,
                    method="fps",
                    init=("index", 0),
                    random_state=select_cfg.random_state,
                ).indices
                # Sorting preserves pick order, which the single-batch identity
                # argument relies on.
                keep = np.sort(keep)
                pool_X = pool_X[keep]
                pool_rows = [pool_rows[i] for i in keep]
                n_evictions += 1
                logger.debug("pool evicted down to %d entries", len(pool_X))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        n_rows = store.finalize()

    if n_rows == 0:
        raise DivselError(
            f"no frames could be described ({n_frames} were attempted). "
            f"See {out_dir / 'frames.csv'} -- the status column says why for "
            "each one. Describing nothing is never a useful result, so this is "
            "an error rather than an empty output directory."
        )
    logger.info("described %d row(s) -> %s", n_rows, out_dir / "descriptors.npy")

    manifest: dict[str, Any] = {
        "schema_version": manifest_mod.SCHEMA_VERSION,
        "run": run,
        "env": manifest_mod.environment_info(),
        "descriptor": {**setup["fields"], "descriptor_id": did},
        "box": {
            "mode": box_cfg.mode,
            "vacuum": box_cfg.vacuum,
            "shape": box_cfg.shape,
            # The cutoff the vacuum had to satisfy, and only for structures
            # that were actually boxed -- see box.ensure_cell.
            "cutoff": featurizer_obj.neighbor_cutoff,
            "on_small_vacuum": on_small_vacuum,
        },
        "inputs": frame_index.source_summary(),
        "sampling": {
            **asdict(sampling),
            "n_before_stride": frame_index.n_before_stride,
            "n_after_stride": n_frames,
            "n_described": int(n_rows),
        },
        "seed_set": {
            "sources": [s.path for s in setup["selected_sources"]],
            # Rows contributed per source, aligned with `sources` and summing to
            # n_seed. Fewer than that file's frame count means some were skipped.
            "n_per_source": list(n_per_source),
            "n_frames_per_source": [len(s.images) for s in setup["selected_sources"]],
            "n_seed": n_seed,
            "descriptor_id": did,
        },
        "streaming": {
            **asdict(stream),
            "n_batches": int(n_batches),
            "per_batch_quota": quotas,
            "pool_cap": int(pool_cap),
            "pool_evictions": int(n_evictions),
        },
    }

    # Per-frame skip accounting, read back from the file we just wrote.
    import pandas as pd

    frames_df = pd.read_csv(out_dir / "frames.csv")
    status_counts = frames_df["status"].value_counts().to_dict()
    manifest["sampling"]["status_counts"] = {str(k): int(v) for k, v in status_counts.items()}

    if select_cfg is None:
        manifest_mod.write_manifest(out_dir / "run_manifest.json", manifest)
        _log_summary(manifest, out_dir, selected=None)
        return manifest

    # ---- final selection -------------------------------------------------
    if n_batches == 1:
        cand_X = single_batch_X if single_batch_X is not None else np.empty(
            (0, featurizer_obj.feature_dim), np.float32
        )
        cand_rows = [single_batch_base + i for i in range(len(cand_X))]
        init = select_cfg.fps_init
    else:
        cand_X, cand_rows = pool_X, pool_rows
        # Starting at pool row 0 is what makes the pooled result equal the
        # un-pooled one when there is a single batch, and keeps multi-batch
        # runs order-stable.
        init = ("index", 0)

    backend_kwargs: dict[str, Any] = {}
    if select_cfg.method == "fps":
        backend_kwargs["init"] = init
    else:
        backend_kwargs.update(
            k=select_cfg.resolved_k(),
            per_cluster=select_cfg.kmeans_per_cluster,
            fill=select_cfg.kmeans_fill,
            trim=select_cfg.kmeans_trim,
            n_init=select_cfg.kmeans_n_init,
            minibatch_threshold=select_cfg.minibatch_threshold,
        )

    result = select_diverse(
        cand_X,
        n_select,
        X_seed=X_user_seed if n_seed else None,
        method=select_cfg.method,
        random_state=select_cfg.random_state,
        min_dist2=select_cfg.min_dist2,
        strict_determinism=select_cfg.strict_determinism,
        **backend_kwargs,
    )

    if select_cfg.method == "kmeans" and result.aux.get("n_from_kmeans") == 0 and n_seed:
        logger.warning(
            "k-means contributed 0 picks: all %d clusters already contain a seed "
            "structure, so the whole selection came from the FPS top-up. This is "
            "expected when the seed is comparable in size to --n_select; raise "
            "--kmeans_k if you want k-means to keep a say.",
            result.aux.get("n_covered_clusters", 0),
        )

    chosen_rows = [int(cand_rows[i]) for i in result.indices]
    clusters = result.aux.get("cluster_of") or None

    # Coverage radius: FPS gets it natively; recompute for k-means so the
    # diagnostic is comparable across methods.
    if select_cfg.method == "fps":
        profile = result.scores
    else:
        profile = fps_order_scores(cand_X[result.indices], X_user_seed if n_seed else None)

    selected_df = _write_selected(
        out_dir, chosen_rows, result, frames_df, profile, clusters
    )

    n_written = gather_frames(
        selected_df.to_dict("records"),
        out_dir / "selected.traj",
        descriptor_id=did,
        run_id=run["run_id"],
        extra_info={_STAMP_FIELDS: json.dumps(setup["fields"], sort_keys=True)},
    )

    finite = profile[np.isfinite(profile)]
    manifest["selection"] = {
        "method": select_cfg.method,
        "n_select": n_select,
        **{k: v for k, v in asdict(select_cfg).items() if k not in ("method", "n_select")},
        "aux": {
            k: v
            for k, v in result.aux.items()
            if not isinstance(v, np.ndarray)
            and not (isinstance(v, list) and len(v) > 50)
        },
    }
    manifest["results"] = {
        "n_selected": int(len(chosen_rows)),
        "n_written_to_traj": int(n_written),
        "shortfall_reason": result.shortfall_reason,
        "coverage_radius_min_d2": float(finite.min()) if finite.size else None,
        "coverage_radius_max_d2": float(finite.max()) if finite.size else None,
        "outputs": {
            "descriptors": str(out_dir / "descriptors.npy"),
            "frames": str(out_dir / "frames.csv"),
            "selected_csv": str(out_dir / "selected.csv"),
            "selected_traj": str(out_dir / "selected.traj"),
        },
    }
    manifest_mod.write_manifest(out_dir / "run_manifest.json", manifest)
    _log_summary(manifest, out_dir, selected=len(chosen_rows))
    return manifest


def _write_selected(out_dir, chosen_rows, result, frames_df, profile, clusters):
    import pandas as pd

    by_row = frames_df[frames_df["row"] >= 0].set_index("row")
    records = []
    for rank, row in enumerate(chosen_rows):
        src = by_row.loc[row]
        records.append(
            {
                "rank": rank,
                "row": int(row),
                "source_file": str(src["source_file"]),
                "frame_index": int(src["frame_index"]),
                "natoms": int(src["natoms"]),
                "formula": str(src["formula"]),
                "pick_stage": _pick_stage(result, rank),
                "cluster": int(clusters[rank]) if clusters is not None else -1,
                "score": float(profile[rank]) if rank < len(profile) else float("nan"),
            }
        )
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_dir / "selected.csv", index=False)
    return df


def _pick_stage(result, rank: int) -> str:
    if result.method == "fps":
        return "fps"
    n_km = int(result.aux.get("n_from_kmeans", 0))
    return "kmeans" if rank < n_km else "fps_topup"


def _log_summary(manifest: dict, out_dir: Path, selected: int | None) -> None:
    sampling = manifest["sampling"]
    counts = sampling.get("status_counts", {})
    skipped = {k: v for k, v in counts.items() if k != "ok"}
    logger.info(
        "summary: %d frame(s) described, %d skipped%s",
        sampling["n_described"],
        sum(skipped.values()),
        f" ({skipped})" if skipped else "",
    )
    if selected is not None:
        res = manifest["results"]
        logger.info(
            "summary: %d selected (coverage radius d2 min=%s), outputs in %s",
            selected,
            (
                f"{res['coverage_radius_min_d2']:.4g}"
                if res["coverage_radius_min_d2"] is not None
                else "n/a"
            ),
            out_dir,
        )
        if res["shortfall_reason"]:
            logger.warning(
                "returned fewer than requested: %s", res["shortfall_reason"]
            )
