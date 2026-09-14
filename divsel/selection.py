"""The one shared selection primitive.

Everything else in divsel is plumbing around :func:`select_diverse`.  It is a
pure function of its arguments -- no I/O, no globals, no configuration objects --
so it can be reasoned about and tested on its own.

THE SEED SET IS ONE MECHANISM WITH TWO ROLES.  ``X_seed`` means "descriptors I
already hold; stay away from these, and never return them".  Two callers use it:

* ``--selected_images``: structures chosen in previous active-learning rounds.
  These are **permanent** -- they are never returned and never reconsidered.
* the streaming carry-forward: structures picked from earlier batches of the
  current run.  These are **provisional**.

The primitive cannot tell the difference, and must not try to.  The distinction
is a *policy* that lives in :mod:`divsel.streaming`, which keeps provisional
picks revocable by pooling them and running one final global pass.  If someone
later "simplifies" that by locking each batch's picks in as a hard seed, batch 0
-- chosen against an empty seed, i.e. with zero global context -- becomes
irreversible, and the order bias that the pool exists to fix comes straight back.

Distances are squared Euclidean throughout.  Any threshold you pass (``min_dist2``)
is therefore in squared units.  With the default L2 normalization,
``d2 = 2 - 2*cos(a, b)`` and lies in [0, 4]; chemically similar frames sit
around d2 < 0.1, which is the scale to think in when setting ``--min_dist2``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .log import get_logger

logger = get_logger(__name__)

__all__ = [
    "SelectionResult",
    "select_diverse",
    "min_sqdist_to_set",
    "fps_order_scores",
    "vstack_nonempty",
]


# --------------------------------------------------------------------------
# result type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of one selection.

    ``indices``   int64, indices INTO ``X_cand``, in pick order.
    ``scores``    per-pick diagnostic.  For FPS this is the min-distance to the
                  already-chosen set at the moment of the pick -- a monotone
                  non-increasing coverage-radius profile.  When it plateaus near
                  zero the space is saturated and further picks are near
                  duplicates.  For k-means it is the representative's squared
                  distance to its centroid.
    ``shortfall_reason``  None when exactly ``n_requested`` were returned.
    """

    indices: np.ndarray
    scores: np.ndarray
    method: str
    n_requested: int
    shortfall_reason: str | None = None
    aux: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.indices.size)

    @property
    def coverage_radius(self) -> float:
        """Smallest finite score, i.e. how well the last pick was still separated."""
        finite = self.scores[np.isfinite(self.scores)]
        return float(finite.min()) if finite.size else float("nan")


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------


def vstack_nonempty(*arrays: np.ndarray | None) -> np.ndarray | None:
    """Stack the arrays that are neither None nor empty; None if nothing remains.

    This is the single line where a user seed set and the streaming carry-forward
    pool become one array.
    """
    parts = [a for a in arrays if a is not None and len(a) > 0]
    if not parts:
        return None
    if len(parts) == 1:
        return np.asarray(parts[0])
    return np.vstack([np.asarray(p) for p in parts])


def _as2d(X: np.ndarray | None, name: str) -> np.ndarray | None:
    if X is None:
        return None
    X = np.asarray(X)
    if X.size == 0:
        return None
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, d), got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or inf")
    return np.ascontiguousarray(X)


def min_sqdist_to_set(
    X: np.ndarray, S: np.ndarray, *, chunk: int = 256, out: np.ndarray | None = None
) -> np.ndarray:
    """Min squared Euclidean distance from every row of ``X`` to any row of ``S``.

    Chunked over ``S`` so peak memory is O(len(X) * chunk) rather than
    O(len(X) * len(S)); no full pairwise matrix is ever formed.
    """
    X = np.asarray(X)
    S = np.asarray(S)
    n = X.shape[0]
    dmin = np.full(n, np.inf, dtype=np.float64) if out is None else out
    if S.size == 0:
        return dmin

    sq_x = np.einsum("ij,ij->i", X, X).astype(np.float64)
    sq_s = np.einsum("ij,ij->i", S, S).astype(np.float64)
    for s0 in range(0, S.shape[0], chunk):
        block = S[s0 : s0 + chunk]
        # ||x-s||^2 = ||x||^2 + ||s||^2 - 2 x.s
        d2 = sq_x[:, None] + sq_s[None, s0 : s0 + block.shape[0]] - 2.0 * (X @ block.T)
        np.minimum(dmin, d2.min(axis=1), out=dmin)
    # The expansion can go slightly negative for near-duplicate rows.
    np.maximum(dmin, 0.0, out=dmin)
    return dmin


@contextlib.contextmanager
def _maybe_single_threaded(enabled: bool):
    """Pin BLAS to one thread so argmax ties cannot flip with thread count.

    BLAS gemm/gemv results can differ in the last bits depending on how the work
    is split across threads.  In 5940-dimensional continuous data an exact tie is
    vanishingly unlikely, so this is off by default -- but when it is on we can
    state reproducibility without hedging.  Selection is cheap next to the
    descriptor computation, so the cost is negligible.
    """
    if not enabled:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl is a hard dependency
        logger.warning("threadpoolctl unavailable; --strict_determinism has no effect")
        yield
        return
    with threadpool_limits(limits=1):
        yield


def _parse_init(init: Any) -> tuple[str, int | None]:
    """Normalize an fps init spec to ``(kind, value)``."""
    if init is None:
        return ("farthest_from_mean", None)
    if isinstance(init, tuple):
        kind, value = init
        return (str(kind), int(value))
    text = str(init)
    if text.startswith("index:"):
        return ("index", int(text.split(":", 1)[1]))
    if text in ("farthest_from_mean", "random"):
        return (text, None)
    raise ValueError(
        f"unknown fps_init {init!r}; expected 'farthest_from_mean', 'random' or 'index:J'"
    )


# --------------------------------------------------------------------------
# farthest-point sampling
# --------------------------------------------------------------------------


def _fps_cold_start(
    X: np.ndarray, sq: np.ndarray, init: Any, random_state: int
) -> tuple[int, str]:
    """First pick when there is no seed.

    Default is ``farthest_from_mean`` because it is invariant to row order.
    --shuffle, batching and the worker pool all reorder rows; a start rule that
    depends on row order would mean the answer changes when you change
    --n_procs, which is indefensible.  ``np.argmax`` breaks ties at the lowest
    index, so the rule is fully deterministic.
    """
    kind, value = _parse_init(init)
    n = X.shape[0]
    if kind == "index":
        idx = int(value) % n
        return idx, f"index:{idx}"
    if kind == "random":
        rng = np.random.default_rng(random_state)
        return int(rng.integers(n)), "random"
    centre = X.mean(axis=0)
    d2 = sq - 2.0 * (X @ centre) + float(centre @ centre)
    return int(np.argmax(d2)), "farthest_from_mean"


def _select_fps(
    X_cand: np.ndarray,
    n_select: int,
    X_seed: np.ndarray | None = None,
    *,
    init: Any = "farthest_from_mean",
    seed_chunk: int = 256,
    random_state: int = 0,
    min_dist2: float | None = None,
    allow_zero: bool = False,
    **_ignored: Any,
) -> tuple[np.ndarray, np.ndarray, str | None, dict]:
    """Greedy farthest-point sampling seeded by an existing set.

    Maintains ``dmin[i] = min over (seed u picked) ||x_i - s||^2`` and repeatedly
    takes its argmax.  Each pick costs one gemv, O(n*d); memory is O(n) beyond the
    inputs.  No n x n matrix is ever formed, which is what makes this usable at
    batch sizes where a pairwise matrix would not fit.
    """
    n = X_cand.shape[0]
    k = min(int(n_select), n)
    if k <= 0:
        return (
            np.empty(0, np.int64),
            np.empty(0, np.float64),
            "exhausted" if n_select > 0 else None,
            {},
        )

    sq_c = np.einsum("ij,ij->i", X_cand, X_cand).astype(np.float64)
    dmin = np.full(n, np.inf, dtype=np.float64)

    if X_seed is not None and len(X_seed):
        min_sqdist_to_set(X_cand, X_seed, chunk=seed_chunk, out=dmin)
        cur = int(np.argmax(dmin))
        start_rule = "seeded"
    else:
        cur, start_rule = _fps_cold_start(X_cand, sq_c, init, random_state)

    picks = np.empty(k, np.int64)
    scores = np.empty(k, np.float64)
    reason: str | None = None
    taken = 0

    for t in range(k):
        picks[t] = cur
        scores[t] = dmin[cur]
        taken = t + 1

        # One gemv against the point just picked.
        d2 = sq_c + sq_c[cur] - 2.0 * (X_cand @ X_cand[cur])
        np.minimum(dmin, d2, out=dmin)
        dmin[cur] = -np.inf  # can never be picked again

        if taken >= k:
            break

        cur = int(np.argmax(dmin))
        best = dmin[cur]
        if best == -np.inf:
            reason = "exhausted"
            break
        if min_dist2 is not None and best < min_dist2:
            reason = "min_dist_stop"
            break
        if not allow_zero and best <= 0.0:
            reason = "exhausted"
            break

    picks, scores = picks[:taken], scores[:taken]
    if reason is None and taken < int(n_select):
        reason = "exhausted"

    aux = {"start_index": int(picks[0]), "start_rule": start_rule}
    return picks, scores, reason, aux


def fps_order_scores(
    X_selected: np.ndarray, X_seed: np.ndarray | None = None
) -> np.ndarray:
    """Coverage-radius profile of an already-chosen set.

    Re-runs the greedy ordering over ``X_selected`` so a k-means selection gets
    the same monotone diagnostic curve FPS produces natively.  The first entry is
    ``inf`` when there is no seed (nothing to be distant from yet).
    """
    X_selected = _as2d(X_selected, "X_selected")
    if X_selected is None:
        return np.empty(0, np.float64)
    _, scores, _, _ = _select_fps(
        X_selected, len(X_selected), _as2d(X_seed, "X_seed"), init="farthest_from_mean"
    )
    return scores


# --------------------------------------------------------------------------
# k-means
# --------------------------------------------------------------------------


def _select_kmeans(
    X_cand: np.ndarray,
    n_select: int,
    X_seed: np.ndarray | None = None,
    *,
    k: int | None = None,
    per_cluster: int = 1,
    second_pick: str = "farthest",
    fill: str = "fps",
    trim: str = "population",
    minibatch: bool | None = None,
    minibatch_threshold: int = 20_000,
    n_init: int = 10,
    random_state: int = 0,
    min_dist2: float | None = None,
    **_ignored: Any,
) -> tuple[np.ndarray, np.ndarray, str | None, dict]:
    """k-means on the union of candidates and seed, honouring an exact n_select.

    Clustering the *union* is what gives "this cluster is already covered" a
    meaning: a cluster containing any seed point yields nothing, which is exactly
    the covered-cluster skip of the original ``select_per_cluster`` strategy 1.
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans

    n = X_cand.shape[0]
    m = 0 if X_seed is None else len(X_seed)
    k = int(np.clip(int(k) if k is not None else int(n_select), 1, max(n, 1)))

    Z = X_cand if m == 0 else np.vstack([X_cand, X_seed])
    use_mb = (len(Z) > minibatch_threshold) if minibatch is None else bool(minibatch)

    if use_mb:
        est = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=max(3, min(n_init, 3)),
            # batch_size must exceed k or sklearn cannot fill every centroid
            batch_size=max(2048, 4 * k),
        )
    else:
        # n_init pinned explicitly: "auto" resolves differently across sklearn
        # versions, which would silently change results on an env upgrade.
        est = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)

    labels = est.fit_predict(Z)
    lab_c = labels[:n]
    lab_s = labels[n:] if m else np.empty(0, dtype=labels.dtype)
    centers = est.cluster_centers_

    covered = np.zeros(k, dtype=bool)
    if m:
        covered[np.unique(lab_s)] = True

    # Group candidates by cluster once: O(n log n), no per-point Python lookups.
    order = np.argsort(lab_c, kind="stable")
    bounds = np.searchsorted(lab_c[order], np.arange(k + 1))

    reps: list[int] = []
    pops: list[int] = []
    dists: list[float] = []
    clusters: list[int] = []
    taken: set[int] = set()

    for j in range(k):
        if covered[j]:
            continue
        members = order[bounds[j] : bounds[j + 1]]
        if members.size == 0:
            continue  # MiniBatchKMeans can leave a centroid unclaimed
        diff = X_cand[members] - centers[j]
        d2 = np.einsum("ij,ij->i", diff, diff)
        # distance ascending, then index ascending -> deterministic
        rank = np.lexsort((members, d2))
        wanted: list[int] = [0]
        if per_cluster > 1:
            if second_pick == "farthest":
                wanted.append(len(rank) - 1)
                wanted.extend(range(1, per_cluster - 1))
            else:
                wanted.extend(range(1, per_cluster))
        seen_slots: set[int] = set()
        for slot in wanted[:per_cluster]:
            if slot in seen_slots or slot >= rank.size:
                continue
            seen_slots.add(slot)
            p = int(members[rank[slot]])
            if p in taken:
                continue
            taken.add(p)
            reps.append(p)
            pops.append(int(members.size))
            dists.append(float(d2[rank[slot]]))
            clusters.append(int(j))

    reps_arr = np.asarray(reps, dtype=np.int64)
    scores_arr = np.asarray(dists, dtype=np.float64)
    n_from_kmeans = int(reps_arr.size)
    reason: str | None = None
    n_topup = 0

    if reps_arr.size > n_select:
        # Over-fill only happens when the user deliberately set kmeans_k >
        # n_select or per_cluster > 1.
        if trim == "fps":
            sub = _as2d(X_cand[reps_arr], "reps")
            keep_local, _, _, _ = _select_fps(
                sub, n_select, X_seed, random_state=random_state
            )
            keep = np.sort(keep_local)
        else:
            pop_arr = np.asarray(pops)
            keep = np.lexsort((reps_arr, scores_arr, -pop_arr))[:n_select]
            keep = np.sort(keep)
        reps_arr = reps_arr[keep]
        scores_arr = scores_arr[keep]
        clusters = [clusters[i] for i in keep]

    elif reps_arr.size < n_select and fill == "fps":
        # Under-fill: the shortfall exists *because* the seed already covers
        # regions, so the remaining budget should go to the least-covered
        # places -- which is the definition of seeded FPS.  Same primitive.
        mask = np.ones(n, dtype=bool)
        mask[reps_arr] = False
        rest = np.flatnonzero(mask)
        if rest.size:
            seed_for_fill = vstack_nonempty(X_seed, X_cand[reps_arr])
            extra, extra_scores, extra_reason, _ = _select_fps(
                X_cand[rest],
                int(n_select) - int(reps_arr.size),
                seed_for_fill,
                random_state=random_state,
                min_dist2=min_dist2,
            )
            n_topup = int(extra.size)
            reps_arr = np.concatenate([reps_arr, rest[extra]])
            scores_arr = np.concatenate([scores_arr, extra_scores])
            # -1 marks a pick that came from the FPS top-up, not from a cluster
            clusters.extend([-1] * n_topup)
            if extra_reason is not None:
                reason = extra_reason
        if reps_arr.size < n_select and reason is None:
            reason = "exhausted"

    elif reps_arr.size < n_select:
        reason = "strict_no_fill"

    aux = {
        "k": int(k),
        "n_covered_clusters": int(covered.sum()),
        "n_from_kmeans": n_from_kmeans,
        "n_from_fps_topup": n_topup,
        "minibatch": bool(use_mb),
        "n_init": int(n_init),
        "per_cluster": int(per_cluster),
        "trim": trim,
        "fill": fill,
        "cluster_of": clusters,
    }
    return reps_arr, scores_arr, reason, aux


# --------------------------------------------------------------------------
# public dispatcher
# --------------------------------------------------------------------------

_BACKENDS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray, str | None, dict]]] = {
    "fps": _select_fps,
    "kmeans": _select_kmeans,
}


def select_diverse(
    X_cand: np.ndarray,
    n_select: int,
    *,
    X_seed: np.ndarray | None = None,
    method: str = "kmeans",
    random_state: int = 0,
    min_dist2: float | None = None,
    strict_determinism: bool = False,
    **backend_kwargs: Any,
) -> SelectionResult:
    """Choose ``n_select`` rows of ``X_cand`` that are diverse w.r.t. each other
    and w.r.t. ``X_seed``.

    Contract
    --------
    1. Returned indices index ``X_cand`` only.  Seed rows are never returned.
    2. No duplicates.  ``len(indices) == n_select`` unless ``shortfall_reason``
       is set.
    3. ``X_seed=None`` and ``X_seed=np.empty((0, d))`` behave identically.
    4. Pure: the same inputs give the same output.
    5. ``X_cand`` and ``X_seed`` must live in the same normalized space and have
       the same ``d``.  The *caller* enforces descriptor-id equality; this
       function only checks shapes.
    6. Distances are squared Euclidean.  ``min_dist2`` is in squared units.

    Parameters
    ----------
    X_cand : (n, d) array, already normalized.
    X_seed : (m, d) array or None -- descriptors already held.
    method : ``"kmeans"`` or ``"fps"``.
    """
    if method not in _BACKENDS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(_BACKENDS)}")

    X_cand = _as2d(X_cand, "X_cand")
    if X_cand is None:
        return SelectionResult(
            indices=np.empty(0, np.int64),
            scores=np.empty(0, np.float64),
            method=method,
            n_requested=int(n_select),
            shortfall_reason="exhausted" if n_select > 0 else None,
        )
    X_seed = _as2d(X_seed, "X_seed")
    if X_seed is not None and X_seed.shape[1] != X_cand.shape[1]:
        raise ValueError(
            f"seed descriptors have dimension {X_seed.shape[1]} but candidates have "
            f"{X_cand.shape[1]}. They are not comparable -- this normally means the "
            "two were computed with different descriptor parameters."
        )

    # float64 accumulation with float32 storage keeps the gemv cheap while
    # keeping the running minimum well conditioned.
    if X_cand.dtype not in (np.float32, np.float64):
        X_cand = X_cand.astype(np.float32)
    if X_seed is not None and X_seed.dtype != X_cand.dtype:
        X_seed = X_seed.astype(X_cand.dtype)

    with _maybe_single_threaded(strict_determinism):
        indices, scores, reason, aux = _BACKENDS[method](
            X_cand,
            int(n_select),
            X_seed,
            random_state=int(random_state),
            min_dist2=min_dist2,
            **backend_kwargs,
        )

    if indices.size != np.unique(indices).size:  # pragma: no cover - invariant
        raise AssertionError("selection produced duplicate indices")

    aux = dict(aux)
    aux["n_seed"] = 0 if X_seed is None else int(len(X_seed))
    aux["n_candidates"] = int(X_cand.shape[0])

    return SelectionResult(
        indices=indices,
        scores=scores,
        method=method,
        n_requested=int(n_select),
        shortfall_reason=reason,
        aux=aux,
    )
