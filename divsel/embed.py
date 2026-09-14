"""Distance diagnostics and 2-D embeddings.

The original pipeline produced one t-SNE figure and nothing else, and that
figure was a featureless round blob.  There were two causes and only one is
fixable.

**t-SNE cannot show distance, by construction.**  It matches a *normalized*
neighbour distribution with a heavy-tailed kernel; the normalization equalizes
apparent cluster density and the repulsion term inflates gaps.  In a t-SNE map,
cluster sizes, between-cluster distances and empty space are all meaningless.
No tuning changes that.

**The metric it was fed was genuinely structureless, and that part is real.**
z-scoring 5,940 SOAP columns divides each by its own standard deviation, which
*upweights* the near-zero-variance channels -- the rare-species cross terms --
so distance ends up dominated by the least informative columns.  In high
dimension all pairwise distances then concentrate toward one value, and a
near-uniform disc is exactly what any 2-D method will draw.  **A perfectly
round, evenly filled blob is the visual signature of concentrated distances.**

So: measure first (:func:`distance_stats`), then embed with something that
preserves distance (:func:`embed`, default PCA, ``mds`` for the most faithful
picture), then report the distortion as a number (:func:`faithfulness`) rather
than letting the reader assume there is none.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .log import get_logger
from .selection import min_sqdist_to_set

logger = get_logger(__name__)

__all__ = [
    "distance_stats",
    "embed",
    "faithfulness",
    "nearest_selected_distance",
    "subsample_indices",
]

CONCENTRATION_WARNING = 0.1


def subsample_indices(n: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _pairwise_d2(X: np.ndarray) -> np.ndarray:
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return d2


def distance_stats(
    X: np.ndarray, *, max_points: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Is there any structure to draw?

    ``rel_spread = std(d) / mean(d)`` over a random subsample.  Below ~0.1 the
    data is concentrated: every 2-D method will give a blob, and the honest
    conclusion is that these structures are near-identical *under this
    descriptor*.  The fix is then chemistry -- a larger ``r_cut``, a
    species-resolved descriptor, a different ``sigma`` -- not plotting.
    """
    rng = np.random.default_rng(seed)
    idx = subsample_indices(len(X), max_points, rng)
    Xs = np.asarray(X[idx], dtype=np.float64)
    d2 = _pairwise_d2(Xs)
    iu = np.triu_indices(len(Xs), k=1)
    d = np.sqrt(d2[iu])

    mean = float(d.mean()) if d.size else float("nan")
    std = float(d.std()) if d.size else float("nan")
    rel = std / mean if mean > 0 else float("nan")

    stats = {
        "n_points_sampled": int(len(Xs)),
        "n_pairs": int(d.size),
        "mean_distance": mean,
        "std_distance": std,
        "rel_spread": rel,
        "min_distance": float(d.min()) if d.size else float("nan"),
        "max_distance": float(d.max()) if d.size else float("nan"),
        "concentrated": bool(np.isfinite(rel) and rel < CONCENTRATION_WARNING),
        "distances": d,
    }
    if stats["concentrated"]:
        logger.warning(
            "distances are concentrated: std/mean = %.3f < %.2f. Every 2-D "
            "embedding of this data will look like a featureless blob, because "
            "the structures really are near-identical under this descriptor. "
            "Widening r_cut or using a species-resolved descriptor will help; "
            "changing the plot will not.",
            rel,
            CONCENTRATION_WARNING,
        )
    else:
        logger.debug("distance spread std/mean = %.3f", rel)
    return stats


def _pca(X: np.ndarray, n_components: int = 2):
    from sklearn.decomposition import PCA

    k = min(n_components, X.shape[1], max(1, X.shape[0] - 1))
    model = PCA(n_components=k, random_state=0)
    Y = model.fit_transform(np.asarray(X, dtype=np.float64))
    return Y, model.explained_variance_ratio_


def _classical_mds(X: np.ndarray, n_components: int = 2):
    """Principal coordinates analysis: eigendecomposition of the centred D^2.

    Deterministic, and it optimizes agreement with the *actual* distances,
    which is the thing t-SNE cannot do.
    """
    d2 = _pairwise_d2(np.asarray(X, dtype=np.float64))
    n = len(d2)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ d2 @ J
    B = (B + B.T) / 2.0
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:n_components]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.clip(vals, 0.0, None)
    Y = vecs * np.sqrt(vals)
    total = float(np.trace(B))
    ratio = vals / total if total > 0 else np.zeros_like(vals)
    return Y, ratio


def embed(
    X: np.ndarray,
    method: str = "pca",
    *,
    n_components: int = 2,
    seed: int = 0,
    perplexity: float | None = None,
    metric_mds: bool = True,
) -> dict[str, Any]:
    """Project to 2-D.  Returns coords plus what the axes actually mean.

    ``pca``   true linear projection -- distances are real, only truncated.
    ``mds``   classical MDS, refined by metric SMACOF when available; minimizes
              stress against the real distance matrix.  The most
              distance-faithful option, and the right default for "how far
              apart are these really?".
    ``tsne``  local neighbourhoods only.  Never read distances off it.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)

    if method == "pca":
        Y, ratio = _pca(X, n_components)
        return {
            "coords": Y,
            "method": "pca",
            "axis_labels": [
                f"PC{i + 1} ({100 * r:.1f}% var)" for i, r in enumerate(ratio)
            ],
            "explained_variance_ratio": ratio.tolist(),
            "distance_faithful": True,
            "caption": (
                f"PCA: a true linear projection, so plotted distances are real "
                f"(lower bounds on the full-space distance). The two axes carry "
                f"{100 * float(ratio[:2].sum()):.1f}% of the total variance."
            ),
        }

    if method == "mds":
        Y, ratio = _classical_mds(X, n_components)
        if metric_mds and n >= 4:
            try:
                from sklearn.manifold import MDS

                d = np.sqrt(_pairwise_d2(X))
                kwargs = dict(
                    n_components=n_components,
                    dissimilarity="precomputed",
                    random_state=seed,
                    n_init=1,
                )
                try:
                    model = MDS(normalized_stress=False, **kwargs)
                except TypeError:  # sklearn < 1.2 has no normalized_stress
                    model = MDS(**kwargs)
                # Warm-start SMACOF from the classical-MDS solution; `init` is a
                # fit_transform argument, not a constructor one.
                Y = model.fit_transform(d, init=Y)
            except Exception as exc:  # pragma: no cover - fall back to PCoA
                logger.debug("metric MDS unavailable (%s); using classical MDS", exc)
        return {
            "coords": Y,
            "method": "mds",
            "axis_labels": ["MDS 1", "MDS 2"],
            "explained_variance_ratio": ratio.tolist(),
            "distance_faithful": True,
            "caption": (
                "MDS: the 2-D layout was fitted to reproduce the real "
                "feature-space distances, so distances here are meaningful "
                "(check the Shepard correlation in the caption)."
            ),
        }

    if method == "tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        # PCA pre-step of 50, not 200: beyond ~50 components t-SNE gains
        # nothing and the neighbour search only gets slower.
        pre = min(50, X.shape[1], max(1, n - 1))
        Xp = PCA(n_components=pre, random_state=seed).fit_transform(X)
        # Scale perplexity with N instead of a fixed 30.
        perp = perplexity if perplexity is not None else float(np.clip(n / 100, 5, 50))
        perp = float(min(perp, max(2, (n - 1) / 3)))
        kwargs = dict(
            n_components=n_components,
            perplexity=perp,
            init="pca",
            random_state=seed,
            metric="euclidean",
        )
        try:
            model = TSNE(max_iter=1000, **kwargs)
        except TypeError:  # sklearn < 1.5 spells it n_iter
            model = TSNE(n_iter=1000, **kwargs)
        Y = model.fit_transform(Xp)
        return {
            "coords": Y,
            "method": "tsne",
            "axis_labels": ["t-SNE 1", "t-SNE 2"],
            "explained_variance_ratio": None,
            "distance_faithful": False,
            "caption": (
                f"t-SNE (perplexity {perp:.0f}): neighbourhoods only. Cluster "
                "sizes, the distances between clusters and the empty space are "
                "NOT interpretable."
            ),
        }

    raise ValueError(f"unknown embedding {method!r}; expected pca, mds or tsne")


def faithfulness(
    X: np.ndarray, Y: np.ndarray, *, max_points: int = 1000, seed: int = 0, k: int = 12
) -> dict[str, Any]:
    """How much did the embedding distort the distances?

    Returns Shepard-diagram data, the Spearman correlation between true and
    embedded distances, and trustworthiness.  These numbers belong in the
    figure caption: a t-SNE plot with Spearman 0.2 is then self-evidently not a
    distance map, which is the information the original figure was missing.
    """
    rng = np.random.default_rng(seed)
    idx = subsample_indices(len(X), max_points, rng)
    Xs = np.asarray(X[idx], dtype=np.float64)
    Ys = np.asarray(Y[idx], dtype=np.float64)

    iu = np.triu_indices(len(Xs), k=1)
    d_true = np.sqrt(_pairwise_d2(Xs)[iu])
    d_emb = np.sqrt(_pairwise_d2(Ys)[iu])

    out: dict[str, Any] = {
        "n_points": int(len(Xs)),
        "d_true": d_true,
        "d_embedded": d_emb,
    }
    try:
        from scipy.stats import spearmanr

        rho = float(spearmanr(d_true, d_emb).statistic)
    except Exception:  # pragma: no cover
        rho = float("nan")
    out["spearman"] = rho

    try:
        from sklearn.manifold import trustworthiness

        kk = int(min(k, max(1, (len(Xs) - 1) // 2)))
        out["trustworthiness"] = float(trustworthiness(Xs, Ys, n_neighbors=kk))
        out["k"] = kk
    except Exception as exc:  # pragma: no cover
        logger.debug("trustworthiness unavailable: %s", exc)
        out["trustworthiness"] = float("nan")
        out["k"] = k
    return out


def nearest_selected_distance(
    X_all: np.ndarray, X_selected: np.ndarray
) -> np.ndarray:
    """Distance from every frame to its nearest selected frame.

    The CDF of this, plotted against the same curve for an equal-sized *random*
    selection, is the most direct answer to "is my selection actually
    covering the space?" -- and it needs no embedding, so it cannot mislead.
    """
    if len(X_selected) == 0:
        return np.full(len(X_all), np.inf)
    return np.sqrt(min_sqdist_to_set(np.asarray(X_all), np.asarray(X_selected)))
