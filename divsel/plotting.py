"""Figures.

matplotlib is imported lazily so it stays an optional dependency.

Two kinds of figure are produced, and the distinction matters:

* **diagnostics.png** -- exact quantities that need no embedding and therefore
  cannot mislead: the pairwise-distance histogram (with std/mean printed), the
  PCA scree, the coverage-radius curve, and the nearest-selected-distance CDF
  against a random-selection baseline.  For "is my selection actually diverse?"
  these answer the question better than any scatter plot.
* **embedding_<method>.png** -- a 2-D projection, stamped with its own Shepard
  correlation and trustworthiness so the reader can see how much it distorted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .embed import (
    distance_stats,
    embed,
    faithfulness,
    nearest_selected_distance,
    subsample_indices,
)
from .log import get_logger
from .selection import fps_order_scores

logger = get_logger(__name__)

__all__ = ["make_figures"]

_GREY = "#bdbdbd"
_SELECTED = "#3a53a4"
_SEED = "#d62728"
_ACCENT = "#0f8140"


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotting needs matplotlib. Install it with:  pip install 'divsel[plot]'"
        ) from exc
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    return plt


def make_figures(
    out_dir: str | Path,
    *,
    method: str = "pca",
    max_points: int = 3000,
    diagnostics: bool = True,
    seed: int = 0,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> list[Path]:
    """Render the figures for a finished run directory."""
    import pandas as pd

    from .store import load_store

    plt = _require_matplotlib()
    out_dir = Path(out_dir)
    X, frames, meta = load_store(out_dir)
    X = np.asarray(X)
    n_rows = int(meta["n_rows"])
    X = X[:n_rows]

    selected_rows: np.ndarray = np.empty(0, dtype=int)
    sel_csv = out_dir / "selected.csv"
    if sel_csv.exists():
        selected_rows = pd.read_csv(sel_csv)["row"].to_numpy(dtype=int)

    written: list[Path] = []
    rng = np.random.default_rng(seed)
    stats = distance_stats(X, seed=seed)

    if diagnostics:
        written += _diagnostics_figure(
            plt, out_dir, X, selected_rows, stats, rng, seed, formats, dpi
        )

    written += _embedding_figure(
        plt, out_dir, X, selected_rows, method, max_points, stats, seed, formats, dpi
    )
    return written


def _save(fig, out_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> list[Path]:
    paths = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths


def _diagnostics_figure(
    plt, out_dir, X, selected_rows, stats, rng, seed, formats, dpi
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))

    # (a) pairwise distance histogram -- the concentration check
    ax = axes[0, 0]
    d = stats["distances"]
    ax.hist(d, bins=60, color=_ACCENT, alpha=0.8, edgecolor="none")
    ax.set_xlabel("pairwise distance")
    ax.set_ylabel("count")
    verdict = "CONCENTRATED" if stats["concentrated"] else "well spread"
    ax.set_title(
        f"(a) distance distribution\nstd/mean = {stats['rel_spread']:.3f}  ({verdict})"
    )
    if stats["concentrated"]:
        ax.text(
            0.02,
            0.95,
            "every 2-D plot of this\nwill be a blob",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#b2182b",
        )

    # (b) PCA scree -- how much of the story a 2-D scatter can show
    ax = axes[0, 1]
    try:
        from sklearn.decomposition import PCA

        k = int(min(20, X.shape[1], max(1, len(X) - 1)))
        ratio = PCA(n_components=k, random_state=seed).fit(X).explained_variance_ratio_
        ax.bar(np.arange(1, k + 1), 100 * ratio, color=_ACCENT, alpha=0.8)
        ax.plot(
            np.arange(1, k + 1), 100 * np.cumsum(ratio), "o-", color=_SELECTED, ms=3, lw=1
        )
        ax.set_xlabel("component")
        ax.set_ylabel("% variance (bar) / cumulative (line)")
        ax.set_title(
            f"(b) PCA scree -- PC1+PC2 carry {100 * float(ratio[:2].sum()):.1f}%"
        )
    except Exception as exc:  # pragma: no cover
        ax.text(0.5, 0.5, f"PCA unavailable\n{exc}", ha="center", va="center")

    # (c) coverage-radius curve -- the stop-the-campaign signal
    ax = axes[1, 0]
    if selected_rows.size:
        profile = fps_order_scores(X[selected_rows])
        finite = np.isfinite(profile)
        ax.plot(
            np.arange(1, profile.size + 1)[finite],
            np.sqrt(profile[finite]),
            "-",
            color=_SELECTED,
            lw=1.4,
        )
        ax.set_yscale("log")
        ax.set_xlabel("pick number")
        ax.set_ylabel("distance to nearest already-chosen")
        ax.set_title("(c) coverage radius\nplateau near 0 = space saturated")
    else:
        ax.text(0.5, 0.5, "no selection in this run", ha="center", va="center")
        ax.set_title("(c) coverage radius")

    # (d) nearest-selected-distance CDF vs a random baseline
    ax = axes[1, 1]
    if selected_rows.size:
        nsd = nearest_selected_distance(X, X[selected_rows])
        rand_rows = rng.choice(len(X), size=min(len(selected_rows), len(X)), replace=False)
        nsd_rand = nearest_selected_distance(X, X[rand_rows])
        for values, label, colour in (
            (nsd, "divsel selection", _SELECTED),
            (nsd_rand, "random, same size", _GREY),
        ):
            v = np.sort(values[np.isfinite(values)])
            ax.plot(v, np.arange(1, v.size + 1) / v.size, label=label, color=colour, lw=1.4)
        ax.set_xlabel("distance to nearest selected frame")
        ax.set_ylabel("fraction of dataset")
        ax.set_title("(d) coverage of the dataset\nfurther left/steeper is better")
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.text(0.5, 0.5, "no selection in this run", ha="center", va="center")
        ax.set_title("(d) coverage of the dataset")

    fig.suptitle(
        f"divsel diagnostics -- {len(X)} frames, "
        f"{len(selected_rows)} selected  (exact quantities, no embedding)",
        fontsize=10,
    )
    fig.tight_layout()
    paths = _save(fig, out_dir, "diagnostics", formats, dpi)
    plt.close(fig)
    logger.info("wrote %s", ", ".join(str(p) for p in paths))
    return paths


def _embedding_figure(
    plt, out_dir, X, selected_rows, method, max_points, stats, seed, formats, dpi
) -> list[Path]:
    rng = np.random.default_rng(seed)
    sel_set = set(int(r) for r in selected_rows)

    # Always keep every selected point; subsample the rest for the O(n^2)
    # methods.  The caption says how many are shown, so nothing is implied.
    others = np.array([i for i in range(len(X)) if i not in sel_set], dtype=int)
    budget = max(0, max_points - len(sel_set))
    keep_others = others[subsample_indices(len(others), budget, rng)] if budget else others[:0]
    idx = np.concatenate([keep_others, selected_rows.astype(int)])
    is_selected = np.zeros(len(idx), dtype=bool)
    is_selected[len(keep_others) :] = True

    Xs = X[idx]
    result = embed(Xs, method, seed=seed)
    Y = np.asarray(result["coords"])
    faith = faithfulness(Xs, Y, seed=seed)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(
        Y[~is_selected, 0],
        Y[~is_selected, 1],
        s=10,
        c=_GREY,
        alpha=0.55,
        edgecolors="none",
        label=f"not selected (n={int((~is_selected).sum())})",
    )
    if is_selected.any():
        ax.scatter(
            Y[is_selected, 0],
            Y[is_selected, 1],
            s=26,
            c=_SELECTED,
            edgecolors="k",
            linewidths=0.3,
            label=f"selected (n={int(is_selected.sum())})",
        )
    labels = result["axis_labels"]
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1] if len(labels) > 1 else "")
    # Equal aspect so the picture reads as a map rather than a decoration.
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, loc="best")

    faith_line = (
        f"Shepard rho = {faith['spearman']:.2f}, "
        f"trustworthiness@{faith['k']} = {faith['trustworthiness']:.2f}"
    )
    caption = f"{result['caption']}\n{faith_line}"
    if not result["distance_faithful"]:
        caption += "\nDO NOT read distances off this plot."
    caption += (
        f"\nshowing {len(idx)} of {len(X)} frames; "
        f"full-space std/mean = {stats['rel_spread']:.3f}"
    )
    ax.set_title(f"divsel embedding: {result['method'].upper()}", fontsize=10)
    fig.text(0.01, -0.02, caption, fontsize=7.5, va="top", wrap=True)

    fig.tight_layout()
    paths = _save(fig, out_dir, f"embedding_{result['method']}", formats, dpi)
    plt.close(fig)
    logger.info(
        "wrote %s (Shepard rho=%.2f, trustworthiness=%.2f)",
        ", ".join(str(p) for p in paths),
        faith["spearman"],
        faith["trustworthiness"],
    )
    return paths
