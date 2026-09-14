"""Command line interface.

Four subcommands::

    divsel describe   compute and persist descriptors only
    divsel select     describe (if needed), select, and write selected.traj
    divsel gather     rebuild a trajectory from a selection CSV
    divsel plot       diagnostics and a 2-D embedding for a finished run

Heavy imports happen inside the handlers, so ``divsel --help`` is fast even
though the package depends on ase, sklearn and dscribe.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from typing import Sequence

from . import __version__
from .log import setup_logging

_METHODS = ("kmeans", "fps")
_FEATURIZERS = ("soap", "uma", "mace")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--verbose",
        choices=("quiet", "normal", "high"),
        default="normal",
        help="quiet: warnings only. normal: stage lines + summary. "
        "high: per-batch detail, species union, cache paths, diagnostics.",
    )


def _add_input(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("input")
    g.add_argument(
        "--traj",
        nargs="+",
        required=True,
        metavar="GLOB",
        help="input structures: .traj, .xyz, .extxyz or anything ASE reads; globs allowed",
    )
    g.add_argument("--out", default="divsel_out", help="output directory (default: %(default)s)")


def _add_featurizer(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("features")
    g.add_argument(
        "--featurizer",
        choices=_FEATURIZERS,
        default="soap",
        help="descriptor backend (default: %(default)s). uma/mace are registered "
        "placeholders and are not implemented yet.",
    )
    s = p.add_argument_group("soap")
    s.add_argument(
        "--species",
        nargs="*",
        default=None,
        metavar="Z|SYM",
        help="atomic numbers or symbols; omit (or 'auto') to take the union "
        "observed during the scan. Note d grows as O(n_species^2).",
    )
    s.add_argument("--r_cut", type=float, default=6.0)
    s.add_argument("--n_max", type=int, default=9)
    s.add_argument("--l_max", type=int, default=3)
    s.add_argument("--sigma", type=float, default=1.0)
    s.add_argument("--rbf", default="gto", choices=("gto", "polynomial"))
    s.add_argument(
        "--soap_average",
        default="outer",
        choices=("outer", "off"),
        help="'outer' is exactly the mean of per-atom SOAP vectors (default)",
    )
    s.add_argument(
        "--on_unknown_species",
        default="error",
        choices=("error", "skip"),
        help="what to do with a structure containing an element outside --species",
    )


def _add_box(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("cell (for non-periodic input)")
    g.add_argument(
        "--cell_mode",
        default="auto",
        choices=("auto", "native", "box"),
        help="auto: add vacuum only along directions that lack a cell, so a slab "
        "keeps its in-plane periodicity. box: always rebuild. native: never.",
    )
    g.add_argument(
        "--vacuum",
        type=float,
        default=15.0,
        help="vacuum padding per side in A (default: %(default)s). Structures "
        "that actually get a box must satisfy 2*vacuum >= r_cut, and that is "
        "checked per structure. Fully periodic input is never boxed, so this "
        "setting does not apply to it and cannot block such a run.",
    )
    g.add_argument("--box_shape", default="orthorhombic", choices=("orthorhombic", "cubic"))
    g.add_argument(
        "--on_small_vacuum",
        default="error",
        choices=("error", "skip"),
        help="what to do with a structure that needs a box when 2*vacuum < "
        "r_cut: 'error' aborts (default, it is a config mistake), 'skip' "
        "records it in frames.csv and carries on with the rest.",
    )


def _add_sampling(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("sampling")
    g.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    g.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        default=True,
        help="shuffle the global frame list before batching (default: on). "
        "The single most effective mitigation of batch-order bias.",
    )
    g.add_argument("--no_shuffle", dest="shuffle", action="store_false")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="process at most N frames, applied after stride and before any "
        "species filtering, so the subset is a reproducible prefix",
    )
    g.add_argument(
        "--no_scan",
        dest="scan",
        action="store_false",
        default=True,
        help="skip the scan pass: no frame count, no random access, so no "
        "--shuffle and --species must be given explicitly",
    )


def _add_stream(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("streaming")
    g.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="frames per batch; 0 disables streaming (one batch, exact answer)",
    )
    g.add_argument("--pool_alpha", type=int, default=4, help="over-select this many x n_select")
    g.add_argument("--pool_cap", type=int, default=None)
    g.add_argument(
        "--pool_stage",
        default="fps",
        choices=("fps", "random"),
        help="how each batch contributes to the carry-forward pool. 'random' is "
        "density-unbiased and is the better choice for --method kmeans over "
        "many batches.",
    )
    g.add_argument("--n_procs", type=int, default=max(1, mp.cpu_count() or 1))


def _add_select(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("selection")
    g.add_argument("--n_select", type=int, required=True)
    g.add_argument(
        "--method",
        choices=_METHODS,
        default="kmeans",
        help="default: %(default)s. fps is reproducible across sklearn versions "
        "and thread counts; kmeans is not.",
    )
    g.add_argument(
        "--selected_images",
        nargs="*",
        default=[],
        metavar="FILE",
        help="trajectories of structures selected in previous rounds, used as a "
        "seed set: new picks must be diverse with respect to these, and these "
        "are never re-selected. Pass one file or several (typically one per "
        "round); all of them must have been described with the same "
        "parameters, which is checked against the stamp each frame carries",
    )
    g.add_argument("--normalize", default="l2", choices=("l2", "none"))
    g.add_argument("--strict_determinism", action="store_true")
    g.add_argument("--min_dist2", type=float, default=None)
    g.add_argument("--cache_root", default=None, help="seed descriptor cache (default .divsel_cache)")

    f = p.add_argument_group("fps")
    f.add_argument("--fps_init", default="farthest_from_mean")

    k = p.add_argument_group("kmeans")
    k.add_argument("--kmeans_k", default="auto")
    k.add_argument("--kmeans_per_cluster", type=int, default=1)
    k.add_argument("--kmeans_fill", default="fps", choices=("fps", "none"))
    k.add_argument("--kmeans_trim", default="population", choices=("population", "fps"))
    k.add_argument("--kmeans_n_init", type=int, default=10)
    k.add_argument("--minibatch_threshold", type=int, default=20_000)

    v = p.add_argument_group("figures")
    v.add_argument("--plot", action="store_true", help="also write diagnostics and an embedding")
    v.add_argument("--embed", default="pca", choices=("pca", "mds", "tsne"))
    v.add_argument("--embed_max_points", type=int, default=3000)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="divsel",
        description="Diversity selection of atomic structures "
        "(SOAP descriptors, k-means / farthest-point sampling, streaming, seed sets).",
    )
    p.add_argument("--version", action="version", version=f"divsel {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("describe", help="compute and persist descriptors only")
    _add_input(d), _add_featurizer(d), _add_box(d), _add_sampling(d), _add_stream(d)
    d.add_argument("--normalize", default="l2", choices=("l2", "none"))
    _add_common(d)

    s = sub.add_parser("select", help="describe, select, and write selected.traj")
    _add_input(s), _add_featurizer(s), _add_box(s), _add_sampling(s), _add_stream(s)
    _add_select(s), _add_common(s)

    g = sub.add_parser("gather", help="rebuild a trajectory from a selection CSV")
    g.add_argument("--csv", required=True)
    g.add_argument("--out_traj", required=True)
    g.add_argument(
        "--traj",
        default=None,
        help="source trajectory, needed only for legacy CSVs that carry "
        "original_index without a source_file column",
    )
    _add_common(g)

    v = sub.add_parser("plot", help="diagnostics and a 2-D embedding for a finished run")
    v.add_argument("--out", required=True, help="a directory written by divsel select/describe")
    v.add_argument("--embed", default="pca", choices=("pca", "mds", "tsne"))
    v.add_argument("--embed_max_points", type=int, default=3000)
    v.add_argument("--no_diagnostics", dest="diagnostics", action="store_false", default=True)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    _add_common(v)

    return p


def _configs(args):
    from .config import (
        BoxConfig,
        SamplingConfig,
        SoapConfig,
        StreamConfig,
        parse_species,
    )

    soap = SoapConfig(
        species=parse_species(args.species),
        r_cut=args.r_cut,
        n_max=args.n_max,
        l_max=args.l_max,
        sigma=args.sigma,
        rbf=args.rbf,
        average=args.soap_average,
        on_unknown_species=args.on_unknown_species,
    )
    box = BoxConfig(mode=args.cell_mode, vacuum=args.vacuum, shape=args.box_shape)
    sampling = SamplingConfig(
        stride=args.stride,
        shuffle=args.shuffle,
        seed=args.seed,
        max_frames=args.max_frames,
        scan=args.scan,
    )
    stream = StreamConfig(
        batch_size=args.batch_size,
        pool_alpha=args.pool_alpha,
        pool_cap=args.pool_cap,
        pool_stage=args.pool_stage,
        n_procs=args.n_procs,
    )
    return soap, box, sampling, stream


def _cmd_describe(args, argv) -> int:
    from .streaming import run_describe

    soap, box, sampling, stream = _configs(args)
    run_describe(
        args.traj,
        args.out,
        featurizer=args.featurizer,
        soap_cfg=soap,
        box_cfg=box,
        sampling=sampling,
        stream=stream,
        normalize=args.normalize,
        on_small_vacuum=args.on_small_vacuum,
        argv=argv,
    )
    return 0


def _cmd_select(args, argv) -> int:
    from .config import SelectConfig
    from .streaming import run_selection

    soap, box, sampling, stream = _configs(args)
    kmeans_k = args.kmeans_k if args.kmeans_k == "auto" else int(args.kmeans_k)
    select = SelectConfig(
        n_select=args.n_select,
        method=args.method,
        normalize=args.normalize,
        random_state=args.seed,
        min_dist2=args.min_dist2,
        strict_determinism=args.strict_determinism,
        fps_init=args.fps_init,
        kmeans_k=kmeans_k,
        kmeans_per_cluster=args.kmeans_per_cluster,
        kmeans_fill=args.kmeans_fill,
        kmeans_trim=args.kmeans_trim,
        kmeans_n_init=args.kmeans_n_init,
        minibatch_threshold=args.minibatch_threshold,
    )
    run_selection(
        args.traj,
        args.out,
        select,
        featurizer=args.featurizer,
        soap_cfg=soap,
        box_cfg=box,
        sampling=sampling,
        stream=stream,
        selected_images=args.selected_images,
        cache_root=args.cache_root,
        on_small_vacuum=args.on_small_vacuum,
        argv=argv,
    )
    if args.plot:
        from .plotting import make_figures

        make_figures(
            args.out,
            method=args.embed,
            max_points=args.embed_max_points,
            seed=args.seed,
        )
    return 0


def _cmd_gather(args, argv) -> int:
    from .gather import gather_from_csv

    gather_from_csv(args.csv, args.out_traj, fallback_source=args.traj)
    return 0


def _cmd_plot(args, argv) -> int:
    from .plotting import make_figures

    make_figures(
        args.out,
        method=args.embed,
        max_points=args.embed_max_points,
        diagnostics=args.diagnostics,
        seed=args.seed,
        formats=args.formats,
    )
    return 0


_HANDLERS = {
    "describe": _cmd_describe,
    "select": _cmd_select,
    "gather": _cmd_gather,
    "plot": _cmd_plot,
}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    logger = setup_logging(args.verbose)

    from .errors import DivselError

    try:
        return _HANDLERS[args.command](args, ["divsel", *argv])
    except DivselError as exc:
        logger.error("%s", exc)
        return 2
    except NotImplementedError as exc:
        logger.error("%s", exc)
        return 3
    except KeyboardInterrupt:  # pragma: no cover
        logger.error("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
