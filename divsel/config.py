"""Configuration dataclasses and the descriptor identity hash.

The single most important thing in this module is :func:`descriptor_id`.  Two
sets of feature vectors may only be compared -- which is what the whole seed-set
mechanism does -- when every parameter defining the feature space matches.  The
id is a short hash over exactly those parameters, and it is stamped onto every
artifact divsel writes: the descriptor store, the cached seed arrays, the run
manifest, and every frame of ``selected.traj``.

This replaces a filename-keyed cache in the original script, which returned a
cache hit after ``--soap-rcut`` changed from 6.0 to 5.0 -- a silent wrong answer,
because r_cut does not change the descriptor length and so the shape check could
not catch it.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from ase.data import atomic_numbers, chemical_symbols

from .errors import BoxError, DescriptorMismatchError

# --------------------------------------------------------------------------
# species helpers
# --------------------------------------------------------------------------


def parse_species(tokens: Iterable[str | int] | None) -> tuple[int, ...] | None:
    """Parse a species list given as atomic numbers or chemical symbols.

    ``None`` or the literal string ``"auto"`` means "derive from the scan pass".
    Returns a sorted tuple of atomic numbers so that two spellings of the same
    set (``["O", "H"]`` and ``[8, 1]``) produce the same descriptor id.
    """
    if tokens is None:
        return None
    tokens = list(tokens)
    if len(tokens) == 1 and str(tokens[0]).lower() == "auto":
        return None
    if not tokens:
        return None

    out: set[int] = set()
    for tok in tokens:
        if isinstance(tok, (int,)) or (isinstance(tok, str) and tok.lstrip("-").isdigit()):
            z = int(tok)
        else:
            try:
                z = atomic_numbers[str(tok)]
            except KeyError:
                raise ValueError(
                    f"unknown chemical symbol {tok!r}; give atomic numbers or ASE symbols"
                ) from None
        if not 1 <= z < len(chemical_symbols):
            raise ValueError(f"atomic number out of range: {z}")
        out.add(z)
    return tuple(sorted(out))


def species_symbols(species: Sequence[int]) -> list[str]:
    return [chemical_symbols[int(z)] for z in species]


def soap_feature_dim(n_species: int, n_max: int, l_max: int) -> int:
    """Length of a dscribe SOAP power spectrum with ``average`` in {off, outer}.

    ``d = (n_sp*n_max) * (n_sp*n_max + 1) / 2 * (l_max + 1)``.  Verified against
    dscribe 2.1.1: species [1,6,7,8,13,35], n_max=9, l_max=3 -> 5940.

    Note the quadratic growth in the number of species: going from 6 to 8 species
    takes d from 5,940 to 10,404, and selection cost is linear in d.  Do not
    over-declare species "just in case".
    """
    nn = int(n_species) * int(n_max)
    return nn * (nn + 1) // 2 * (int(l_max) + 1)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SoapConfig:
    """Parameters of the dscribe SOAP descriptor.

    Defaults reproduce the original pipeline.  ``sigma`` and ``rbf`` are dscribe
    defaults made explicit here so they are recorded rather than assumed; the
    actual values are re-read off the constructed SOAP object before hashing.

    ``average="outer"`` is *exactly* equivalent to the original
    ``soap.create(atoms).mean(axis=0)`` -- verified to maxdiff 0.0 against
    dscribe 2.1.1.  ``average="inner"`` is a different quantity entirely
    (maxdiff 28.2 on the same structure) and must not be substituted.
    """

    species: tuple[int, ...] | None = None  # None => auto-detect during the scan
    r_cut: float = 6.0
    n_max: int = 9
    l_max: int = 3
    sigma: float = 1.0
    rbf: str = "gto"
    periodic: bool = True
    average: str = "outer"
    on_unknown_species: str = "error"  # error | skip

    def __post_init__(self) -> None:
        if self.average not in ("outer", "off"):
            raise ValueError(
                f"soap average must be 'outer' or 'off', got {self.average!r}. "
                "'inner' is NOT equivalent to averaging per-atom SOAP vectors."
            )
        if self.on_unknown_species not in ("error", "skip"):
            raise ValueError("on_unknown_species must be 'error' or 'skip'")
        if self.r_cut <= 0 or self.n_max < 1 or self.l_max < 0:
            raise ValueError("r_cut must be > 0, n_max >= 1, l_max >= 0")

    def with_species(self, species: Iterable[int]) -> "SoapConfig":
        return replace(self, species=tuple(sorted({int(z) for z in species})))

    @property
    def feature_dim(self) -> int:
        if self.species is None:
            raise ValueError(
                "feature_dim is undefined until the species set is resolved; "
                "call with_species() after the scan pass"
            )
        return soap_feature_dim(len(self.species), self.n_max, self.l_max)


@dataclass(frozen=True)
class BoxConfig:
    """How to give a non-periodic structure a cell.

    ``2 * vacuum >= r_cut`` is the condition under which periodic SOAP on the
    boxed structure equals non-periodic SOAP: ``ase.Atoms.center(vacuum=v)`` sets
    each cell length to ``extent + 2v``, so the minimum image separation is
    exactly ``2v``.  Verified against dscribe 2.1.1: at vacuum 15 / r_cut 6 the
    two agree to 1.6e-11; at vacuum 2 they differ by 466.
    """

    mode: str = "auto"  # auto | native | box
    vacuum: float = 15.0
    shape: str = "orthorhombic"  # orthorhombic | cubic
    min_length_tol: float = 1e-6

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "native", "box"):
            raise ValueError("cell_mode must be 'auto', 'native' or 'box'")
        if self.shape not in ("orthorhombic", "cubic"):
            raise ValueError("box_shape must be 'orthorhombic' or 'cubic'")
        if self.vacuum < 0:
            raise ValueError("vacuum must be >= 0")

    def vacuum_is_sufficient(self, cutoff: float | None) -> bool:
        """Is this vacuum enough for a structure that gets boxed?

        ``ase.Atoms.center(vacuum=v)`` makes the minimum image separation
        exactly ``2v``, so boxing reproduces the non-periodic descriptor iff
        ``2v >= cutoff``.  ``cutoff is None`` means the featurizer has no
        meaningful cutoff, so there is nothing to satisfy.

        This is a pure predicate on purpose.  Whether it *applies* to a given
        structure is decided in :func:`divsel.box.ensure_cell`, which is the
        only place that knows whether the structure is actually being boxed --
        a fully periodic cell is never boxed and the vacuum never touches it.
        """
        if cutoff is None or self.mode == "native":
            return True
        return 2.0 * self.vacuum >= cutoff

    def vacuum_error_message(self, cutoff: float, where: str | None = None) -> str:
        """Explain the violation, naming the structure that triggered it."""
        subject = f"{where} needs a box, but the" if where else "The"
        return (
            f"{subject} vacuum is too small for this cutoff: "
            f"2*vacuum = {2 * self.vacuum:g} A < cutoff = {cutoff:g} A.\n"
            "A boxed structure would then interact with its own periodic images, "
            "so its descriptor would not match an equivalent non-periodic one.\n"
            f"Fix: --vacuum {cutoff / 2:.1f} or larger (recommended --vacuum {cutoff:.1f}); "
            "or --cell_mode native to refuse boxing instead; or "
            "--on_small_vacuum skip to skip just the structures that need a box.\n"
            "Fully periodic structures are never boxed and are unaffected by this."
        )


@dataclass(frozen=True)
class SamplingConfig:
    """Which frames are considered, and in what order.

    Order of operations is pinned and documented: enumerate -> stride -> shuffle
    -> batch.  Stride keeps its meaning (decorrelating consecutive MD frames);
    shuffle then only affects which batch a frame lands in.

    ``max_frames`` applies *after stride, before species filtering*, so the
    processed subset is a reproducible prefix.  The original ``--limit`` counted
    kept structures, which made the prefix depend on the species filter.
    """

    stride: int = 1
    shuffle: bool = True
    seed: int = 0
    max_frames: int | None = None
    scan: bool = True

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be >= 1 or None")
        if self.shuffle and not self.scan:
            raise ValueError(
                "--shuffle needs the scan pass for random access; "
                "use --no_scan together with --no_shuffle, or drop --no_scan."
            )


@dataclass(frozen=True)
class StreamConfig:
    """Batching and the bounded carry-forward pool.

    ``batch_size=0`` means "no streaming": one batch, exact answer.

    The pool holds ``pool_alpha * n_select`` provisional picks so the final pass
    can discard early-batch mistakes.  Locking each batch's picks in as a hard
    seed would make batch 0 -- chosen against an empty seed, with zero global
    context -- irreversible.
    """

    batch_size: int = 2000
    pool_alpha: int = 4
    pool_cap: int | None = None  # None => min(20 * n_select, 20000)
    pool_stage: str = "fps"  # fps | random
    n_procs: int = field(default_factory=lambda: max(1, mp.cpu_count() or 1))

    def __post_init__(self) -> None:
        if self.batch_size < 0:
            raise ValueError("batch_size must be >= 0 (0 disables streaming)")
        if self.pool_alpha < 1:
            raise ValueError("pool_alpha must be >= 1")
        if self.pool_stage not in ("fps", "random"):
            raise ValueError("pool_stage must be 'fps' or 'random'")
        if self.n_procs < 1:
            raise ValueError("n_procs must be >= 1")

    def resolved_pool_cap(self, n_select: int) -> int:
        if self.pool_cap is not None:
            return max(int(self.pool_cap), int(n_select))
        return max(int(n_select), min(20 * int(n_select), 20_000))


@dataclass(frozen=True)
class SelectConfig:
    """Which selection backend runs, and its knobs."""

    n_select: int
    method: str = "kmeans"  # kmeans | fps
    normalize: str = "l2"  # l2 | none
    random_state: int = 0
    min_dist2: float | None = None
    strict_determinism: bool = False
    # fps
    fps_init: str = "farthest_from_mean"  # farthest_from_mean | random | index:J
    # kmeans
    kmeans_k: int | str = "auto"  # "auto" => n_select
    kmeans_per_cluster: int = 1
    kmeans_fill: str = "fps"  # fps | none
    kmeans_trim: str = "population"  # population | fps
    kmeans_n_init: int = 10
    minibatch_threshold: int = 20_000

    def __post_init__(self) -> None:
        if self.n_select < 1:
            raise ValueError("n_select must be >= 1")
        if self.method not in ("kmeans", "fps"):
            raise ValueError("method must be 'kmeans' or 'fps'")
        if self.normalize not in ("l2", "none"):
            raise ValueError("normalize must be 'l2' or 'none'")
        if self.kmeans_fill not in ("fps", "none"):
            raise ValueError("kmeans_fill must be 'fps' or 'none'")
        if self.kmeans_trim not in ("population", "fps"):
            raise ValueError("kmeans_trim must be 'population' or 'fps'")
        if self.kmeans_per_cluster < 1:
            raise ValueError("kmeans_per_cluster must be >= 1")
        if isinstance(self.kmeans_k, str) and self.kmeans_k != "auto":
            raise ValueError("kmeans_k must be an integer or 'auto'")

    def resolved_k(self) -> int:
        """k = n_select by default.

        Deliberately not ``n_select + n_seed``: k would grow without bound across
        active-learning rounds (20 rounds x 200 picks is k > 4000 on a 10k batch),
        which turns k-means into a slow nearest-neighbour graph.  A shortfall from
        seed-covered clusters is repaired by the FPS top-up instead.
        """
        if self.kmeans_k == "auto":
            return int(self.n_select)
        return int(self.kmeans_k)


# --------------------------------------------------------------------------
# descriptor identity
# --------------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    """Make a value JSON-canonical and stable across platforms."""
    import numpy as np

    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        # Round so that 6.0 and 6.0000000001 do not produce different ids, while
        # genuinely different cutoffs still do.
        return format(round(value, 10), ".10g")
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def descriptor_id(fields: Mapping[str, Any]) -> str:
    """16-hex-character blake2b digest over the canonicalized fields.

    Callers must include everything that changes the meaning of a feature vector:
    the featurizer name, its own identity fields, and the normalization mode.
    """
    payload = json.dumps(_canonical(fields), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def format_descriptor_diff(
    seed_fields: Mapping[str, Any],
    this_fields: Mapping[str, Any],
    *,
    seed_id: str | None = None,
    this_id: str | None = None,
    seed_label: str = "seed",
    this_label: str = "this",
    header: str | None = None,
    advice: str | None = None,
) -> str:
    """Human-readable field-by-field diff for DescriptorMismatchError.

    ``header`` and ``advice`` override the first and last lines.  They exist so
    that the seed-versus-this-run case and the seed-files-versus-each-other case
    share one diff body instead of drifting into two formatters that describe
    the same comparison differently.
    """
    keys = sorted(set(seed_fields) | set(this_fields))
    differing = [k for k in keys if seed_fields.get(k) != this_fields.get(k)]
    width = max((len(k) for k in differing), default=1)
    # Labels are usually the words "seed"/"this" but may be file paths, in
    # which case the padding simply does nothing.
    label_width = max(len(seed_label), len(this_label), 5)

    lines = [
        header or "seed descriptors are not comparable to this run.",
        f"  {seed_label:<{label_width}} ({seed_id or '?'}):",
    ]
    for k in differing:
        lines.append(f"      {k:<{width}} = {seed_fields.get(k, '<absent>')!r}")
    lines.append(f"  {this_label:<{label_width}} ({this_id or '?'}):")
    for k in differing:
        lines.append(f"      {k:<{width}} = {this_fields.get(k, '<absent>')!r}")
    lines.append("")
    lines.append(
        advice
        or (
            "Recompute the seed with the parameters above, or rerun everything with "
            "the seed's parameters. Descriptors from different parameter sets are "
            "not comparable, so mixing them would silently corrupt the selection."
        )
    )
    return "\n".join(lines)


def require_same_descriptor(
    seed_fields: Mapping[str, Any],
    this_fields: Mapping[str, Any],
    **kwargs: Any,
) -> None:
    """Raise DescriptorMismatchError unless the two descriptor identities agree."""
    seed_id = descriptor_id(seed_fields)
    this_id = descriptor_id(this_fields)
    if seed_id != this_id:
        raise DescriptorMismatchError(
            format_descriptor_diff(
                seed_fields, this_fields, seed_id=seed_id, this_id=this_id, **kwargs
            )
        )
