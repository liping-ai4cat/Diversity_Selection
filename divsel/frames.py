"""Enumerating, indexing and reading input frames.

The scan pass is what makes everything downstream sizeable.  It reads atomic
numbers and frame counts only -- never a descriptor -- so it is cheap, and it
produces four things:

1. **The species union over every frame.**  This fixes the descriptor length
   before any SOAP is computed, so ``descriptors.npy`` can be allocated once as
   an ``(N, d)`` memmap and filled row-in-place.  Scanning *all* frames matters:
   the original v1 script sampled the first image of at most 100 files, so an
   element appearing only in a later frame was missed -- which then either
   crashed mid-run or silently skipped structures.
2. ``N``, the number of frames after stride, which sizes the memmap and the
   per-batch quotas.
3. Per-frame byte offsets for .xyz/.extxyz, giving the exact random access that
   ``--shuffle`` needs.  ``.traj`` already has O(1) random access.
4. ``natoms`` per frame where it is free to collect.

Order of operations is pinned: **enumerate -> stride -> max_frames -> shuffle
-> batch**.  Stride keeps its meaning (decorrelating consecutive MD frames),
``max_frames`` is therefore a reproducible prefix, and shuffle only decides
which batch a frame lands in.  Shuffling is global across all input files --
shuffling within each file would leave every batch file-homogeneous, which for
MD trajectories means chemically homogeneous, which is the whole problem.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from ase import Atoms

from .errors import ScanError
from .log import get_logger

logger = get_logger(__name__)

__all__ = [
    "FrameRef",
    "FrameIndex",
    "scan_sources",
    "iter_frames_sequential",
    "expand_patterns",
    "peek_first_frame",
]

_XYZ_SUFFIXES = {".xyz", ".extxyz"}


@dataclass(frozen=True)
class FrameRef:
    """Where a descriptor row came from.

    This is the provenance primitive.  The original pipeline carried only a
    running counter of kept structures, written out of order by
    ``imap_unordered``, so a row number did not identify a source frame.
    """

    file_id: int
    frame_index: int
    natoms: int = -1

    def as_key(self) -> tuple[int, int]:
        return (self.file_id, self.frame_index)


@dataclass
class _Source:
    path: Path
    kind: str  # traj | xyz | generic
    n_frames: int
    offsets: np.ndarray | None = None  # byte offset of each frame (xyz only)
    lengths: np.ndarray | None = None  # byte length of each frame (xyz only)
    natoms: np.ndarray | None = None


def expand_patterns(patterns: Sequence[str]) -> list[Path]:
    """Resolve globs to a sorted, de-duplicated list of existing files."""
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = sorted(glob(str(pattern)))
        if not matches:
            candidate = Path(pattern)
            if candidate.exists():
                matches = [str(candidate)]
        if not matches:
            raise ScanError(f"no files matched {pattern!r}")
        for m in matches:
            rp = str(Path(m).resolve())
            if rp not in seen:
                seen.add(rp)
                paths.append(Path(m))
    if not paths:
        raise ScanError(f"no input files found for {list(patterns)!r}")
    return paths


def peek_first_frame(path: str | Path) -> Atoms | None:
    """Read just the first frame of a file, or None if it cannot be read.

    Used by the boxing probe to answer "will anything in this file need a box?"
    before the scan pass starts.  One frame per input file is negligible, and
    it turns a mid-run failure into an immediate one for the common case where
    a file is periodicity-homogeneous.  Returning None on failure is
    deliberate: a probe must never be the thing that breaks a run, the real
    read will report the problem properly.
    """
    path = Path(path)
    try:
        if path.suffix.lower() == ".traj":
            from ase.io.trajectory import Trajectory

            with Trajectory(str(path)) as traj:
                return traj[0] if len(traj) else None
        from ase.io import read as ase_read

        return ase_read(str(path), index=0)
    except Exception as exc:  # pragma: no cover - probe must not be fatal
        logger.debug("could not peek at %s: %s", path, exc)
        return None


def _kind_of(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".traj":
        return "traj"
    if suffix in _XYZ_SUFFIXES:
        return "xyz"
    return "generic"


def _scan_xyz(path: Path, collect_species: bool) -> _Source:
    """Index an (ext)xyz file by byte offset.

    Reads the natoms header and skips natoms+1 lines per frame, so this runs at
    I/O speed and is negligible next to descriptor computation.  When species
    are needed we also take the first token of each atom line, which is far
    cheaper than a full ASE parse.
    """
    offsets: list[int] = []
    natoms: list[int] = []
    species: set[str] = set()

    with open(path, "rb") as fh:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                na = int(stripped)
            except ValueError:
                raise ScanError(
                    f"{path}: expected an atom count at byte {pos}, got {stripped[:40]!r}. "
                    "Is this really an (ext)xyz file?"
                ) from None
            offsets.append(pos)
            natoms.append(na)
            if not fh.readline():
                raise ScanError(f"{path}: truncated after the atom count at byte {pos}")
            for _ in range(na):
                atom_line = fh.readline()
                if not atom_line:
                    raise ScanError(f"{path}: truncated inside the frame at byte {pos}")
                if collect_species:
                    parts = atom_line.split(None, 1)
                    if parts:
                        species.add(parts[0].decode("utf-8", "replace"))

    size = path.stat().st_size
    off = np.asarray(offsets, dtype=np.int64)
    lengths = np.diff(np.append(off, size)) if off.size else np.empty(0, np.int64)

    src = _Source(
        path=path,
        kind="xyz",
        n_frames=len(offsets),
        offsets=off,
        lengths=lengths,
        natoms=np.asarray(natoms, dtype=np.int64),
    )
    src_species = species  # attached by the caller
    setattr(src, "_species_tokens", src_species)
    return src


def _species_from_tokens(tokens: Iterable[str]) -> set[int]:
    from ase.data import atomic_numbers

    out: set[int] = set()
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            out.add(int(tok))
            continue
        symbol = tok.capitalize() if tok.isalpha() else tok
        if symbol in atomic_numbers:
            out.add(atomic_numbers[symbol])
        else:
            raise ScanError(
                f"unrecognised element token {tok!r} while scanning; "
                "pass --species explicitly if the file uses non-standard labels"
            )
    return out


class FrameIndex:
    """The frames this run will process, in the order it will process them."""

    def __init__(
        self,
        sources: list[_Source],
        refs: list[FrameRef],
        species: set[int] | None,
        *,
        n_before_stride: int,
    ) -> None:
        self._sources = sources
        self.refs = refs
        self.species = species
        self.n_before_stride = n_before_stride
        self._cache_file_id: int | None = None
        self._cache_images: list[Atoms] | None = None

    # -- introspection -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.refs)

    @property
    def files(self) -> list[Path]:
        return [s.path for s in self._sources]

    def file_path(self, file_id: int) -> Path:
        return self._sources[file_id].path

    def source_summary(self) -> list[dict]:
        out = []
        for s in self._sources:
            stat = s.path.stat()
            out.append(
                {
                    "path": str(s.path),
                    "n_frames": int(s.n_frames),
                    "size_bytes": int(stat.st_size),
                    "mtime": float(stat.st_mtime),
                }
            )
        return out

    # -- reading -----------------------------------------------------------
    def read(self, ref: FrameRef) -> Atoms:
        """Random access to one frame."""
        src = self._sources[ref.file_id]
        if src.kind == "traj":
            from ase.io.trajectory import Trajectory

            with Trajectory(str(src.path)) as traj:
                return traj[ref.frame_index]
        if src.kind == "xyz":
            from ase.io import read as ase_read

            assert src.offsets is not None and src.lengths is not None
            with open(src.path, "rb") as fh:
                fh.seek(int(src.offsets[ref.frame_index]))
                blob = fh.read(int(src.lengths[ref.frame_index]))
            text = io.StringIO(blob.decode("utf-8", "replace"))
            return ase_read(text, format="extxyz", index=0)
        # generic: cache the whole file, one file at a time
        if self._cache_file_id != ref.file_id:
            from ase.io import read as ase_read

            images = ase_read(str(src.path), index=":")
            self._cache_images = images if isinstance(images, list) else [images]
            self._cache_file_id = ref.file_id
        assert self._cache_images is not None
        return self._cache_images[ref.frame_index]

    def iter_slots(self, refs: Sequence[FrameRef]) -> Iterator[tuple[int, Atoms]]:
        """Yield ``(slot, atoms)`` for a batch, grouping by file.

        ``slot`` is the position within ``refs``, not a global row.  Grouping by
        file means each trajectory is opened once per batch instead of once per
        frame.
        """
        by_file: dict[int, list[tuple[int, FrameRef]]] = {}
        for slot, ref in enumerate(refs):
            by_file.setdefault(ref.file_id, []).append((slot, ref))

        for file_id, items in by_file.items():
            src = self._sources[file_id]
            if src.kind == "traj":
                from ase.io.trajectory import Trajectory

                with Trajectory(str(src.path)) as traj:
                    for slot, ref in items:
                        yield slot, traj[ref.frame_index]
            else:
                for slot, ref in items:
                    yield slot, self.read(ref)

    def batches(self, batch_size: int) -> Iterator[list[FrameRef]]:
        """Split the (already ordered) refs into batches.

        ``batch_size <= 0`` means one batch: no streaming, exact answer.
        """
        if batch_size is None or batch_size <= 0 or batch_size >= len(self.refs):
            if self.refs:
                yield list(self.refs)
            return
        for start in range(0, len(self.refs), batch_size):
            yield list(self.refs[start : start + batch_size])

    def n_batches(self, batch_size: int) -> int:
        if batch_size is None or batch_size <= 0 or batch_size >= len(self.refs):
            return 1 if self.refs else 0
        return (len(self.refs) + batch_size - 1) // batch_size


def scan_sources(
    patterns: Sequence[str],
    *,
    stride: int = 1,
    max_frames: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    collect_species: bool = True,
) -> FrameIndex:
    """Enumerate and index every input frame.

    Set ``collect_species=False`` when the user gave ``--species`` explicitly:
    counting frames is O(1) for .traj and an I/O-speed scan for .xyz, whereas
    deriving the species union from a .traj requires actually reading frames.
    """
    paths = expand_patterns(patterns)
    sources: list[_Source] = []
    species: set[int] | None = set() if collect_species else None

    for path in paths:
        kind = _kind_of(path)
        if kind == "traj":
            from ase.io.trajectory import Trajectory

            with Trajectory(str(path)) as traj:
                n = len(traj)
                if collect_species:
                    # No cheap numbers-only path exists for .traj, so this reads
                    # frames.  Still far cheaper than SOAP (no neighbour lists),
                    # and correctness requires seeing every frame.
                    nat = np.empty(n, dtype=np.int64)
                    for i in range(n):
                        atoms = traj[i]
                        species.update(int(z) for z in atoms.numbers)
                        nat[i] = len(atoms)
                    src = _Source(path, kind, n, natoms=nat)
                else:
                    src = _Source(path, kind, n)
        elif kind == "xyz":
            src = _scan_xyz(path, collect_species)
            if collect_species:
                species.update(_species_from_tokens(getattr(src, "_species_tokens")))
        else:
            from ase.io import read as ase_read

            images = ase_read(str(path), index=":")
            images = images if isinstance(images, list) else [images]
            src = _Source(
                path,
                kind,
                len(images),
                natoms=np.asarray([len(a) for a in images], dtype=np.int64),
            )
            if collect_species:
                for a in images:
                    species.update(int(z) for z in a.numbers)

        logger.debug("scanned %s: %d frames (%s)", path, src.n_frames, src.kind)
        sources.append(src)

    # enumerate -> stride -> max_frames -> shuffle
    all_refs: list[FrameRef] = []
    for file_id, src in enumerate(sources):
        nat = src.natoms
        for i in range(src.n_frames):
            all_refs.append(
                FrameRef(file_id, i, int(nat[i]) if nat is not None else -1)
            )
    n_before_stride = len(all_refs)

    refs = all_refs[:: max(1, int(stride))]
    if max_frames is not None:
        refs = refs[: int(max_frames)]
    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(refs))
        refs = [refs[i] for i in order]

    logger.info(
        "scan: %d frames in %d file(s) -> %d after stride=%d%s%s",
        n_before_stride,
        len(sources),
        len(refs),
        stride,
        f", max_frames={max_frames}" if max_frames else "",
        ", shuffled" if shuffle else "",
    )
    if collect_species and species:
        from ase.data import chemical_symbols

        logger.debug(
            "scan: species union = %s",
            [f"{chemical_symbols[z]}({z})" for z in sorted(species)],
        )

    return FrameIndex(sources, refs, species, n_before_stride=n_before_stride)


def iter_frames_sequential(
    patterns: Sequence[str],
    *,
    stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[tuple[FrameRef, Atoms]]:
    """Stream frames without a scan pass (``--no_scan``).

    No frame count, no random access, so no shuffle and no preallocated memmap.
    Supported as a fallback; the scan path is strictly better when available.
    """
    from ase.io import iread

    paths = expand_patterns(patterns)
    kept = 0
    seen = 0
    for file_id, path in enumerate(paths):
        for frame_index, atoms in enumerate(iread(str(path), index=":")):
            if seen % max(1, int(stride)) != 0:
                seen += 1
                continue
            seen += 1
            yield FrameRef(file_id, frame_index, len(atoms)), atoms
            kept += 1
            if max_frames is not None and kept >= max_frames:
                return
