"""On-disk descriptor storage.

Replaces the original ``natoms,features`` CSV in which each vector was a
``json.dumps`` list.  That format cost ~20 bytes per float instead of 4, had to
be re-parsed (tolerantly, with a mode-length filter that silently dropped rows)
on every use, and carried no frame identity.

Here:

* ``descriptors.npy`` -- ``(n_rows, d)`` float32, created with
  ``np.lib.format.open_memmap`` once the scan has fixed both numbers, then
  filled row-in-place.  Memory-mappable and randomly addressable.
* ``frames.csv`` -- one row per *attempted* frame:
  ``row, source_file, frame_index, natoms, formula, cell_mode, status``.
  Skipped frames get ``row = -1`` and a status saying why, so a species skip is
  recorded rather than merely counted.
* ``store.json`` -- descriptor id, dimensions, row count.

Rows are positional in a preallocated array, so the "alignment is the user's
responsibility" hazard of the original loader simply does not arise.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .log import get_logger

logger = get_logger(__name__)

__all__ = ["DescriptorStore", "seed_cache_path", "load_store"]

FRAME_COLUMNS = [
    "row",
    "source_file",
    "frame_index",
    "natoms",
    "formula",
    "cell_mode",
    "status",
]

_COPY_CHUNK = 4096  # rows per copy step when resizing; keeps memory bounded


class DescriptorStore:
    """Writer for one run's descriptor block."""

    def __init__(
        self,
        out_dir: str | Path,
        feature_dim: int,
        descriptor_id: str,
        *,
        n_rows_max: int | None = None,
    ) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.feature_dim = int(feature_dim)
        self.descriptor_id = descriptor_id
        self.n_rows_max = None if n_rows_max is None else int(n_rows_max)
        self.n_rows = 0
        self._shards: list[Path] = []
        self._frames_fh = open(self.dir / "frames.csv", "w", newline="")
        self._frames_writer = csv.DictWriter(self._frames_fh, fieldnames=FRAME_COLUMNS)
        self._frames_writer.writeheader()

        if self.n_rows_max is not None:
            self._mm = np.lib.format.open_memmap(
                self.dir / "descriptors.npy",
                mode="w+",
                dtype=np.float32,
                shape=(self.n_rows_max, self.feature_dim),
            )
            logger.debug(
                "allocated descriptors.npy (%d, %d) float32 = %.2f GB",
                self.n_rows_max,
                self.feature_dim,
                self.n_rows_max * self.feature_dim * 4 / 1e9,
            )
        else:
            # --no_scan: the row count is unknown, so buffer shards and
            # consolidate in finalize().
            self._mm = None
            logger.debug("no row count known; using shard mode")

    # -- writing -----------------------------------------------------------
    def append(self, X: np.ndarray, meta: Sequence[dict]) -> None:
        """Append one batch.

        ``meta`` has one record per attempted frame with ``local_row`` indexing
        ``X`` (or -1 for a skipped frame); it is rewritten to global rows.
        """
        n = int(X.shape[0])
        if X.shape[1] != self.feature_dim:
            raise ValueError(
                f"batch has dimension {X.shape[1]}, store expects {self.feature_dim}"
            )
        base = self.n_rows

        if n:
            if self._mm is not None:
                if base + n > self.n_rows_max:
                    raise RuntimeError(
                        f"store overflow: {base + n} rows written into a store "
                        f"allocated for {self.n_rows_max}"
                    )
                self._mm[base : base + n] = X
            else:
                shard = self.dir / f"_shard_{len(self._shards):05d}.npy"
                np.save(shard, X)
                self._shards.append(shard)
        self.n_rows += n

        for record in meta:
            local = record.get("local_row", -1)
            row = base + local if local is not None and local >= 0 else -1
            self._frames_writer.writerow(
                {
                    "row": row,
                    "source_file": record["source_file"],
                    "frame_index": record["frame_index"],
                    "natoms": record["natoms"],
                    "formula": record["formula"],
                    "cell_mode": record["cell_mode"],
                    "status": record["status"],
                }
            )
        self._frames_fh.flush()

    # -- closing -----------------------------------------------------------
    def finalize(self) -> int:
        """Flush, right-size ``descriptors.npy``, and write ``store.json``."""
        self._frames_fh.close()

        if self._mm is not None:
            self._mm.flush()
            del self._mm
            self._mm = None
            if self.n_rows < (self.n_rows_max or 0):
                # Frames were skipped, so the preallocated array is too long.
                # Copy down in chunks rather than loading it all.
                self._resize_to(self.n_rows)
        else:
            self._consolidate_shards()

        (self.dir / "store.json").write_text(
            json.dumps(
                {
                    "descriptor_id": self.descriptor_id,
                    "n_rows": int(self.n_rows),
                    "feature_dim": int(self.feature_dim),
                    "dtype": "float32",
                },
                indent=2,
            )
        )
        return self.n_rows

    def _resize_to(self, n_rows: int) -> None:
        src_path = self.dir / "descriptors.npy"
        tmp_path = self.dir / "_descriptors_tmp.npy"
        src = np.load(src_path, mmap_mode="r")
        dst = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=np.float32, shape=(n_rows, self.feature_dim)
        )
        for start in range(0, n_rows, _COPY_CHUNK):
            stop = min(start + _COPY_CHUNK, n_rows)
            dst[start:stop] = src[start:stop]
        dst.flush()
        del dst, src
        os.replace(tmp_path, src_path)
        logger.debug("trimmed descriptors.npy to %d rows", n_rows)

    def _consolidate_shards(self) -> None:
        dst = np.lib.format.open_memmap(
            self.dir / "descriptors.npy",
            mode="w+",
            dtype=np.float32,
            shape=(self.n_rows, self.feature_dim),
        )
        at = 0
        for shard in self._shards:
            block = np.load(shard, mmap_mode="r")
            dst[at : at + len(block)] = block
            at += len(block)
            del block
            shard.unlink()
        dst.flush()
        del dst
        self._shards.clear()

    def __enter__(self) -> "DescriptorStore":
        return self

    def __exit__(self, *exc) -> None:
        if not self._frames_fh.closed:
            self.finalize()


def load_store(out_dir: str | Path) -> tuple[np.ndarray, "object", dict]:
    """Open a finished store: ``(X_memmap, frames_dataframe, store_meta)``."""
    import pandas as pd

    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "store.json").read_text())
    X = np.load(out_dir / "descriptors.npy", mmap_mode="r")
    frames = pd.read_csv(out_dir / "frames.csv")
    return X, frames, meta


def seed_cache_path(
    descriptor_id: str, source: str | Path, *, cache_root: str | Path | None = None
) -> Path:
    """Where a seed file's descriptors are cached.

    Keyed on the descriptor id *directory* and a cheap file fingerprint.  The
    original cached on filename alone, so changing ``--soap-rcut`` from 6.0 to
    5.0 produced a cache hit on stale vectors -- undetectable by a shape check,
    since r_cut does not change the descriptor length.  Here a parameter change
    lands in a different directory and simply cannot collide.
    """
    root = Path(cache_root) if cache_root is not None else Path(".divsel_cache")
    source = Path(source)
    try:
        stat = source.stat()
        fingerprint = f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        fingerprint = str(source)
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:24]
    return root / descriptor_id / f"{source.stem}_{digest}.npy"
