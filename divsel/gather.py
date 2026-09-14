"""Pull the selected frames out of their source files into one trajectory.

This replaces ``orginal_code_v1/gather_images.py``, which was a separate manual
step with hardcoded ``csv_path`` / ``traj_path`` / ``out_train`` and which
assumed every selected frame came from a single trajectory.  Here it runs
automatically at the end of every selection, and is also exposed as
``divsel gather`` for rebuilding a trajectory from a CSV later.

Kept from the original because it was right:

* streaming -- random-access read, write one frame at a time, never a list of
  Atoms in memory.  This matters once ``n_select`` is in the thousands.
* a bounds check against the source length, naming the out-of-range indices.
* de-duplication of repeated indices, keeping the first occurrence.

Fixed:

* multi-file selections, via the ``source_file`` column;
* output ordered by selection rank, so ``selected.traj[rank]`` is
  ``selected.csv`` row ``rank``;
* no hardcoded paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ase import Atoms

from .log import get_logger

logger = get_logger(__name__)

__all__ = ["gather", "gather_from_csv"]


class _SourceReader:
    """Random access to frames of one file, opened once."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._traj = None
        self._images: list[Atoms] | None = None
        if self.path.suffix.lower() == ".traj":
            from ase.io.trajectory import Trajectory

            self._traj = Trajectory(str(self.path))
        else:
            from ase.io import read as ase_read

            images = ase_read(str(self.path), index=":")
            self._images = images if isinstance(images, list) else [images]

    def __len__(self) -> int:
        return len(self._traj) if self._traj is not None else len(self._images or [])

    def __getitem__(self, i: int) -> Atoms:
        if self._traj is not None:
            return self._traj[i]
        assert self._images is not None
        return self._images[i]

    def close(self) -> None:
        if self._traj is not None:
            self._traj.close()


def gather(
    rows: Sequence[Mapping[str, Any]],
    out_traj: str | Path,
    *,
    descriptor_id: str | None = None,
    run_id: str | None = None,
    extra_info: Mapping[str, Any] | None = None,
) -> int:
    """Write the frames named by ``rows`` to ``out_traj``, in row order.

    Each row needs ``source_file`` and ``frame_index``; ``rank``, ``cluster``
    and ``score`` are used for the stamps when present.

    Every written frame is stamped so the trajectory is self-describing and the
    next active-learning round can validate it automatically::

        atoms.info["divsel_descriptor_id"]   # -> next round checks this
        atoms.info["divsel_run_id"]
        atoms.info["divsel_src"]             # "<file>:<frame>"
        atoms.info["divsel_rank"]
        atoms.info["original_index"]         # legacy aliases, kept so existing
        atoms.info["cluster"]                # downstream scripts keep working
    """
    from ase.io.trajectory import Trajectory

    out_traj = Path(out_traj)
    out_traj.parent.mkdir(parents=True, exist_ok=True)

    # de-duplicate on (file, frame), keeping the first occurrence
    seen: set[tuple[str, int]] = set()
    ordered: list[Mapping[str, Any]] = []
    for row in rows:
        key = (str(row["source_file"]), int(row["frame_index"]))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(row)
    if len(ordered) != len(rows):
        logger.warning(
            "gather: dropped %d duplicate (file, frame) entries", len(rows) - len(ordered)
        )

    readers: dict[str, _SourceReader] = {}
    written = 0
    out_of_range: list[tuple[str, int]] = []

    try:
        with Trajectory(str(out_traj), mode="w") as out:
            for rank, row in enumerate(ordered):
                src = str(row["source_file"])
                idx = int(row["frame_index"])
                reader = readers.get(src)
                if reader is None:
                    reader = readers[src] = _SourceReader(src)
                if not 0 <= idx < len(reader):
                    out_of_range.append((src, idx))
                    continue

                atoms = reader[idx]
                info = dict(atoms.info)
                info["divsel_src"] = f"{src}:{idx}"
                info["divsel_rank"] = int(row.get("rank", rank))
                info["original_index"] = idx  # legacy alias
                if descriptor_id is not None:
                    info["divsel_descriptor_id"] = descriptor_id
                if run_id is not None:
                    info["divsel_run_id"] = run_id
                if row.get("cluster") is not None:
                    info["cluster"] = int(row["cluster"])  # legacy alias
                if row.get("score") is not None:
                    info["divsel_score"] = float(row["score"])
                if extra_info:
                    info.update(extra_info)
                atoms.info = info
                out.write(atoms)
                written += 1
    finally:
        for reader in readers.values():
            reader.close()

    if out_of_range:
        shown = out_of_range[:5]
        logger.warning(
            "gather: %d selected frame(s) were out of range and were skipped "
            "(examples: %s)",
            len(out_of_range),
            shown,
        )

    logger.info("gather: wrote %d structure(s) -> %s", written, out_traj)
    return written


def gather_from_csv(
    csv_path: str | Path,
    out_traj: str | Path,
    *,
    fallback_source: str | Path | None = None,
    **kwargs: Any,
) -> int:
    """``divsel gather`` -- rebuild a trajectory from a selection CSV.

    Accepts the divsel schema (``source_file``, ``frame_index``) and, for
    hand-made or legacy files, the old ``original_index`` / ``cluster`` pair --
    in which case ``fallback_source`` supplies the trajectory those indices
    refer to.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, skipinitialspace=True)

    if "source_file" in df.columns and "frame_index" in df.columns:
        rows = df.to_dict("records")
    elif "original_index" in df.columns:
        if fallback_source is None:
            raise ValueError(
                f"{csv_path} uses the legacy 'original_index' schema, which does not "
                "say which file the indices refer to. Pass --traj to name it."
            )
        df = df.copy()
        df["frame_index"] = pd.to_numeric(df["original_index"], errors="coerce")
        bad = df["frame_index"].isna()
        if bad.any():
            logger.warning("dropped %d malformed row(s) from %s", int(bad.sum()), csv_path)
            df = df.loc[~bad]
        df["frame_index"] = df["frame_index"].astype(int)
        df["source_file"] = str(fallback_source)
        rows = df.to_dict("records")
    else:
        raise ValueError(
            f"{csv_path} has neither (source_file, frame_index) nor original_index; "
            f"columns are {list(df.columns)}"
        )

    return gather(rows, out_traj, **kwargs)
