"""Selection on precomputed feature vectors -- a backend that looks up, not computes.

``--input_features`` points at a table of vectors somebody else already made,
and selection proceeds on those instead of generating SOAP.  Two on-disk forms
are accepted:

**A ``latent_features`` output set** (``<prefix>.npz`` + ``<prefix>.index.csv``
+ ``<prefix>.metadata.json``), as written by
``latent_spaces_as_features/latent_features/io.py``::

    np.savez(self.npz_path, features=features, valid=valid)

**A previous divsel output directory** (``descriptors.npy`` + ``frames.csv`` +
``store.json``).  This is the round-to-round reuse path: round 2 can point at
round 1's directory and skip regenerating identical SOAP.

Why a Featurizer rather than a shortcut
---------------------------------------
This satisfies the ordinary :class:`~divsel.featurizers.base.Featurizer`
contract, so the batch loop, the descriptor store, ``select_diverse``, the
seed path, the manifest and the ``descriptor_id`` compatibility guard all work
unchanged -- ``featurize`` just indexes an array instead of calling a model.
The alternative, teaching :mod:`divsel.streaming` to bypass featurization, would
have duplicated all of that.

Why the join is by provenance and never by position
---------------------------------------------------
``features[k]`` is **not** input frame ``k``.  ``latent_features`` drops whole
structures under ``on_error="skip"`` with no placeholder and no log line
(``extractor.py:271``), and its ``surface``/``per_atom`` modes emit several rows
per structure.  Its ``row`` column is dense and renumbered afterwards, so a skip
leaves no trace in it.  Meanwhile divsel applies ``--stride``, ``--max_frames``
and ``--shuffle`` to its own frame list.  The two orderings have no reason to
agree, so every row is keyed by ``"<basename>:<frame_index>"`` and looked up.
A positional join would appear to work on the happy path and silently pair the
wrong vector with the wrong structure the moment either side dropped anything.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms

from ..errors import FeatureTableError
from ..log import get_logger
from .base import Featurizer

logger = get_logger(__name__)

__all__ = ["TableFeaturizer", "export_table", "provenance_key", "SRC_INFO_KEY"]

#: ``atoms.info`` key carrying "<source_file>:<frame_index>".  Written by
#: :func:`divsel.descriptors._prepare_frames` for candidates and by
#: :func:`divsel.gather.gather` for previously-selected frames, which is what
#: lets a seed trajectory be resolved against the same table.
SRC_INFO_KEY = "divsel_src"


def provenance_key(source_file: Any, frame_index: Any) -> str:
    """The join key: basename of the file, then the frame index.

    Basename rather than full path because ``latent_features`` records only
    ``os.path.basename(path)`` (``sources.py:139``) and cannot be made to record
    more after the fact.  :meth:`TableFeaturizer.check_coverage` rejects an input
    set whose basenames collide, so discarding the directory is safe there.
    """
    return f"{os.path.basename(str(source_file))}:{int(frame_index)}"


def key_from_src(src: str) -> str:
    """Normalize an ``atoms.info['divsel_src']`` value into a table key."""
    path, _, frame = str(src).rpartition(":")
    if not path:  # no colon at all -- not a provenance string
        raise FeatureTableError(f"malformed {SRC_INFO_KEY} value: {src!r}")
    return provenance_key(path, frame)


# --------------------------------------------------------------------------
# loading the two accepted on-disk forms
# --------------------------------------------------------------------------


class _Table:
    """Rows plus the provenance keys and identity that make them meaningful."""

    def __init__(
        self,
        X: np.ndarray,
        valid: np.ndarray,
        keys: Sequence[str],
        groups: Sequence[str] | None,
        identity: dict,
        described: str,
        skipped: dict[str, str] | None = None,
    ) -> None:
        self.X = X
        self.valid = valid
        self.keys = list(keys)
        self.groups = list(groups) if groups is not None else None
        self.identity = identity
        self.described = described
        #: key -> the status the source run recorded for a frame it dropped
        self.skipped = dict(skipped or {})


def _resolve(source: str | Path) -> tuple[str, Path]:
    """Classify the source path.  Returns ``(kind, path)``."""
    p = Path(source)
    if p.is_dir() and (p / "descriptors.npy").exists():
        return "divsel", p
    prefix = p.with_suffix("") if p.suffix == ".npz" else p
    if prefix.with_suffix(".npz").exists() or Path(str(prefix) + ".npz").exists():
        return "latent_features", prefix
    raise FeatureTableError(
        f"--input_features {source!r} is neither a divsel output directory "
        f"(expected {p / 'descriptors.npy'}) nor a latent_features prefix "
        f"(expected {prefix}.npz).\n"
        "Pass either the .npz written by latent-features (its .index.csv must "
        "sit beside it) or a directory written by a previous `divsel select`."
    )


def _load_latent_features(prefix: Path) -> _Table:
    npz_path = Path(str(prefix) + ".npz")
    index_path = Path(str(prefix) + ".index.csv")
    meta_path = Path(str(prefix) + ".metadata.json")

    if not index_path.exists():
        raise FeatureTableError(
            f"{npz_path} has no sidecar {index_path.name}.\n"
            "The index CSV is the only sound way to map rows to frames -- row "
            "order alone cannot be trusted, because latent_features drops "
            "structures silently under --skip-errors. Copy the .index.csv "
            "next to the .npz."
        )

    with np.load(npz_path, allow_pickle=False) as z:
        if "features" not in z:
            raise FeatureTableError(
                f"{npz_path} has no 'features' array (found: {list(z.keys())}). "
                "Expected the latent_features layout: features, valid."
            )
        X = np.asarray(z["features"])
        valid = (
            np.asarray(z["valid"], dtype=bool)
            if "valid" in z
            else np.ones(len(X), dtype=bool)
        )

    with open(index_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != len(X):
        raise FeatureTableError(
            f"{index_path.name} has {len(rows)} rows but {npz_path.name} has "
            f"{len(X)} -- they are not from the same run, so no join is safe."
        )

    cols = set(rows[0]) if rows else set()
    keys: list[str] = []
    for i, r in enumerate(rows):
        if "source_file" in cols and "frame_index" in cols and r.get("frame_index", ""):
            keys.append(provenance_key(r["source_file"], r["frame_index"]))
        elif "structure_id" in cols and ":" in str(r.get("structure_id", "")):
            # "<basename>:<frame_index>" for file input (sources.py:145)
            keys.append(key_from_src(r["structure_id"]))
        else:
            raise FeatureTableError(
                f"{index_path.name} row {i} carries no usable provenance: need "
                "either source_file + frame_index columns, or a structure_id of "
                "the form '<file>:<frame>'. A table built from an ASE database "
                "(db_id only) cannot be joined to a trajectory."
            )

    groups = [r.get("group_label", "") for r in rows] if "group_label" in cols else None

    metadata: dict = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception:  # pragma: no cover - corrupt sidecar is not fatal
            logger.warning("could not parse %s; identity will be coarser", meta_path)
    else:
        # Not fatal, but it costs the identity its teeth: with no backend or
        # model_id recorded, two unrelated embeddings of the same width hash
        # alike and the cross-round seed guard can no longer separate them.
        logger.warning(
            "%s is missing, so descriptor_id cannot record which model made "
            "these vectors; two different embeddings of the same width would "
            "hash alike. Copy the .metadata.json next to the .npz.",
            meta_path.name,
        )

    # Semantic identity only -- what the vectors *mean*. Deliberately excludes
    # run_uid and any content hash: re-extracting with the same model and mode
    # yields comparable vectors, and pinning the bytes would reject that.
    identity = {
        "featurizer": "latent_features",
        "backend": metadata.get("backend"),
        "model_id": metadata.get("model_id"),
        "task_name": metadata.get("task_name"),
        "mode": metadata.get("mode"),
        "layer_spans": metadata.get("layer_spans") or metadata.get("layer_dims"),
        "feature_dim": int(X.shape[1]) if X.ndim == 2 else 0,
    }
    return _Table(X, valid, keys, groups, identity, f"latent_features {npz_path.name}")


def _load_divsel_dir(d: Path) -> _Table:
    X = np.load(d / "descriptors.npy", mmap_mode="r")
    frames_path = d / "frames.csv"
    if not frames_path.exists():
        raise FeatureTableError(
            f"{d} has descriptors.npy but no frames.csv, so its rows cannot be "
            "mapped back to source frames."
        )
    with open(frames_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    keys: list[str] = []
    take: list[int] = []
    skipped: dict[str, str] = {}
    for r in rows:
        row = int(r["row"])
        key = provenance_key(r["source_file"], r["frame_index"])
        if row < 0:
            # That run dropped this frame and said why. Remember the reason so
            # reuse reproduces the same frame set instead of failing on a gap
            # the user cannot close.
            skipped[key] = r.get("status") or "skipped in the source run"
            continue
        take.append(row)
        keys.append(key)
    if len(take) != len(X):
        raise FeatureTableError(
            f"{frames_path.name} accounts for {len(take)} rows but "
            f"descriptors.npy has {len(X)}; the directory is inconsistent."
        )
    order = np.argsort(np.asarray(take, dtype=np.int64))
    X = np.asarray(X)[np.asarray(take, dtype=np.int64)[order]]
    keys = [keys[i] for i in order]

    # Reuse the previous run's descriptor identity verbatim, so re-selecting
    # from round 1's SOAP hashes to the same id round 1 had -- which is what
    # keeps `--selected_images round1/selected.traj` validating.
    identity: dict = {}
    manifest = d / "run_manifest.json"
    if manifest.exists():
        try:
            desc = dict(json.loads(manifest.read_text()).get("descriptor", {}))
            desc.pop("descriptor_id", None)
            desc.pop("normalize", None)  # re-applied per run, added by the caller
            identity = desc
        except Exception:  # pragma: no cover
            logger.warning("could not parse %s; identity will be coarser", manifest)
    if not identity:
        store = d / "store.json"
        did = json.loads(store.read_text()).get("descriptor_id") if store.exists() else None
        identity = {"featurizer": "divsel_store", "source_descriptor_id": did}

    valid = np.ones(len(X), dtype=bool)
    return _Table(X, valid, keys, None, identity, f"divsel store {d}", skipped)


# --------------------------------------------------------------------------
# the backend
# --------------------------------------------------------------------------


class TableFeaturizer(Featurizer):
    """Serves precomputed vectors, keyed by ``<basename>:<frame_index>``."""

    name = "table"
    #: a dict lookup must not pay process-pool spawn cost
    parallel_mode = "in_process_batched"
    #: never reads coordinates, so boxing and the vacuum guard are skipped
    needs_geometry = False

    def __init__(
        self,
        *,
        source: str | Path,
        feature_group: str | None = None,
        cfg: Any = None,  # accepted and ignored: SOAP settings are irrelevant here
    ) -> None:
        self.source = str(source)
        self.feature_group = feature_group
        kind, path = _resolve(source)
        table = _load_divsel_dir(path) if kind == "divsel" else _load_latent_features(path)

        if table.groups is not None:
            table = self._select_group(table)

        X = np.ascontiguousarray(table.X, dtype=np.float32)
        self._valid = table.valid
        self._skipped = dict(table.skipped)
        self._identity = dict(table.identity)
        # Only record the group when one was actually chosen. Adding a null key
        # would change descriptor_id, and reusing a previous run's descriptors
        # must hash to exactly what that run hashed to -- otherwise seeding the
        # new run with the old run's selected.traj raises a spurious mismatch.
        if feature_group is not None:
            self._identity["feature_group"] = feature_group
        self._described = table.described

        self._row: dict[str, int] = {}
        dupes: list[str] = []
        for i, k in enumerate(table.keys):
            if k in self._row:
                dupes.append(k)
            else:
                self._row[k] = i
        if dupes:
            raise FeatureTableError(
                f"{table.described}: {len(dupes)} frame(s) have more than one "
                f"row after group selection, e.g. {dupes[:5]}. The join would be "
                "ambiguous. Pass --feature_group to pick one row per structure."
            )

        self._X = X
        logger.info(
            "feature table: %d row(s) x %d dim from %s",
            len(self._X),
            self.feature_dim,
            self._described,
        )

    # -- construction helpers ---------------------------------------------
    def _select_group(self, table: "_Table") -> "_Table":
        labels = sorted({g for g in (table.groups or [])})
        if self.feature_group is not None:
            if self.feature_group not in labels:
                raise FeatureTableError(
                    f"--feature_group {self.feature_group!r} is not in this table; "
                    f"available: {labels}"
                )
            keep = [i for i, g in enumerate(table.groups or []) if g == self.feature_group]
        elif len(labels) == 1:
            keep = list(range(len(table.keys)))
        else:
            raise FeatureTableError(
                f"{table.described} has {len(labels)} group labels {labels}, i.e. "
                "more than one row per structure. divsel needs exactly one vector "
                "per frame -- pass --feature_group to choose which."
            )
        idx = np.asarray(keep, dtype=np.int64)
        return _Table(
            np.asarray(table.X)[idx],
            np.asarray(table.valid)[idx],
            [table.keys[i] for i in keep],
            None,
            table.identity,
            table.described,
            table.skipped,
        )

    # -- Featurizer contract -----------------------------------------------
    def prepare(self, frame_index: Any) -> None:
        """Nothing to do: the table was fully loaded in __init__."""

    @property
    def feature_dim(self) -> int:
        return int(self._X.shape[1]) if self._X.ndim == 2 else 0

    def featurize(self, atoms_list: Sequence[Atoms]) -> np.ndarray:
        out = np.empty((len(atoms_list), self.feature_dim), dtype=np.float32)
        for i, atoms in enumerate(atoms_list):
            src = atoms.info.get(SRC_INFO_KEY)
            if src is None:
                raise FeatureTableError(
                    f"a structure reached the table backend without "
                    f"{SRC_INFO_KEY!r} in atoms.info, so it cannot be looked up. "
                    "This is a divsel bug unless the structure came from outside "
                    "the normal candidate or seed path."
                )
            row = self._row.get(key_from_src(src))
            if row is None:  # check_coverage / check_available should have caught it
                raise FeatureTableError(f"no feature row for {src}")
            out[i] = self._X[row]
        return out

    def identity_fields(self) -> dict:
        return dict(self._identity)

    def spec(self) -> dict:
        return {
            "name": "table",
            "kwargs": {"source": self.source, "feature_group": self.feature_group},
        }

    def inherited_status(self, atoms: Atoms) -> str | None:
        src = atoms.info.get(SRC_INFO_KEY)
        if src is None:
            return None
        return self._skipped.get(key_from_src(src))

    def check_available(self, atoms: Atoms) -> str | None:
        src = atoms.info.get(SRC_INFO_KEY)
        if src is None:
            return f"no {SRC_INFO_KEY} in atoms.info"
        row = self._row.get(key_from_src(src))
        if row is None:
            return "no row in the feature table"
        if not bool(self._valid[row]):
            return "row is marked invalid (all-NaN)"
        return None

    # -- pre-flight --------------------------------------------------------
    def check_coverage(self, frame_index: Any, *, on_invalid: str = "error") -> None:
        """Fail before any work if the table does not cover the frame set.

        Reporting every missing frame at once beats discovering them one batch
        at a time, and it happens before the first structure is read.
        """
        wanted: list[tuple[str, str]] = []
        basenames: dict[str, set[str]] = {}
        for ref in frame_index.refs:
            path = str(frame_index.file_path(ref.file_id))
            basenames.setdefault(os.path.basename(path), set()).add(path)
            wanted.append((provenance_key(path, ref.frame_index), f"{path}:{ref.frame_index}"))

        collisions = {b: sorted(p) for b, p in basenames.items() if len(p) > 1}
        if collisions:
            shown = list(collisions.items())[:3]
            raise FeatureTableError(
                "two or more input files share a basename, and the feature table "
                "records only basenames, so the join is ambiguous:\n  "
                + "\n  ".join(f"{b}: {paths}" for b, paths in shown)
                + "\nRename the inputs, or select from one directory at a time."
            )

        # A frame the source run itself dropped is not a gap -- its status is
        # known and will be carried across. Only an unexplained absence is.
        missing = [
            label
            for key, label in wanted
            if key not in self._row and key not in self._skipped
        ]
        if missing:
            raise FeatureTableError(
                f"{len(missing)} of {len(wanted)} frame(s) have no row in "
                f"{self._described}; e.g.\n  " + "\n  ".join(missing[:10]) + "\n"
                "Every candidate frame needs a feature vector -- dropping any "
                "would shift the selection onto the wrong structures. Re-extract "
                "features over these frames, or narrow the input with --stride / "
                "--max_frames so the two sets agree."
            )

        inherited = sum(1 for key, _ in wanted if key in self._skipped)
        if inherited:
            logger.info(
                "%d frame(s) were skipped by the run that produced these "
                "features; carrying those skips across", inherited
            )

        bad = [
            label
            for key, label in wanted
            if key in self._row and not bool(self._valid[self._row[key]])
        ]
        if bad:
            if on_invalid == "error":
                raise FeatureTableError(
                    f"{len(bad)} frame(s) have a row marked valid=False, meaning "
                    f"an all-NaN feature vector; e.g.\n  " + "\n  ".join(bad[:10]) + "\n"
                    "A NaN propagates through every distance and corrupts the "
                    "selection silently. Pass --on_invalid_features skip to drop "
                    "these frames instead."
                )
            logger.warning(
                "%d frame(s) have invalid (all-NaN) feature rows and will be "
                "skipped and recorded in frames.csv", len(bad)
            )

        logger.debug(
            "feature table covers all %d frame(s); %d unused row(s)",
            len(wanted),
            len(self._X) - (len(wanted) - inherited),
        )


def export_table(out_dir: str | Path, dest: str | Path) -> Path:
    """Write a finished run's descriptors in the interchange format.

    ``<dest>.npz`` gets ``features``/``valid`` and ``<dest>.index.csv`` gets
    ``row, structure_id, valid, source_file, frame_index`` -- the columns this
    module joins on, named as ``latent_features`` names them. The result is
    readable both by ``--input_features`` and by anything that reads
    ``latent_features`` output.

    Pointing ``--input_features`` straight at ``out_dir`` does the same job
    without this step; the export exists for handing features to other tools.
    """
    out_dir = Path(out_dir)
    dest = Path(str(dest)[:-4] if str(dest).endswith(".npz") else str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)

    X = np.load(out_dir / "descriptors.npy", mmap_mode="r")
    with open(out_dir / "frames.csv", newline="", encoding="utf-8") as fh:
        frames = [r for r in csv.DictReader(fh) if int(r["row"]) >= 0]
    frames.sort(key=lambda r: int(r["row"]))
    if len(frames) != len(X):
        raise FeatureTableError(
            f"{out_dir}: frames.csv accounts for {len(frames)} rows but "
            f"descriptors.npy has {len(X)}"
        )

    np.savez(
        Path(str(dest) + ".npz"),
        features=np.asarray(X),
        valid=np.ones(len(X), dtype=bool),
    )
    with open(str(dest) + ".index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "structure_id", "valid", "source_file", "frame_index"])
        for i, r in enumerate(frames):
            base = os.path.basename(r["source_file"])
            w.writerow([i, f"{base}:{r['frame_index']}", "True", base, r["frame_index"]])
    logger.info("exported %d feature row(s) -> %s.npz", len(X), dest)
    return Path(str(dest) + ".npz")
