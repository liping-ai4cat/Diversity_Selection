"""SOAP descriptors via dscribe.

Reproduces the original pipeline exactly.  The one substitution is
``average="outer"`` in place of a manual ``soap.create(atoms).mean(axis=0)``:
verified against dscribe 2.1.1 to be bit-identical (maxdiff 0.0) while avoiding
an ``(n_atoms, 5940)`` float64 temporary per structure.

``average="inner"`` is NOT the same quantity -- it averages the density
coefficients before forming the power spectrum, and differs from the mean of
per-atom vectors by maxdiff 28.2 on a 7-atom test structure.  :class:`SoapConfig`
rejects it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols

from ..config import SoapConfig, soap_feature_dim
from ..errors import SpeciesError
from ..log import get_logger
from .base import Featurizer

logger = get_logger(__name__)

__all__ = ["SoapFeaturizer"]


class SoapFeaturizer(Featurizer):
    name = "soap"
    parallel_mode = "process_pool"

    def __init__(self, cfg: SoapConfig | None = None, **kwargs) -> None:
        if cfg is None:
            cfg = SoapConfig(**kwargs)
        self.cfg = cfg
        self._soap = None  # built lazily; dscribe import is not free

    # -- lifecycle ---------------------------------------------------------
    def prepare(self, frame_index) -> None:
        """Resolve the species set, which fixes the descriptor length."""
        if self.cfg.species is None:
            observed = getattr(frame_index, "species", None)
            if not observed:
                raise SpeciesError(
                    "--species auto needs the scan pass to observe atomic numbers. "
                    "Either drop --no_scan or pass --species explicitly."
                )
            self.cfg = self.cfg.with_species(observed)
            logger.debug(
                "species resolved from scan: %s",
                [f"{chemical_symbols[z]}({z})" for z in self.cfg.species],
            )
        self._allowed = frozenset(self.cfg.species)
        logger.debug(
            "SOAP d = %d  (n_species=%d, n_max=%d, l_max=%d)",
            self.feature_dim,
            len(self.cfg.species),
            self.cfg.n_max,
            self.cfg.l_max,
        )

    @property
    def feature_dim(self) -> int:
        return soap_feature_dim(len(self.cfg.species), self.cfg.n_max, self.cfg.l_max)

    def _descriptor(self):
        if self._soap is None:
            from dscribe.descriptors import SOAP

            self._soap = SOAP(
                species=sorted(set(self.cfg.species)),
                r_cut=self.cfg.r_cut,
                n_max=self.cfg.n_max,
                l_max=self.cfg.l_max,
                sigma=self.cfg.sigma,
                rbf=self.cfg.rbf,
                periodic=self.cfg.periodic,
                sparse=False,
                **({} if self.cfg.average == "off" else {"average": "outer"}),
            )
        return self._soap

    @property
    def neighbor_cutoff(self) -> float | None:
        """``r_cut``, but only when the descriptor is periodic.

        With ``periodic=False`` dscribe ignores the cell entirely, so a box is
        cosmetic and no amount of vacuum is required -- return None so the
        boxing guard does not fire.
        """
        return float(self.cfg.r_cut) if self.cfg.periodic else None

    # -- work --------------------------------------------------------------
    def check_supported(self, atoms: Atoms) -> str | None:
        extra = {int(z) for z in atoms.get_atomic_numbers()} - self._allowed
        if extra:
            symbols = ",".join(f"{chemical_symbols[z]}({z})" for z in sorted(extra))
            return f"unsupported_species:{symbols}"
        return None

    def featurize(self, atoms_list: Sequence[Atoms]) -> np.ndarray:
        soap = self._descriptor()
        d = self.feature_dim
        out = np.empty((len(atoms_list), d), dtype=np.float32)
        for i, atoms in enumerate(atoms_list):
            vec = soap.create(atoms)
            vec = np.asarray(vec)
            if vec.ndim == 2:  # average="off": mean over atoms, as the original did
                vec = vec.mean(axis=0)
            out[i] = vec.astype(np.float32, copy=False)
        return out

    # -- identity ----------------------------------------------------------
    def identity_fields(self) -> dict:
        """Read sigma/rbf off the constructed object, not off argparse defaults.

        dscribe could change a default between versions; the identity must
        reflect what was actually computed.
        """
        soap = self._descriptor()
        sigma = getattr(soap, "sigma", None)
        if sigma is None:
            sigma = getattr(soap, "_sigma", self.cfg.sigma)
        rbf = getattr(soap, "rbf", None)
        if rbf is None:
            rbf = getattr(soap, "_rbf", self.cfg.rbf)
        average = getattr(soap, "average", self.cfg.average)

        import dscribe

        version = getattr(dscribe, "__version__", None)
        if version is None:  # dscribe 2.1.1 does not expose __version__
            try:
                from importlib.metadata import version as _pkg_version

                version = _pkg_version("dscribe")
            except Exception:  # pragma: no cover - best effort only
                version = "unknown"
        # Hash only major.minor: a patch bump should not invalidate every
        # descriptor you own, but it is still recorded in the manifest.
        major_minor = ".".join(str(version).split(".")[:2])

        return {
            "species_z": list(self.cfg.species),
            "r_cut": float(self.cfg.r_cut),
            "n_max": int(self.cfg.n_max),
            "l_max": int(self.cfg.l_max),
            "sigma": float(sigma),
            "rbf": str(rbf),
            "periodic": bool(self.cfg.periodic),
            "average": str(average),
            "n_features": int(self.feature_dim),
            "dscribe": major_minor,
        }

    def spec(self) -> dict:
        from dataclasses import asdict

        return {"name": self.name, "kwargs": {"cfg_dict": asdict(self.cfg)}}

    @classmethod
    def from_spec_kwargs(cls, cfg_dict: dict) -> "SoapFeaturizer":
        cfg_dict = dict(cfg_dict)
        if cfg_dict.get("species") is not None:
            cfg_dict["species"] = tuple(cfg_dict["species"])
        obj = cls(SoapConfig(**cfg_dict))
        obj._allowed = frozenset(obj.cfg.species or ())
        return obj
