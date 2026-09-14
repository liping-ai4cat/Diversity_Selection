#!/usr/bin/env python3
"""Build a small, deliberately heterogeneous demo dataset.

Three structurally distinct families, so a good embedding must show three
separated groups and the diagnostics must report a well-spread distance
distribution:

* **bulk**      rattled diamond-carbon supercell -- a genuine 3-D periodic cell
                (exercises ``cell_mode=native``);
* **slab**      a rank-2 carbon slab, ``pbc=[True, True, False]`` -- must be
                padded along z ONLY, keeping its in-plane periodicity
                (exercises ``cell_mode=padded``);
* **molecules** ethanol and ammonia with no cell at all -- must be boxed
                (exercises ``cell_mode=boxed``).

Written to two files in two formats, so the demo also covers:

* the .traj random-access path and the (ext)xyz byte-offset scan path;
* a species that appears **only in the last file** (N, in the ammonia frames),
  which the original v1 auto-detection -- first image of at most 100 files --
  would have missed;
* one frame containing a stray Fe atom, to exercise the species screen.
"""

from __future__ import annotations

import numpy as np
from ase.build import bulk, fcc111, molecule
from ase.io import write


def main(out_a: str = "demo_a.traj", out_b: str = "demo_b.extxyz", seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    frames_a, frames_b = [], []

    base_bulk = bulk("C", "diamond", a=3.567) * (2, 2, 2)
    for _ in range(60):
        atoms = base_bulk.copy()
        atoms.rattle(stdev=0.08, seed=int(rng.integers(1 << 30)))
        frames_a.append(atoms)

    base_slab = fcc111("C", size=(2, 2, 3), a=3.567, vacuum=None)
    assert base_slab.cell.rank == 2 and not base_slab.pbc[2], "expected a rank-2 slab"
    for _ in range(60):
        atoms = base_slab.copy()
        atoms.rattle(stdev=0.08, seed=int(rng.integers(1 << 30)))
        frames_a.append(atoms)

    base_mol = molecule("CH3CH2OH")
    base_mol.set_cell(None)
    base_mol.set_pbc(False)
    for _ in range(50):
        atoms = base_mol.copy()
        atoms.rattle(stdev=0.06, seed=int(rng.integers(1 << 30)))
        frames_b.append(atoms)

    # Nitrogen appears ONLY here, in the last file.
    base_nh3 = molecule("NH3")
    base_nh3.set_cell(None)
    base_nh3.set_pbc(False)
    for _ in range(40):
        atoms = base_nh3.copy()
        atoms.rattle(stdev=0.06, seed=int(rng.integers(1 << 30)))
        frames_b.append(atoms)

    # One frame with an element outside a C/H/O/N species list.
    stray = base_mol.copy()
    stray.symbols[0] = "Fe"
    frames_b.append(stray)

    write(out_a, frames_a)
    write(out_b, frames_b)
    print(f"wrote {len(frames_a)} frames -> {out_a}  (bulk + slab, periodic / rank-2)")
    print(f"wrote {len(frames_b)} frames -> {out_b}  (molecules, no cell; N only here; 1 Fe frame)")


if __name__ == "__main__":
    main()
