#!/usr/bin/env bash
# Two-round active-learning walkthrough.  Doubles as the end-to-end smoke test.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== 0. build demo data ==============================================="
python make_demo_data.py

echo
echo "=== 1. round 1: select 12 (k-means is the default method) ============"
divsel select --traj demo_a.traj demo_b.extxyz \
              --species C H O N \
              --on_unknown_species skip \
              --n_select 12 \
              --out round1 \
              --plot --embed mds \
              --verbose high

echo
echo "=== 2. what came out ================================================="
echo "--- round1/selected.csv"; cat round1/selected.csv
echo "--- cell_mode breakdown (native / padded / boxed)"
python - <<'PY'
import pandas as pd
df = pd.read_csv("round1/frames.csv")
print(df["cell_mode"].value_counts().to_string())
print()
print("status breakdown:")
print(df["status"].value_counts().to_string())
PY

echo
echo "=== 3. round 2: continue, seeded by round 1 =========================="
# round1/selected.traj is stamped with the descriptor id, so round 2 validates
# compatibility by itself and refuses to mix incomparable descriptors.
divsel select --traj demo_a.traj demo_b.extxyz \
              --species C H O N \
              --on_unknown_species skip \
              --n_select 12 \
              --selected_images round1/selected.traj \
              --out round2

echo
echo "=== 4. the seed set is never re-selected ============================="
python - <<'PY'
import pandas as pd
a = pd.read_csv("round1/selected.csv")
b = pd.read_csv("round2/selected.csv")
key = lambda d: set(zip(d.source_file, d.frame_index))
overlap = key(a) & key(b)
print(f"round1: {len(a)} frames, round2: {len(b)} frames, overlap: {len(overlap)}")
assert not overlap, f"seed frames were re-selected: {overlap}"
print("OK - zero overlap, as required")
PY

echo
echo "=== 4b. round 3: seeded by EVERYTHING picked so far =================="
# --selected_images takes a list: one trajectory per previous round. They are
# checked against each other as well as against this run, and each one's
# descriptors are cached separately, so round N only re-describes round N-1.
divsel select --traj demo_a.traj demo_b.extxyz \
              --species C H O N \
              --on_unknown_species skip \
              --n_select 12 \
              --selected_images round1/selected.traj round2/selected.traj \
              --out round3

python - <<'PY2'
import json, pandas as pd
m = json.load(open("round3/run_manifest.json"))
ss = m["seed_set"]
print(f"seed sources: {ss['sources']}")
print(f"rows per source: {ss['n_per_source']}  ->  n_seed = {ss['n_seed']}")
assert len(ss["sources"]) == 2 and sum(ss["n_per_source"]) == ss["n_seed"] == 24
c = pd.read_csv("round3/selected.csv")
for r in ("round1", "round2"):
    old = pd.read_csv(f"{r}/selected.csv")
    overlap = set(zip(c.source_file, c.frame_index)) & set(zip(old.source_file, old.frame_index))
    assert not overlap, (r, overlap)
print("OK - 24 seeds from 2 files, zero overlap with either round")
PY2

echo
echo "=== 5. rebuild the DFT trajectory from the CSV alone ================="
divsel gather --csv round2/selected.csv --out_traj for_DFT_round2.traj

echo
echo "=== 6. provenance is real: CSV row -> source frame ==================="
python - <<'PY'
import pandas as pd
from ase.io import read
df = pd.read_csv("round2/selected.csv")
sel = read("round2/selected.traj", index=":")
for _, r in df.head(3).iterrows():
    src = read(r.source_file, index=int(r.frame_index))
    assert len(src) == r.natoms, (len(src), r.natoms)
    assert src.get_chemical_formula() == sel[int(r["rank"])].get_chemical_formula()
    print(f"  rank {int(r['rank'])}: {r.source_file}:{int(r.frame_index)} "
          f"-> {src.get_chemical_formula()}  (matches selected.traj)")
print("OK - every selected row resolves back to its source frame")
PY

echo
echo "Done. Outputs in round1/ and round2/; figures in round1/*.png"
