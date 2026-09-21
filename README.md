# divsel

Pick a diverse subset of atomic structures by **k-means** or **farthest-point
sampling (FPS)** — on SOAP descriptors it generates for you, or on feature
vectors you already have (MLIP latents, anything). With streaming for datasets
that do not fit in memory, and a **seed set** so you can continue across
active-learning rounds without ever re-selecting what you already have.

```bash
pip install -e .

# generate SOAP and select
divsel select --traj 'md/*.traj' --n_select 200 --out round1

# or select on precomputed vectors, skipping SOAP entirely
divsel select --traj 'md/*.traj' --n_select 200 --out round1 \
              --input_features uma_features.npz
```

---

## The one idea

Everything in `divsel` is built around a single function:

```python
from divsel import select_diverse

result = select_diverse(X_candidates, n_select=200, X_seed=X_already_have,
                        method="kmeans")          # or "fps"
result.indices        # rows of X_candidates, in pick order
result.scores         # coverage-radius profile
```

`X_seed` means *"descriptors I already hold — stay away from these, and never
return them"*. Two things use it, and they are **the same code path**:

| caller | what the seed is |
|---|---|
| `--selected_images` | structures chosen in previous AL rounds — one file or several (permanent) |
| streaming carry-forward | structures picked from earlier batches of this run (provisional) |

They meet on one line in `divsel/streaming.py`:

```python
seed = vstack_nonempty(X_user_seed, pool_X)
picks = select_diverse(Xb, quota, X_seed=seed, method="fps")
```

---

## Worked example

`examples/run_demo.sh` runs all of this end to end. The short version:

```bash
# Round 1 --------------------------------------------------------------
divsel select --traj demo_a.traj demo_b.extxyz \
              --species C H O N --on_unknown_species skip \
              --n_select 12 --out round1 --plot --embed mds
```

```
round1/
  descriptors.npy    (N, d) float32 memmap
  frames.csv         row, source_file, frame_index, natoms, formula, cell_mode, status
  selected.csv       rank, row, source_file, frame_index, natoms, formula,
                     pick_stage, cluster, score
  selected.traj      ready for DFT *and* ready to be the next round's seed
  run_manifest.json  every parameter, version and count needed to reproduce this
  diagnostics.png  embedding_mds.png
```

```bash
# Round 2 — seeded by round 1 -------------------------------------------
divsel select --traj demo_a.traj demo_b.extxyz \
              --species C H O N --on_unknown_species skip \
              --n_select 12 --selected_images round1/selected.traj --out round2

# Round 3 — seeded by everything picked so far --------------------------
divsel select --traj demo_a.traj demo_b.extxyz \
              --species C H O N --on_unknown_species skip \
              --n_select 12 --out round3 \
              --selected_images round1/selected.traj round2/selected.traj
```

`--selected_images` takes as many previously-selected trajectories as you have
rounds; one descriptor cache entry is kept per file, so round 10 re-describes
only round 9.

`round2/selected.csv` shares **zero** `(source_file, frame_index)` pairs with
round 1, and round 2 validates the seed's descriptor parameters automatically —
`selected.traj` carries them in `atoms.info`, so mixing incomparable descriptors
raises `DescriptorMismatchError` instead of silently corrupting the selection.
With several seed files they are checked against *each other* as well, which is
what catches the round where `--r_cut` quietly changed.

```bash
# Rebuild a DFT trajectory from a CSV alone -----------------------------
divsel gather --csv round2/selected.csv --out_traj for_DFT_round2.traj
```

Note the two output conventions: `--out` is a **directory** (`round2/` above),
while `gather --out_traj` is a single **file**.

---

## Streaming

```bash
divsel select --traj 'huge/*.xyz' --n_select 500 --batch_size 2000 --out round1
```

Frames are described one batch at a time and written into a memmap; each
batch's picks go into a bounded pool that is carried forward as the seed for
the next batch; one final pass over the pool makes the actual selection.

**Guaranteed**

* peak memory is `O((batch_size + |pool| + n_seed) · d)`, not `O(N · d)`;
* no seed structure is ever re-selected;
* exactly `n_select` outputs whenever enough candidates exist;
* with a single batch (`--batch_size 0`, or a batch larger than the dataset)
  the result is the exact, un-pooled answer.

**Not guaranteed**

* Multi-batch is *not* the same as single-batch. FPS's k-center
  2-approximation does not survive batching, and there is no known bound for
  pooled streaming FPS. With `--shuffle` (on by default) and `--pool_alpha 4`
  it is empirically close — an empirical claim, not a theorem.
* `--method kmeans` across several batches selects from an FPS-thinned pool,
  which is density-distorted: FPS deliberately flattens density and k-means is
  a density method. Use `--pool_stage random` there; it is density-unbiased.

Order of operations is pinned: **enumerate → stride → max_frames → shuffle →
batch**. `--shuffle` permutes the *global* frame list across all input files;
shuffling within each file would leave every batch chemically homogeneous,
which is the problem it exists to solve.

---

## Non-periodic input

Molecules and clusters get a box automatically.

```bash
divsel select ... --cell_mode auto --vacuum 15.0
```

* `auto` (default) pads **only the directions that lack a cell**. A slab with
  `pbc=[True, True, False]` keeps its in-plane periodicity and is padded along
  the plane normal — not re-boxed, which would destroy it.
* Each frame's treatment is recorded in `frames.csv` as
  `cell_mode ∈ {native, padded, boxed}`.
* `2 · vacuum ≥ r_cut` is **required and checked — but only for structures that
  actually get a box.** `ase`'s centering makes the minimum image separation
  exactly `2 · vacuum`; below the cutoff a boxed structure would interact with
  its own images. Verified against dscribe 2.1.1: at vacuum 15 / r_cut 6
  periodic and non-periodic SOAP agree to `1.6e-11`; at vacuum 2 they differ by
  `466`.

  A fully periodic cell is never boxed, so the vacuum has no bearing on it and
  **a periodic-only run is never blocked by this**, whatever `--r_cut` and
  `--vacuum` you set. The check is applied per structure inside `ensure_cell`,
  the one place that knows whether a box is being built. A cheap probe of the
  first frame of each input file reports the common case before the scan
  starts.

  When a structure that needs a box violates it, the run aborts naming that
  frame. `--on_small_vacuum skip` instead records those frames in `frames.csv`
  as `small_vacuum:...` and carries on with the rest — at the cost of a
  selection that silently covers only part of your dataset.

---

## Reading the figures

`--plot` writes two things, and the difference matters.

**`diagnostics.png`** — exact quantities, no embedding, so they cannot mislead:

1. the pairwise-distance histogram, with `std/mean` printed. **Below ~0.1 your
   data is concentrated**: every 2-D method will draw a featureless blob,
   because the structures really are near-identical *under this descriptor*.
   The fix is then chemistry — a larger `r_cut`, a species-resolved descriptor,
   a different `sigma` — not plotting;
2. the PCA scree, so you know what fraction of the story a 2-D scatter shows;
3. the **coverage-radius curve** — distance to the nearest already-chosen
   structure at each pick. Monotone non-increasing; a plateau near zero means
   the space is saturated and further picks are near-duplicates. This is the
   signal to stop the campaign;
4. the **nearest-selected-distance CDF** against an equal-sized *random*
   selection. If the two curves separate, the selection is doing its job. This
   is the most direct one-figure justification of the method.

**`embedding_<method>.png`** — `--embed {pca,mds,tsne}`:

| method | preserves | use for |
|---|---|---|
| `pca` (default) | a true linear projection — distances are real, only truncated | the honest default |
| `mds` | fitted to reproduce the actual distance matrix | **"how far apart are these really?"** |
| `tsne` | local neighbourhoods only | finding clusters |

Every embedding is stamped with its own **Shepard correlation** and
**trustworthiness**, so distortion is a number rather than an assumption. Note
that in a t-SNE map cluster sizes, between-cluster distances and empty space are
all meaningless by construction; the figure says so in its caption.

---

---

## Using your own features

SOAP is one way to get vectors; it is not the only one. `--input_features`
skips descriptor generation entirely and selects on a table you supply:

```bash
divsel select --traj 'md/*.traj' --n_select 200 --out round1 \
              --input_features xxx.npz
```

It accepts either:

* **a `latent_features` output set** — `xxx.npz` (or the bare prefix) with its
  `.index.csv` beside it, as written by
  [`latent_spaces_as_features`](https://github.com/liping-ai4cat/latent_spaces_as_features);
* **a previous divsel output directory** — `--input_features round1`, which
  reuses that run's `descriptors.npy` instead of recomputing identical SOAP.
  On the demo set that turns 21.5 s into 4.9 s and selects the *same frames*.

The npz contract, verified against `latent_features/io.py:492`:

| key | shape | meaning |
|---|---|---|
| `features` | `(N_rows, D)` | one pooled vector per (structure, group) |
| `valid` | `(N_rows,)` bool | `False` means that row is all-NaN |

### Keep `<prefix>.metadata.json` with the `.npz`

Only `.npz` + `.index.csv` are *required*, but the metadata sidecar is what
tells divsel **which model produced these vectors**. It becomes the descriptor
identity: `backend`, `model_id`, `task_name`, `mode`, `layer_spans`.

Without it those fields are `None`, and two unrelated 128-dim embeddings hash
to the same `descriptor_id` — so the cross-round guard can no longer tell them
apart. divsel warns when the file is absent.

It also changes the id, which bites in a specific way. Run round 1 without the
sidecar, copy it in, then seed round 2 from round 1:

```
DescriptorMismatchError: seed descriptors are not comparable to this run.
  seed  (4f6faa3513c3058e):      model_id = None
  this  (6450113d20cb1f33):      model_id = 'uma-s-1p1'
```

That is the guard working, not a bug — divsel cannot know the two spaces are
the same. Regenerate the earlier round with the sidecar in place and both
rounds agree.

### Rows are matched to frames by provenance, never by position

`features[k]` is **not** input frame `k`. `latent_features` drops whole
structures under `--skip-errors` with no placeholder, and its `surface` /
`per_atom` modes emit several rows per structure; meanwhile divsel applies
`--stride`, `--max_frames` and `--shuffle` to its own frame list. So every row
is keyed `"<basename>:<frame_index>"` from the index CSV and looked up. A
positional join would look fine until either side dropped one frame, and would
then pair every later structure with the wrong vector.

Everything that could misalign raises instead:

| situation | what happens |
|---|---|
| a frame has no row | abort, naming up to 10 of them |
| two input files share a basename | abort — the table stores basenames only, so the join is genuinely ambiguous |
| more than one row per structure | abort listing the `group_label`s; pick one with `--feature_group` |
| a row has `valid=False` | abort (a NaN would silently corrupt every distance); `--on_invalid_features skip` records and drops it instead |
| the source run itself skipped a frame | that skip is inherited, with its original reason, so reuse reproduces the same frame set |

### Interaction with the rest

**`--selected_images` works unchanged.** Seed frames are resolved against the
same table via `atoms.info["divsel_src"]`, which `selected.traj` already
carries. No second flag, as long as your features cover the frames the seeds
came from; if one is missing the run aborts naming it. The descriptor-identity
guard still applies, so seeding a UMA-featured round with a SOAP-produced
`selected.traj` raises `DescriptorMismatchError` rather than mixing two
incomparable spaces.

**Streaming switches itself off.** Batching exists to bound the cost of
*generating* descriptors. With them precomputed it saves no memory — the table
is resident either way — and costs accuracy, since pooled multi-batch FPS has
no approximation bound while a single batch is exact. `--input_features`
therefore selects in one batch, and warns if you passed `--batch_size` anyway.
Note an `.npz` is a zip and cannot be memory-mapped, so a very large table
loads in full; `--stride` / `--max_frames` remain the way to subset it.

### A real example

[`test/select_rhea4sqs.py`](test/select_rhea4sqs.py) selects from 6,545
relaxed RHEA4SQS structures on their 128-dim UMA `omat` latents — **8.4 s**,
no SOAP. It is worth reading as a template: it preflights the table, runs the
selection, then *independently* re-derives the row↔frame mapping from the index
CSV and asserts every picked structure carries the vector its index entry
points at (worst |Δ| 5.3e-08, float32 rounding from the L2 normalization).

Round 2 seeded by round 1 picks 200 more with **zero** overlap.

**`--save_features out.npz`** writes this run's descriptors in the same
interchange format (`features`/`valid` + `.index.csv`), for handing to other
tools. For plain round-to-round reuse you do not need it — point
`--input_features` at the previous output directory.

---

## Options

```
divsel select --traj GLOB [GLOB ...] --n_select N [--out DIR]

  method     --method kmeans|fps          (default kmeans)
  features   --featurizer soap|uma|mace   (default soap; uma/mace not implemented)
             --input_features PATH        precomputed vectors: a latent_features
                                          .npz/prefix, or a previous --out dir
             --feature_group LABEL        when the table has >1 row per structure
             --on_invalid_features error|skip
             --save_features PATH.npz     export this run's descriptors
  seed       --selected_images FILE [FILE ...]   (trajectories selected in
                                                  previous rounds; must share
                                                  descriptor parameters)
  soap       --species auto|Z|SYM ...  --r_cut 6.0  --n_max 9  --l_max 3
             --sigma 1.0  --rbf gto  --soap_average outer|off
             --on_unknown_species error|skip
  cell       --cell_mode auto|native|box  --vacuum 15.0
             --box_shape orthorhombic|cubic
  sampling   --stride 1  --shuffle/--no_shuffle  --seed 0  --max_frames N  --no_scan
  streaming  --batch_size 10000  --pool_alpha 4  --pool_cap N
             --pool_stage fps|random  --n_procs <cpus>
  fps        --fps_init farthest_from_mean|random|index:J  --min_dist2 F
  kmeans     --kmeans_k auto|N  --kmeans_per_cluster 1  --kmeans_fill fps|none
             --kmeans_trim population|fps  --kmeans_n_init 10
             --minibatch_threshold 20000
  other      --normalize l2|none  --strict_determinism  --plot  --embed pca|mds|tsne
             --verbose quiet|normal|high
```

### Defaults worth knowing

| flag | default | why |
|---|---|---|
| `--method` | `kmeans` | matches the previous workflow. `fps` is the reproducible one — see below |
| `--normalize` | `l2` | stateless, so every batch and every round share one space; Euclidean distance becomes the normalized SOAP-kernel distance `d² = 2(1−cos)` ∈ [0,4] |
| `--soap_average` | `outer` | *exactly* the mean of per-atom SOAP vectors (verified maxdiff 0.0), without the per-structure temporary |
| `--shuffle` | on | the single most effective mitigation of batch-order bias |
| `--vacuum` | `15.0` | comfortably above `r_cut/2 = 3.0` |
| `--batch_size` | `2000` | ≈48 MB at `d = 5940`; tune to your RAM, not to the algorithm |
| `--kmeans_k` | `n_select` | *not* `n_select + n_seed`, which would grow without bound across rounds |

**Reproducibility, honestly.** FPS is deterministic and reproducible.
k-means is reproducible only within a fixed scikit-learn version and thread
count — k-means++ has changed between releases. `--strict_determinism` pins
BLAS to one thread inside the selection so argmax ties cannot flip.

**Species cost.** `d = (n_sp·n_max)(n_sp·n_max+1)/2·(l_max+1)`. Going from 6 to
8 species takes `d` from 5,940 to 10,404 and selection cost is linear in `d`.
Do not over-declare `--species` "just in case".

---

## Python API

```python
from divsel import run_selection, SelectConfig, SoapConfig, BoxConfig

manifest = run_selection(
    ["md/*.traj"],
    out_dir="round2",              # a DIRECTORY -- selected.traj, selected.csv,
                                   # frames.csv, descriptors.npy and
                                   # run_manifest.json are written inside it
    select_cfg=SelectConfig(n_select=200, method="fps"),
    soap_cfg=SoapConfig(species=(1, 6, 7, 8), r_cut=6.0),
    box_cfg=BoxConfig(vacuum=15.0),
    # everything picked so far -- one file per previous round
    selected_images=["round0/selected.traj", "round1/selected.traj"],
)
print(manifest["results"]["coverage_radius_min_d2"])
```

`out_dir` is a directory, not a filename: `"round2"` is right and
`"round2.traj"` would create a *directory* by that name. The one output path
that is a file is `divsel gather --out_traj for_DFT.traj`.

To select on features you already have, pass `input_features` instead of a
`soap_cfg` — everything else is the same:

```python
manifest = run_selection(
    ["RHEA4SQS_relaxed_omat.traj"],
    out_dir="round1",
    select_cfg=SelectConfig(n_select=200, method="fps"),
    input_features="RHEA4SQS_relaxed_omat",   # prefix, .npz, or a previous out_dir
)
```

`selected_images` accepts any number of previously-selected trajectories, and
every one of them must have been described with the same parameters. That is
checked, not assumed: each file's stamp is compared against the others and
against this run, so passing a round computed with a different `--r_cut` raises
`DescriptorMismatchError` naming both files. Descriptors are cached per file, so
adding one round costs one featurization rather than a recompute of all of them.

---

## Install

```bash
pip install -e .                # core
pip install -e '.[plot]'        # + matplotlib, for --plot
```

Requires `numpy`, `scipy`, `pandas`, `scikit-learn>=1.2`, `ase>=3.22`,
`dscribe>=2.0`, `threadpoolctl`.

> **Note:** `dscribe` is *not* installed in the `Python_vasp` conda env. Use an
> env that has it (e.g. `care_env_new_version-v2`) or `pip install dscribe`.

`uma` and `mace` appear in `--featurizer` but are **registered placeholders**
that raise `NotImplementedError`. The contract they must satisfy is in
`divsel/featurizers/base.py`; `divsel/featurizers/uma.py` documents how to wire
UMA up by delegating to the existing `latent_features` package. They are
optional extras on purpose — fairchem needs python 3.12 / numpy 2.x while
dscribe here runs on python 3.10 / numpy 1.26, so the two cannot be one hard
requirement.

---

## What changed from the original scripts

The previous two-script workflow lives untouched in `orginal_code_v1/` and
`orginal_code_v2/`. The substantive differences:

* **Frame provenance is real.** The old descriptor CSV was written from
  `imap_unordered` in *completion* order with the frame index discarded, and
  that index was a kept-structure counter shifted by every species skip — so a
  row number did not identify a source frame. Rows are now scattered by an
  authoritative slot, and `frames.csv` records `source_file` + `frame_index`
  for every attempted frame, skips included.
* **The `(round_tag, original_index)` bookkeeping is gone.** "Already selected"
  is now *descriptor vectors in a seed matrix*, so there is nothing to track.
* **A stale-cache bug is closed.** The old seed cache was keyed on filename, so
  changing `--soap-rcut` from 6.0 to 5.0 returned a cache hit on old-cutoff
  descriptors — invisible to a shape check, since `r_cut` does not change `d`.
  Caches are now keyed on a hash of every descriptor-defining parameter.
* **`StandardScaler` is gone.** It was fit on the stacked all-rounds matrix, so
  adding one seed file perturbed every candidate-to-candidate distance; and
  z-scoring upweights the near-zero-variance rare-species channels, making the
  metric noisiest where it is least informative. Selections will differ from
  old runs — intentionally.
* **`gather_images.py` is built in.** No hardcoded paths, and it handles
  selections spanning several source files.

### A k-means behaviour worth knowing

`--kmeans_k` defaults to `n_select`, and a cluster containing *any* seed
structure yields nothing. So when the seed set is comparable in size to
`n_select`, every cluster can end up covered and k-means contributes **zero**
picks — the entire budget then comes from the seeded-FPS top-up. That is by
design (the alternative, `k = n_select + n_seed`, grows without bound across
rounds), it is logged as a warning, and `run_manifest.json` always records the
split as `n_from_kmeans` / `n_from_fps_topup`. Raise `--kmeans_k` if you want
k-means to keep a say.

---

## Citing this work

If `divsel` contributes to work you publish, please cite:

```bibtex
@ARTICLE{Liu2026-MLIP,
  title        = {Molecular origins of {CO2} capture behavior in Amine-appended
                  nanoporous frameworks},
  author       = {Liu, Liping and Zhou, Zihui and Daglar, Hilal and Siepmann,
                  Ilja and Yaghi, Omar M and Gagliardi, Laura},
  journaltitle = {ChemRxiv},
  date         = {2026-08-10},
  urldate      = {2026-08-13},
  language     = {en}
}
```
