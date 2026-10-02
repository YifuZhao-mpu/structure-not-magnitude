# Structure, Not Just Magnitude — code and raw run records

Code and raw run records for re-deriving the results reported in *Structure, Not Just Magnitude, Drives the Damage from Misprojected LiDAR Depth Supervision in 3D Gaussian
Splatting*, and for re-running its experiments from scratch.

## What is here

| Path | Contents |
|---|---|
| `code/synthetic/` | scene generator, the two trainers, the depth-sliced opacity probe, three adjudicators, the numerical verifier, the regression suite |
| `code/nusc/` | nuScenes corroboration: scene builder, trainer, adjudicator |
| `outputs/multiscene_v3/` | 300 analytic run records (10 scenes × 10 conditions × 3 seeds) |
| `outputs/nusc_r4/` | 90 nuScenes run records |
| `outputs/probe_M/` | 90 mechanism-probe run records **and their per-view opacity maps** |
| `data/nusc_index/` | the nuScenes scene index: file paths, calibrations, poses, box annotations |
| `outputs/multiscene/` | 270 run records of the first pipeline version, which Table 8 compares against; only `verify_manuscript_numbers.py` reads them |
| `data/multiscene/sc*/meta.json` | the recorded parameters of the first version's ten scenes, which `test_scene_regression.py` checks against the generator |

## Condition names

The paper names its label conditions in words; the run records, file names and scripts
use short keys.

| In the paper | In the records |
|---|---|
| *correct* | `correct` |
| *misprojected* | `wrong` |
| *RGB-only* | `none` |
| *shuffled-space* | `shuffled_space` |
| *resampled-magnitude* | `resampled_mag` |
| *sign-flipped* | `sign_flipped` |
| *native* | `native` |
| *naive transfer* | `native_naive` |
| *oracle transfer* | `native_oracle` |
| *native ×2* | `native_x2` |

*RGB-only* is the one condition renamed rather than transliterated, because its key
overstates what it controls for: that condition is still initialised from LiDAR returns.

## What is deliberately not here

- **`data/multiscene_v3/`** — the analytic scene arrays. They are not shipped because
  they rebuild byte-for-byte from their seeds; `test_scene_regression.py` asserts this
  (set `ARS_REGEN_TEST=1` for the full 120-array comparison).
- **`data/nusc_r4/`** — the nuScenes-derived scenes. They are not ours to redistribute.
  `code/nusc/gen_nusc_r4.py` rebuilds them from the released index against an official
  nuScenes installation, subject to that dataset's own licence.
- **The pre-analysis plans, the source-review audit records and the literature-gate
  record.** As the paper's Data-availability statement says, they are not part of the
  public release: what they specify is implemented in `adjudicate.py`, `adjudicate_M.py`
  and `code/nusc/adjudicate_r4.py`, and what they report is stated in §4.5 and §5.3 of
  the paper.
- **The manuscript source.** `verify_manuscript_numbers.py` checks the numbers in the
  manuscript's text against these run records, so it reads that text from
  `paper-stage/manuscript.md`; without it the script stops and says so.
- **The figures.** `make_figures.py` and `make_figure_M.py` re-derive the seven data
  figures from the run records — Figures 2–8 of the paper, written as `fig1_…` to
  `fig7_…`. Figure 1 is a schematic of the study design.

## Reproducing

```bash
# 0. environment: gsplat 1.6.0, PyTorch 2.8, CUDA 12.8
# 1. rebuild the ten analytic scenes from their seeds
bash code/synthetic/gen_multiscene.sh 10
# 2. the 300-run matrix (two GPUs, two shards)
python code/synthetic/run_matrix.py 0 2 0 &  python code/synthetic/run_matrix.py 1 2 1
# 3. apply the frozen criteria -- the verdict is computed, not asserted
python code/synthetic/adjudicate.py
# 4. the mechanism probe (90 runs) and its frozen criteria
python code/synthetic/run_probe_M.py 0 2 0 &  python code/synthetic/run_probe_M.py 1 2 1
python code/synthetic/adjudicate_M.py
# 5. check the manuscript's numbers against the raw records (needs the manuscript
#    source, see above, and step 1: some checks read the scenes' own primitive ids and depths)
python code/synthetic/verify_manuscript_numbers.py     # expects 269 / 269, 0 mismatches
python code/synthetic/test_scene_regression.py         # 8 tests

# step 1 writes data/multiscene_v3/, which every later step reads. adjudicate.py and
# code/nusc/adjudicate_r4.py need only the shipped run records; adjudicate_M.py and
# step 5 also read the scenes' geometry, so they need step 1 (CPU, seconds per scene).
# The training runs of steps 2 and 4 are the GPU work and reproduce the shipped records.
# The nuScenes path additionally needs NUSCENES_ROOT set to an official installation:
# `NUSCENES_ROOT=/path/to/nuscenes python code/nusc/gen_nusc_r4.py`
```

## Versions

- **v1.0.1** (2026-10-02) repairs the reproduction path. In v1.0.0, `gen_multiscene.sh`
  wrote the scenes to `data/multiscene/` while every later step reads
  `data/multiscene_v3/`, so step 2 failed on a fresh checkout; and the verifier's Table 8
  checks and one regression test read files the release did not carry. v1.0.1 fixes the
  path, adds the first version's run records and scene metadata, gives the verifier and
  the two figure scripts the paper's condition names, and brings this README and
  `CITATION.cff` in line with the paper's Data-availability statement. The run records
  of v1.0.0 are unchanged. The public path — scene regeneration, the three adjudicators,
  the full regression suite and the verifier — was re-run end to end on the assembled
  release: the regenerated scenes are byte-identical to the originals and the verifier
  reports 269 of 269.
- **v1.0.0** (2026-09-24) is the version the paper's results were produced from,
  doi:[10.5281/zenodo.22930375](https://doi.org/10.5281/zenodo.22930375).
  doi:[10.5281/zenodo.22930374](https://doi.org/10.5281/zenodo.22930374) resolves to the
  latest version.

## A note on the plans

Three pre-analysis plans were frozen **before** the data they judge existed, and each is
executed by one of the `adjudicate*.py` scripts here rather than restated in prose. They
rejected five of the seven predictions this study registered, including the mechanism the
paper originally asserted for its own main result. Three independent source-level reviews
found fifteen defects in this pipeline, none of which we found ourselves.

## Licence and what it covers

The code in `code/` and the run records in `outputs/` are released under the **MIT
Licence** (see `LICENSE`).

`data/nusc_index/` is **not** covered by that grant. It is derived from the nuScenes
dataset — file paths, calibrations, poses and box annotations — and remains subject to
the nuScenes terms, which are non-commercial. Use it accordingly, and cite nuScenes as
well as this work.
