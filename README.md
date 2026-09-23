# Structure, Not Magnitude — code, run records and frozen plans

Everything needed to re-derive every number in the paper from raw run records, and to
re-run the experiments from scratch.

## What is here

| Path | Contents |
|---|---|
| `code/synthetic/` | scene generator, the two trainers, the depth-sliced opacity probe, three adjudicators, the numerical verifier, the regression suite |
| `code/nusc/` | nuScenes corroboration: scene builder, trainer, adjudicator |
| `outputs/multiscene_v3/` | 300 analytic run records (10 scenes × 10 conditions × 3 seeds) |
| `outputs/nusc_r4/` | 90 nuScenes run records |
| `outputs/probe_M/` | 90 mechanism-probe run records **and their per-view opacity maps** |
| `data/nusc_index/` | the nuScenes scene index: file paths, calibrations, poses, box annotations |


The paper-authoring directory is not part of this release. Four items can be added with
flags to `make_release.sh`:

- `--with-plans` — the three frozen pre-analysis plans, the three source-review audits,
  the three adjudication records, the integrity records and the literature gate. **The
  paper's Data-availability statement says these are available here**, so either ship
  them or amend that statement.
- `--with-manuscript` — the manuscript text, which `verify_manuscript_numbers.py` needs.
- `--with-process-record` — the internal process record.
- `--with-simulated-review` — the drafting-time **simulated** review documents. These
  carry a real risk of being mistaken for genuine peer review of this paper, so they ship
  with a notice saying they are not.

## What is deliberately not here

- **`data/multiscene_v3/`** — the analytic scene arrays. They are not shipped because
  they rebuild byte-for-byte from their seeds; `test_scene_regression.py` asserts this
  (set `ARS_REGEN_TEST=1` for the full 120-array comparison).
- **`data/nusc_r4/`** — the nuScenes-derived scenes. They are not ours to redistribute.
  `code/nusc/gen_nusc_r4.py` rebuilds them from the released index against an official
  nuScenes installation, subject to that dataset's own licence.

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
# 5. check every number in the manuscript against the raw records
#    (step 1 must have run: some checks read the scenes' own primitive ids and depths)
python code/synthetic/verify_manuscript_numbers.py     # expects 269 / 269, 0 mismatches
python code/synthetic/test_scene_regression.py         # 8 tests

# steps 3 and 5 need only the shipped run records; steps 2 and 4 are the GPU work and
# reproduce them. The nuScenes path additionally needs NUSCENES_ROOT set to an official
# installation: `NUSCENES_ROOT=/path/to/nuscenes python code/nusc/gen_nusc_r4.py`
```

`verify_manuscript_numbers.py` compares the **manuscript text** against these run
records, so it needs `paper-stage/manuscript.md`. Build with `--with-manuscript`, or drop
the published text in at that path. Figures are not shipped either; `make_figures.py` and
`make_figure_M.py` re-derive all seven from the run records.

## A note on the plans

Three pre-analysis plans were frozen **before** the data they judge existed, and each is
executed by one of the `adjudicate*.py` scripts here rather than restated in prose. They
rejected five of the seven predictions this study registered, including the mechanism the
paper originally asserted for its own main result. Three independent source-level reviews
found fifteen defects in this pipeline, none of which we found ourselves. The plans and
the audit records are shipped with `--with-plans`.

## Licence and what it covers

The code in `code/` and the run records in `outputs/` are released under the **MIT
Licence** (see `LICENSE`).

`data/nusc_index/` is **not** covered by that grant. It is derived from the nuScenes
dataset — file paths, calibrations, poses and box annotations — and remains subject to
the nuScenes terms, which are non-commercial. Use it accordingly, and cite nuScenes as
well as this work.
