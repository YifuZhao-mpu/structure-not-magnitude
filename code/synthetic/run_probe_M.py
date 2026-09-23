#!/usr/bin/env python3
"""
§5.1 mechanism measurement run matrix (PRE_ANALYSIS_PLAN_M.md).

Three label conditions only -- `correct` is the reference the depletion maps are
differenced against, `wrong` is the coherent contamination, `shuffled_space` the
magnitude-matched incoherent control. Everything else is identical to the v3 matrix
(same scenes, same seeds, same 8000 iters), so these runs are drawn from the same
distribution as the ones the paper already reports.

Writes to outputs/probe_M/ -- the v3 results are not touched.
"""
import os, sys, subprocess, itertools, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))   # repo root, not a fixed path
OUT = f"{ROOT}/outputs/probe_M"
SCENES = [f"sc{k:02d}" for k in range(10)]
CONDS = ["correct", "shuffled_space", "wrong"]
SEEDS = [0, 1, 2]


def main():
    gpu, nshard, shard = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    jobs = list(itertools.product(SCENES, CONDS, SEEDS))
    todo = [j for i, j in enumerate(jobs) if i % nshard == shard]
    done = skipped = failed = 0
    t0 = time.time()
    for sc, cond, sd in todo:
        tag = f"{sc}_{cond}_s{sd}"
        out = f"{OUT}/{tag}.json"
        if os.path.exists(out) and os.path.getsize(out) > 0:
            skipped += 1
            continue
        r = subprocess.run(
            [sys.executable, f"{ROOT}/code/synthetic/train_gateA.py",
             "--data", f"data/multiscene_v3/{sc}", "--cond", cond,
             "--seed", str(sd), "--iters", "8000",
             "--probe", f"{OUT}/maps/{tag}", "--out", out],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode != 0:
            failed += 1
            print(f"[s{shard}] FAIL {tag}\n{r.stderr.decode()[-500:]}", flush=True)
        else:
            done += 1
        n = done + skipped
        if n % 5 == 0:
            el = time.time() - t0
            print(f"[s{shard}] {n}/{len(todo)} fail={failed} {el/60:.1f}min "
                  f"eta={(len(todo)-n)*el/max(done,1)/60:.0f}min", flush=True)
    print(f"[s{shard}] COMPLETE done={done} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
