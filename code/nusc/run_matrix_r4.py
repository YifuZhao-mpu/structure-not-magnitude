#!/usr/bin/env python3
"""R-4 matrix: 10 nuScenes scenes x 3 conditions x 3 seeds. Resumable."""
import os, sys, glob, subprocess, itertools, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))   # repo root, not a fixed path
OUT = f"{ROOT}/outputs/nusc_r4"
SCENES = [os.path.basename(p) for p in sorted(glob.glob(f"{ROOT}/data/nusc_r4/scene-*"))]
CONDS = ["native", "native_naive", "native_x2"]
SEEDS = [0, 1, 2]

def main():
    gpu, nsh, sh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    jobs = [j for k, j in enumerate(itertools.product(SCENES, CONDS, SEEDS)) if k % nsh == sh]
    done = skip = fail = 0; t0 = time.time()
    for sc, cond, sd in jobs:
        out = f"{OUT}/{sc}_{cond}_s{sd}.json"
        if os.path.exists(out) and os.path.getsize(out) > 0:
            skip += 1; continue
        r = subprocess.run(["python", f"{ROOT}/code/nusc/train_nusc_r4.py",
                            "--data", f"data/nusc_r4/{sc}", "--cond", cond,
                            "--seed", str(sd), "--iters", "8000", "--out", out],
                           cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode:
            fail += 1; print(f"[{sh}] FAIL {sc} {cond} s{sd}\n{r.stderr.decode()[-500:]}", flush=True)
        else:
            done += 1
    print(f"[{sh}] COMPLETE done={done} skip={skip} fail={fail} "
          f"{(time.time()-t0)/60:.1f}min", flush=True)

if __name__ == "__main__":
    main()
