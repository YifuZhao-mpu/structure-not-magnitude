#!/usr/bin/env python3
"""
Multi-scene run matrix. SCENE is the unit of replication -- that is the whole point:
with one hand-built scene, training seeds are noise replicates and a paired t over
them has the wrong statistical unit.

Resumable: an existing non-empty output json is skipped, so the job can be re-launched
after an interruption without redoing work.
"""
import os, sys, json, subprocess, itertools, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))   # repo root, not a fixed path
OUT = f"{ROOT}/outputs/multiscene_v3"
SCENES = [f"sc{k:02d}" for k in range(10)]
SEEDS = [0, 1, 2]
GATE_A = ["none", "correct", "wrong", "shuffled_space", "resampled_mag", "sign_flipped"]
# native_x2 doubles the native loss weight: without it, `native` carries total depth
# weight 0.5 while both transfer conditions carry 1.0, so a transfer advantage could
# be a supervision-budget effect rather than a label-correctness effect.
GATE_B = ["native", "native_naive", "native_oracle", "native_x2"]


def jobs():
    for sc, cond, sd in itertools.product(SCENES, GATE_A, SEEDS):
        yield ("A", sc, cond, sd)
    for sc, cond, sd in itertools.product(SCENES, GATE_B, SEEDS):
        yield ("B", sc, cond, sd)


def main():
    gpu, nshard, shard = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    todo = [j for i, j in enumerate(jobs()) if i % nshard == shard]
    done = skipped = failed = 0
    t0 = time.time()
    for kind, sc, cond, sd in todo:
        out = f"{OUT}/{sc}_{cond}_s{sd}.json"
        if os.path.exists(out) and os.path.getsize(out) > 0:
            skipped += 1
            continue
        script = "train_gateA.py" if kind == "A" else "train_gateB.py"
        cmd = ["python", f"{ROOT}/code/synthetic/{script}",
               "--data", f"data/multiscene_v3/{sc}", "--cond", cond,
               "--seed", str(sd), "--iters", "8000", "--out", out]
        r = subprocess.run(cmd, cwd=ROOT, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode != 0:
            failed += 1
            print(f"[shard{shard}] FAIL {sc} {cond} s{sd}\n"
                  f"{r.stderr.decode()[-600:]}", flush=True)
        else:
            done += 1
        if (done + skipped) % 20 == 0:
            el = time.time() - t0
            rate = el / max(done, 1)
            print(f"[shard{shard}] {done+skipped}/{len(todo)} done={done} skip={skipped} "
                  f"fail={failed} {el/60:.1f}min eta={(len(todo)-done-skipped)*rate/60:.0f}min",
                  flush=True)
    print(f"[shard{shard}] COMPLETE done={done} skipped={skipped} failed={failed} "
          f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
