#!/usr/bin/env python3
"""
Scene-level analysis of the multi-scene matrix.

The unit of replication is the SCENE. Within each (scene, condition) cell the training
seeds are averaged first, because repeated runs at a FIXED seed were measured to differ
(gsplat's atomicAdd backward is accumulation-order sensitive), so a seed is a noise
replicate, not an independent condition. Averaging the cell then pairing across scenes
puts the test at the level the claim is actually about.

Also reports the variance decomposition -- within-cell (run-to-run) sd vs between-scene
sd -- because a contrast smaller than run-to-run noise is not a finding.
"""
import os
import json, glob, os, math, collections, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))   # repo root, not a fixed path
METRICS = ["depth_mae", "depth_mae_silhouette", "mae_wall", "mae_pole", "mae_car", "psnr"]
GATE_A = ["none", "correct", "shuffled_space", "sign_flipped", "resampled_mag", "wrong"]
GATE_B = ["native", "native_naive", "native_oracle", "native_x2"]


def load():
    cells = collections.defaultdict(list)          # (scene, cond) -> [run dicts]
    for f in glob.glob(f"{ROOT}/outputs/multiscene_v3/*.json"):
        b = os.path.basename(f)[:-5]
        sc = b.split("_")[0]
        seed = int(b.rsplit("_s", 1)[1])
        cond = b[len(sc) + 1: b.rindex("_s")]
        try:
            cells[(sc, cond)].append(json.load(open(f)))
        except Exception:
            pass
    return cells


def sd(x):
    return st.stdev(x) if len(x) > 1 else float("nan")


def paired(cellmean, conds, a, b, metric):
    """paired across SCENES; each side is that scene's seed-mean"""
    scs = sorted({s for (s, c) in cellmean if c == a} & {s for (s, c) in cellmean if c == b})
    d = [cellmean[(s, a)][metric] - cellmean[(s, b)][metric]
         for s in scs
         if cellmean[(s, a)].get(metric) is not None and cellmean[(s, b)].get(metric) is not None]
    if len(d) < 2:
        return None
    m, s_ = st.mean(d), sd(d)
    se = s_ / math.sqrt(len(d))
    t = m / se if se else float("inf")
    # 95% CI, normal approx (n=10 -> t_.975,9 = 2.262)
    crit = 2.262 if len(d) == 10 else 2.0
    return dict(n=len(d), delta=m, t=t, dz=m / s_ if s_ else float("inf"),
                lo=m - crit * se, hi=m + crit * se)


def preflight(cells):
    """Refuse to analyse an incomplete or corrupted matrix. A missing cell silently
    unbalances the paired test; a None stratified metric silently drops a scene from
    one contrast but not another."""
    scenes = [f"sc{k:02d}" for k in range(10)]
    conds = GATE_A + GATE_B
    missing, short, bad = [], [], []
    for sc in scenes:
        for c in conds:
            runs = cells.get((sc, c), [])
            if not runs:
                missing.append(f"{sc}/{c}")
            elif len(runs) != 3:
                short.append(f"{sc}/{c}(n={len(runs)})")
            for r in runs:
                for k in METRICS:
                    v = r.get(k)
                    if v is None or not (isinstance(v, (int, float)) and math.isfinite(v)):
                        bad.append(f"{sc}/{c}/s{r.get('seed')}:{k}={v}")
    print("## Preflight\n")
    print(f"- expected cells: {len(scenes)*len(conds)} (10 scenes x 9 conds), "
          f"found: {len(scenes)*len(conds) - len(missing)}")
    print(f"- missing cells : {len(missing)}" + (f" -> {missing[:8]}" if missing else ""))
    print(f"- wrong-n cells : {len(short)}" + (f" -> {short[:8]}" if short else ""))
    print(f"- null/NaN metrics: {len(bad)}" + (f" -> {bad[:8]}" if bad else ""))
    ok = not (missing or short or bad)
    print(f"- **status: {'COMPLETE' if ok else 'INCOMPLETE — results below are provisional'}**\n")
    return ok


def main():
    cells = load()
    if not preflight(cells):
        # An incomplete or non-finite matrix silently unbalances the paired tests.
        # Refusing here is the point of the check; previously its verdict was ignored.
        raise SystemExit("preflight FAILED - refusing to emit an analysis")
    scenes = sorted({s for s, _ in cells})
    print(f"# Multi-scene analysis — {len(scenes)} scenes, {sum(len(v) for v in cells.values())} runs\n")

    cellmean, within = {}, collections.defaultdict(list)
    for (sc, cond), runs in cells.items():
        cm = {}
        for k in METRICS:
            v = [r[k] for r in runs if r.get(k) is not None]
            cm[k] = st.mean(v) if v else None
            if len(v) > 1:
                within[k].append(sd(v))
        cm["_n"] = len(runs)
        cellmean[(sc, cond)] = cm

    for title, conds in [("GATE A — label-error conditions", GATE_A),
                         ("GATE B — transfer on a strong native baseline", GATE_B)]:
        have = [c for c in conds if any(c == cc for _, cc in cellmean)]
        if not have:
            continue
        print(f"\n## {title}\n")
        print("| cond | scenes | global MAE | silhouette | wall | pole | car | PSNR |")
        print("|---|---|---|---|---|---|---|---|")
        for c in have:
            rows = [cellmean[(s, c)] for s in scenes if (s, c) in cellmean]
            if not rows:
                continue
            def col(k):
                v = [r[k] for r in rows if r.get(k) is not None]
                return f"{st.mean(v):.4f}±{sd(v):.4f}" if v else "--"
            print(f"| {c} | {len(rows)} | {col('depth_mae')} | {col('depth_mae_silhouette')} | "
                  f"{col('mae_wall')} | {col('mae_pole')} | {col('mae_car')} | {col('psnr')} |")

        # ---- rank reversal: does the metric you pick change the winner? ----
        print(f"\n**Rank under each metric** (1 = best; MAE lower better, PSNR higher better)\n")
        print("| metric | " + " | ".join(have) + " |")
        print("|---" * (len(have) + 1) + "|")
        for k in ["depth_mae", "depth_mae_silhouette", "mae_pole", "mae_car", "psnr"]:
            vals = {}
            for c in have:
                rows = [cellmean[(s, c)][k] for s in scenes
                        if (s, c) in cellmean and cellmean[(s, c)].get(k) is not None]
                if rows:
                    vals[c] = st.mean(rows)
            if len(vals) < 2:
                continue
            order = sorted(vals, key=lambda c: -vals[c] if k == "psnr" else vals[c])
            rank = {c: i + 1 for i, c in enumerate(order)}
            print(f"| {k} | " + " | ".join(str(rank.get(c, "-")) for c in have) + " |")

        print(f"\n**Paired contrasts across scenes** (unit = scene, seeds averaged within cell)\n")
        print("| contrast | metric | n | delta | t | d_z | 95% CI |")
        print("|---|---|---|---|---|---|---|")
        pairs = ([("wrong", c) for c in ["shuffled_space", "resampled_mag", "sign_flipped", "correct"]]
                 + [("none", "correct")] if conds is GATE_A else
                 [("native_naive", "native"), ("native_oracle", "native"),
                  ("native_oracle", "native_naive")])
        for a, b in pairs:
            for k in ["depth_mae_silhouette", "depth_mae", "mae_pole", "mae_car", "psnr"]:
                r = paired(cellmean, have, a, b, k)
                if r:
                    print(f"| {a} − {b} | {k} | {r['n']} | {r['delta']:+.4f} | {r['t']:+.2f} | "
                          f"{r['dz']:+.2f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] |")

    print("\n## Variance decomposition — is a contrast bigger than run-to-run noise?\n")
    print("| metric | within-cell sd (run-to-run, fixed seed set) | between-scene sd |")
    print("|---|---|---|")
    for k in METRICS:
        w = st.mean(within[k]) if within[k] else float("nan")
        bs = []
        for c in GATE_A + GATE_B:
            v = [cellmean[(s, c)][k] for s in scenes
                 if (s, c) in cellmean and cellmean[(s, c)].get(k) is not None]
            if len(v) > 1:
                bs.append(sd(v))
        print(f"| {k} | {w:.4f} | {st.mean(bs):.4f} |" if bs else f"| {k} | {w:.4f} | -- |")


if __name__ == "__main__":
    main()
