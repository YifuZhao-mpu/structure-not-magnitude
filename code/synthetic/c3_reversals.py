#!/usr/bin/env python3
"""C3: Kendall tau between metric rankings + explicit enumeration of decision-relevant
reversals, exactly as pre-registered in paper-stage/PRE_ANALYSIS_PLAN.md §3."""
import os
import json, glob, os, itertools, collections, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))   # repo root, not a fixed path
METRICS = ["depth_mae", "depth_mae_silhouette", "mae_wall", "mae_pole", "mae_car", "psnr"]
GATE_A = ["none", "correct", "shuffled_space", "sign_flipped", "resampled_mag", "wrong"]
GATE_B = ["native", "native_naive", "native_oracle"]

cells = collections.defaultdict(list)
for f in glob.glob(f"{ROOT}/outputs/multiscene_v3/*.json"):
    b = os.path.basename(f)[:-5]; sc = b.split("_")[0]
    cond = b[len(sc)+1: b.rindex("_s")]
    cells[(sc, cond)].append(json.load(open(f)))

def mean_of(cond, k):
    v = [st.mean([r[k] for r in cells[(s, cond)]])
         for s in sorted({a for a, c in cells if c == cond})]
    return st.mean(v)

def kendall(a, b):
    """a,b: dict cond->rank. Returns tau-b over the shared conds."""
    ks = sorted(set(a) & set(b)); n = len(ks); c = d = 0
    for i, j in itertools.combinations(range(n), 2):
        s = (a[ks[i]] - a[ks[j]]) * (b[ks[i]] - b[ks[j]])
        c += s > 0; d += s < 0
    return (c - d) / (n * (n - 1) / 2)

for name, conds in [("GATE A", GATE_A), ("GATE B", GATE_B)]:
    print(f"\n## {name}\n")
    vals = {k: {c: mean_of(c, k) for c in conds} for k in METRICS}
    ranks = {}
    for k in METRICS:
        order = sorted(conds, key=lambda c: -vals[k][c] if k == "psnr" else vals[k][c])
        ranks[k] = {c: i + 1 for i, c in enumerate(order)}
    print("**Kendall tau between metric rankings** (1.0 = identical ordering)\n")
    hdr = [m for m in METRICS]
    print("| | " + " | ".join(hdr) + " |"); print("|---" * (len(hdr) + 1) + "|")
    for a in hdr:
        print(f"| {a} | " + " | ".join(f"{kendall(ranks[a], ranks[b]):+.2f}" for b in hdr) + " |")

    print("\n**Decision-relevant reversals** (best under metric X is not best under metric Y)\n")
    seen = set()
    for a, b in itertools.combinations(METRICS, 2):
        ba = min(ranks[a], key=ranks[a].get); bb = min(ranks[b], key=ranks[b].get)
        if ba != bb and (ba, bb) not in seen:
            seen.add((ba, bb))
            print(f"- `{a}` picks **{ba}** ; `{b}` picks **{bb}**  "
                  f"(under `{b}`, {ba} ranks {ranks[b][ba]}/{len(conds)})")

    print("\n**Pairwise order flips between global and stratified geometry**\n")
    for a, b in [("depth_mae", "mae_pole"), ("depth_mae", "mae_car"),
                 ("depth_mae_silhouette", "mae_pole"), ("psnr", "depth_mae")]:
        flips = [(x, y) for x, y in itertools.combinations(conds, 2)
                 if (ranks[a][x] - ranks[a][y]) * (ranks[b][x] - ranks[b][y]) < 0]
        if flips:
            print(f"- `{a}` vs `{b}` — {len(flips)} flipped pair(s):")
            for x, y in flips:
                print(f"    - {x} vs {y}: `{a}` says {x if ranks[a][x]<ranks[a][y] else y} better "
                      f"({vals[a][x]:.4f} vs {vals[a][y]:.4f}); "
                      f"`{b}` says {x if ranks[b][x]<ranks[b][y] else y} better "
                      f"({vals[b][x]:.4f} vs {vals[b][y]:.4f})")
