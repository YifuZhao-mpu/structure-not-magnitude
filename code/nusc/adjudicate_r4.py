#!/usr/bin/env python3
"""
Strict adjudication of R-4 against paper-stage/PRE_ANALYSIS_PLAN_R4.md.

Written before the matrix finished, for the same reason the synthetic adjudicator was:
a criterion restated in prose gets read loosely. Rules implemented literally:
  unit = scene (n=10), seeds averaged within cell;
  paired two-sided t at alpha=.05 (t_crit=2.262);
  substantive threshold = mean within-cell run-to-run SD for that metric;
  corroborated iff native_naive - native > 0, significant AND above threshold;
  downgraded if native_x2 - native is also positive and of comparable magnitude.
"""
import os
import json, glob, os, math, collections, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))   # repo root, not a fixed path
PRIMARY = "range_mae"
METRICS = ["range_mae", "range_mae_fg", "range_mae_bg", "psnr"]
CONDS = ["native", "native_naive", "native_x2"]
TCRIT = 2.262

cells = collections.defaultdict(list)
for f in glob.glob(f"{ROOT}/outputs/nusc_r4/*.json"):
    b = os.path.basename(f)[:-5]
    sc = b.split("_")[0]
    cond = b[len(sc) + 1: b.rindex("_s")]
    cells[(sc, cond)].append(json.load(open(f)))
scenes = sorted({s for s, _ in cells})
cm = {k: {m: st.mean([r[m] for r in v if r.get(m) is not None]) for m in METRICS}
      for k, v in cells.items()}
within = {m: st.mean([st.stdev([r[m] for r in v if r.get(m) is not None])
                      for v in cells.values() if len(v) > 1]) for m in METRICS}


def paired(a, b, m):
    d = [cm[(s, a)][m] - cm[(s, b)][m] for s in scenes if (s, a) in cm and (s, b) in cm]
    if len(d) < 2:
        return None
    mu, sd = st.mean(d), st.stdev(d)
    se = sd / math.sqrt(len(d))
    return dict(n=len(d), d=mu, t=(mu / se if se else float("inf")),
                lo=mu - TCRIT * se, hi=mu + TCRIT * se)


print("# R-4 adjudication vs PRE_ANALYSIS_PLAN_R4.md\n")
print(f"- scenes {len(scenes)}, runs {sum(len(v) for v in cells.values())}, "
      f"conditions {len({c for _, c in cells})}")
print(f"- primary `{PRIMARY}`, substantive threshold (within-cell run-to-run SD) "
      f"= {within[PRIMARY]:.4f} m\n")

print("## Condition means (scene-level, seeds averaged)\n")
print("| condition | range MAE (m) | fg | bg | PSNR |")
print("|---|---|---|---|---|")
for c in CONDS:
    v = {m: [cm[(s, c)][m] for s in scenes if (s, c) in cm] for m in METRICS}
    print(f"| `{c}` | " + " | ".join(
        f"{st.mean(v[m]):.4f} ± {st.stdev(v[m]):.4f}" if m != "psnr"
        else f"{st.mean(v[m]):.2f} ± {st.stdev(v[m]):.2f}" for m in METRICS) + " |")

print("\n## Contrasts\n")
print("| contrast | metric | Δ | t | 95% CI | > threshold |")
print("|---|---|---|---|---|---|")
rows = {}
for a, b in [("native_naive", "native"), ("native_x2", "native"),
             ("native_naive", "native_x2")]:
    for m in METRICS:
        r = paired(a, b, m)
        if not r:
            continue
        rows[(a, b, m)] = r
        ok = abs(r["d"]) > within[m]
        print(f"| `{a}` − `{b}` | {m} | {r['d']:+.4f} | {r['t']:+.2f} | "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}] | {'yes' if ok else '**NO**'} |")

print("\n## Verdict\n")
mn = rows.get(("native_naive", "native", PRIMARY))
x2 = rows.get(("native_x2", "native", PRIMARY))
harm = mn and mn["d"] > 0 and abs(mn["t"]) > TCRIT and abs(mn["d"]) > within[PRIMARY]
x2_harm = x2 and x2["d"] > 0 and abs(x2["t"]) > TCRIT and abs(x2["d"]) > within[PRIMARY]
comparable = x2_harm and mn and abs(x2["d"]) > 0.5 * abs(mn["d"])

print(f"- `native_naive` − `native` on {PRIMARY}: Δ={mn['d']:+.4f}, t={mn['t']:+.2f}, "
      f"threshold {within[PRIMARY]:.4f} → {'harm established' if harm else 'NOT established'}")
print(f"- weight control `native_x2` − `native`: Δ={x2['d']:+.4f}, t={x2['t']:+.2f} → "
      f"{'also harmful' if x2_harm else 'not harmful'}")
if harm and not comparable:
    print("\n**CORROBORATED.** Unfiltered camera-side transfer harms the native-ray "
          "baseline on real data, and the harm is not reproduced by doubling the native "
          "loss weight.")
elif harm and comparable:
    print("\n**CORROBORATED BUT DOWNGRADED.** Harm is established, but the weight control "
          "is harmful by a comparable amount, so it cannot be separated from loss weighting.")
else:
    print("\n**NOT CORROBORATED.** The synthetic finding did not reproduce on real data "
          "under the frozen rule. Reported as such.")
