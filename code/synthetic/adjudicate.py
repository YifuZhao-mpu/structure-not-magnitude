#!/usr/bin/env python3
"""
Strict adjudication against PRE_ANALYSIS_PLAN.md, implemented literally.

This exists because the criteria were once read too loosely: the plan defines a
"decision-relevant reversal" by the WINNING condition differing between metrics, and a
previous pass instead pointed at pairwise inversions among non-winning conditions and
declared C3 supported. Encoding the rule removes that discretion.

Usage: python adjudicate.py [outputs/multiscene_v2]
"""
import json, glob, os, sys, math, collections, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "outputs/multiscene_v3"
PRIMARY = "depth_mae_silhouette"
GEOM = ["depth_mae", "depth_mae_silhouette", "mae_wall", "mae_pole", "mae_car"]
APPEAR = ["psnr"]
GLOBALG = ["depth_mae", "depth_mae_silhouette", "mae_wall"]
FINE = ["mae_pole"]
ALLM = GEOM + APPEAR
GATE_A = ["none", "correct", "shuffled_space", "resampled_mag", "sign_flipped", "wrong"]
GATE_B = ["native", "native_naive", "native_oracle", "native_x2"]

cells = collections.defaultdict(list)
for f in glob.glob(os.path.join(ROOT, OUTDIR, "*.json")):
    b = os.path.basename(f)[:-5]; sc = b.split("_")[0]
    cells[(sc, b[len(sc) + 1: b.rindex("_s")])].append(json.load(open(f)))
scenes = sorted({s for s, _ in cells})
cm = {k: {m: st.mean([r[m] for r in v]) for m in ALLM} for k, v in cells.items()}
within = {m: st.mean([st.stdev([r[m] for r in v]) for v in cells.values() if len(v) > 1])
          for m in ALLM}


def paired(a, b, m):
    d = [cm[(s, a)][m] - cm[(s, b)][m] for s in scenes]
    mu, sd = st.mean(d), st.stdev(d)
    se = sd / math.sqrt(len(d))
    return mu, (mu / se if se else float("inf")), mu - 2.262 * se, mu + 2.262 * se


def best(conds, m):
    """Winner under metric m, plus whether that win is ESTABLISHED.

    The plan's substantive threshold ("a difference smaller than run noise is not a
    finding, regardless of p-value") applies here too: claiming that metric X picks a
    different winner from metric Y is only a finding if each winner is actually ahead
    of its runner-up. An argmin alone is not a winner. Omitting this check is how the
    first adjudicator pass over-reported C3.
    """
    v = {c: st.mean([cm[(s, c)][m] for s in scenes]) for c in conds if (scenes[0], c) in cm}
    order = sorted(v, key=lambda c: -v[c] if m == "psnr" else v[c])
    w, runner = order[0], order[1]
    mu, t, _, _ = paired(w, runner, m)
    established = abs(mu) > within[m] and abs(t) > 2.262
    return w, v, established, abs(mu), t, runner


def verdict(ok): return "**SUPPORTED**" if ok else "**NOT SUPPORTED**"


print(f"# Strict adjudication vs PRE_ANALYSIS_PLAN.md  ({OUTDIR})\n")
print(f"- scenes {len(scenes)}, runs {sum(len(v) for v in cells.values())}, "
      f"conditions {len({c for _, c in cells})}")
print(f"- primary metric `{PRIMARY}`, substantive threshold "
      f"(within-cell run-to-run SD) = {within[PRIMARY]:.4f}\n")

# ---------------- C1 ----------------
print("## C1 — structured vs magnitude-matched unstructured\n")
print("Rule: all three contrasts positive, |t|>2.262, AND |delta| > within-cell SD.\n")
print("| contrast | delta | t | 95% CI | > threshold | verdict |")
print("|---|---|---|---|---|---|")
c1 = []
for b in ["shuffled_space", "resampled_mag", "sign_flipped"]:
    mu, t, lo, hi = paired("wrong", b, PRIMARY)
    ok = mu > 0 and abs(t) > 2.262 and abs(mu) > within[PRIMARY]
    c1.append(ok)
    print(f"| wrong - {b} | {mu:+.4f} | {t:+.2f} | [{lo:+.4f},{hi:+.4f}] | "
          f"{'yes' if abs(mu) > within[PRIMARY] else 'NO'} | {'pass' if ok else 'FAIL'} |")
print(f"\n**C1 = {verdict(all(c1))}** ({sum(c1)}/3 contrasts pass)\n")

# ---------------- C2 ----------------
print("## C2 — filtering before transfer\n")
print("Rule: native_naive-native > 0 AND native_oracle-native < 0, both significant "
      "and above threshold.\n")
print("| contrast | delta | t | 95% CI | verdict |")
print("|---|---|---|---|---|")
c2 = []
for a, b, want in [("native_naive", "native", +1), ("native_oracle", "native", -1)]:
    mu, t, lo, hi = paired(a, b, PRIMARY)
    ok = (mu * want > 0) and abs(t) > 2.262 and abs(mu) > within[PRIMARY]
    c2.append(ok)
    print(f"| {a} - {b} | {mu:+.4f} | {t:+.2f} | [{lo:+.4f},{hi:+.4f}] | "
          f"{'pass' if ok else 'FAIL'} |")
if ("native_x2" in {c for _, c in cells}):
    mu, t, lo, hi = paired("native_oracle", "native_x2", PRIMARY)
    # PRE_ANALYSIS_PLAN amendment: if oracle does not beat the budget control, the
    # filtering benefit cannot be separated from supervision budget and C2 degrades.
    # This check was previously printed but never entered the verdict.
    ok = mu < 0 and abs(t) > 2.262 and abs(mu) > within[PRIMARY]
    c2.append(ok)
    print(f"| native_oracle - native_x2 (budget control) | {mu:+.4f} | {t:+.2f} | "
          f"[{lo:+.4f},{hi:+.4f}] | {'pass' if ok else 'FAIL'} |")
print(f"\n**C2 = {verdict(all(c2))}**\n")

# ---------------- C3 ----------------
print("## C3 — bidirectional decision-relevant reversal\n")
print("Rule (verbatim): a decision-relevant reversal is when the BEST condition under\n"
      "one metric is not the BEST under another. Two OPPOSITE-direction reversals are\n"
      "required: (i) appearance-best != geometry-best, (ii) global-geometry-best !=\n"
      "fine-structure-best.\n")
for gate, conds in [("GATE A", GATE_A), ("GATE B", GATE_B)]:
    print(f"\n### {gate}\n")
    winners, estab = {}, {}
    print("| metric | best condition | margin over runner-up | t | established? |")
    print("|---|---|---|---|---|")
    for m in ALLM:
        w, _, ok, mg, t, ru = best(conds, m)
        winners[m], estab[m] = w, ok
        print(f"| {m} | `{w}` | {mg:.4f} over `{ru}` (thr {within[m]:.4f}) | {t:+.2f} | "
              f"{'yes' if ok else '**NO — within run noise**'} |")
    # only ESTABLISHED winners may participate in a reversal claim
    ge = [m for m in GEOM if estab[m]]
    d1 = estab["psnr"] and bool(ge) and winners["psnr"] not in {winners[m] for m in ge}
    gg = [m for m in GLOBALG if estab[m]]; ff = [m for m in FINE if estab[m]]
    gset = {winners[m] for m in gg}; fset = {winners[m] for m in ff}
    d2 = bool(gg) and bool(ff) and not (gset & fset)
    print(f"\n- direction (i) appearance-best `{winners['psnr']}` vs established "
          f"geometry-best {sorted({winners[m] for m in ge})} -> "
          f"{'REVERSAL' if d1 else 'no established reversal'}")
    print(f"- direction (ii) established global-geometry-best {sorted(gset)} vs established "
          f"fine-structure-best {sorted(fset) if ff else '(none established)'} -> "
          f"{'REVERSAL' if d2 else 'no established reversal'}")
    print(f"- **{gate}: {verdict(d1 and d2)}**")

print("\n### Weaker surviving observation (reported separately, not as C3)\n")
for gate, conds in [("GATE A", GATE_A), ("GATE B", GATE_B)]:
    import itertools
    rk = {}
    for m in ALLM:
        _, v, *_ = best(conds, m)
        order = sorted(v, key=lambda c: -v[c] if m == "psnr" else v[c])
        rk[m] = {c: i for i, c in enumerate(order)}
    flips = sum(1 for a, b in itertools.combinations([c for c in conds if (scenes[0], c) in cm], 2)
                if (rk["depth_mae"][a] - rk["depth_mae"][b]) * (rk["mae_pole"][a] - rk["mae_pole"][b]) < 0)
    print(f"- {gate}: global depth MAE vs pole MAE disagree on **{flips}** condition pairs")
