#!/usr/bin/env python3
"""
§5.1 mechanism measurement adjudicator. Implements PRE_ANALYSIS_PLAN_M.md literally,
including Appendix A (equal-pixel-budget M3).

Unit of replication is the SCENE (n=10); the 3 seeds inside a (scene,condition) cell
are averaged first and their spread supplies the substantive threshold.

Fails closed. An incomplete matrix, a NaN metric or an undefined M3 is an INVALID
MEASUREMENT, never evidence against the mechanism -- the frozen table disposes of
claims that were *tested*, and a claim that could not be tested has not failed.
"""
import os, sys, json, glob, itertools
import numpy as np
from scipy import ndimage, stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))   # repo root, not a fixed path
OUT = f"{ROOT}/outputs/probe_M"
DATA = f"{ROOT}/data/multiscene_v3"
SCENES = [f"sc{k:02d}" for k in range(10)]
CONDS = ["correct", "shuffled_space", "wrong"]
SEEDS = [0, 1, 2]
PID_FG = (2, 3, 4)          # carL, carR, pole
CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
DEPL_TAU, MIN_COMP = 0.05, 25


def cells(metric):
    """-> per-scene mean over seeds, and the mean within-cell run-to-run sd.
    Every cell must be complete and finite; otherwise the metric is unusable."""
    M = np.full((len(SCENES), len(CONDS)), np.nan)
    sds = []
    for i, sc in enumerate(SCENES):
        for j, c in enumerate(CONDS):
            v = []
            for sd in SEEDS:
                f = f"{OUT}/{sc}_{c}_s{sd}.json"
                if os.path.exists(f) and os.path.getsize(f):
                    x = json.load(open(f)).get(metric)
                    if x is not None and np.isfinite(x):
                        v.append(x)
            if len(v) == len(SEEDS):
                M[i, j] = np.mean(v)
                sds.append(np.std(v, ddof=1))
    if not sds or not np.isfinite(M).all():
        raise SystemExit(f"INVALID: metric {metric!r} has incomplete or non-finite "
                         f"cells ({int(np.isfinite(M).sum())}/{M.size} present). "
                         f"No verdict is issued.")
    return M, float(np.mean(sds))


def paired(M, ja, jb, sd_thresh, label):
    """test M[:,ja] > M[:,jb] over scenes; 'established' needs BOTH criteria.
    The critical value follows the ACTUAL n, not a hardcoded df."""
    a, b = M[:, ja], M[:, jb]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise SystemExit(f"INVALID: non-finite values in contrast {label!r}")
    d = a - b
    n = len(d)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t, p = stats.ttest_rel(a, b)
    est = (abs(d.mean()) > sd_thresh) and (abs(t) > tcrit)
    print(f"  {label:<40} delta={d.mean():+.4f}  t={t:+.2f}  p={p:.4f}  "
          f"n={n}  t*={tcrit:.3f}  wins={int((d > 0).sum())}/{n}  "
          f"{'ESTABLISHED' if est else 'not established'}")
    return d.mean(), float(t), bool(est)


def fg_mask(sc, i):
    g = np.load(f"{DATA}/{sc}/rgb/{i:04d}_depth.npy")
    pid = np.load(f"{DATA}/{sc}/rgb/{i:04d}_pid.npy")
    return (g > 0) & np.isin(pid, PID_FG)


def topn_masks(Dlist, masks, N):
    """select EXACTLY N pixels with the largest depletion, pooled over views.
    A `>= threshold` rule would take every pixel tied at the cutoff and break the
    equal-budget guarantee, so select by index with a deterministic tie rule."""
    vals = np.concatenate([D[m] for D, m in zip(Dlist, masks)])
    if N <= 0 or N > vals.size:
        return None
    order = np.lexsort((np.arange(vals.size), -vals))[:N]   # ties -> lower index first
    take = np.zeros(vals.size, bool); take[order] = True
    out, o = [], 0
    for D, m in zip(Dlist, masks):
        k = int(m.sum()); sub = np.zeros_like(m)
        sub[m] = take[o:o + k]; o += k
        out.append(sub)
    assert sum(int(s.sum()) for s in out) == N
    return out


def coherence(sel):
    """fraction of selected pixels lying in 4-connected components of >= MIN_COMP,
    labelled SEPARATELY PER VIEW"""
    tot = big = 0
    for b in sel:
        if not b.any():
            continue
        lab, _ = ndimage.label(b, structure=CONN4)
        sz = np.bincount(lab.ravel())[1:]
        tot += int(b.sum()); big += int(sz[sz >= MIN_COMP].sum())
    return (big / tot) if tot else np.nan


def m3():
    """equal-budget spatial coherence of the depletion (Appendix A), plus the
    unmatched fixed-threshold diagnostic with its OWN variability threshold"""
    n_views = json.load(open(f"{DATA}/sc00/meta.json"))["n_views"]
    views = list(range(0, n_views, 4))
    Mm = np.full((len(SCENES), 2), np.nan)     # [:,0]=shuffled  [:,1]=wrong
    Mu = np.full((len(SCENES), 2), np.nan)
    sdm, sdu = [], []
    for i, sc in enumerate(SCENES):
        rows_m, rows_u = [], []
        masks = [fg_mask(sc, v) for v in views]
        for sd in SEEDS:
            got = {}
            for c in CONDS:
                fs = [f"{OUT}/maps/{sc}_{c}_s{sd}/anear_{v:04d}.npy" for v in views]
                if all(os.path.exists(x) for x in fs):
                    got[c] = [np.load(x).astype(np.float64) for x in fs]
            if len(got) != 3:
                raise SystemExit(f"INVALID: missing probe maps for {sc} seed {sd}. "
                                 f"No verdict is issued.")
            D = {c: [got["correct"][k] - got[c][k] for k in range(len(views))]
                 for c in ("shuffled_space", "wrong")}
            N = min(int(sum(int(((D[c][k] > DEPL_TAU) & masks[k]).sum())
                            for k in range(len(views))))
                    for c in ("shuffled_space", "wrong"))
            rm, ru = [], []
            for c in ("shuffled_space", "wrong"):
                sel = topn_masks(D[c], masks, N)
                rm.append(np.nan if sel is None else coherence(sel))
                ru.append(coherence([(D[c][k] > DEPL_TAU) & masks[k]
                                     for k in range(len(views))]))
            rows_m.append(rm); rows_u.append(ru)
        rows_m, rows_u = np.array(rows_m), np.array(rows_u)
        if rows_m.shape[0] != len(SEEDS) or not np.isfinite(rows_m).all():
            raise SystemExit(f"INVALID: M3 undefined for {sc} "
                             f"(equal-budget N may be 0). No verdict is issued.")
        Mm[i] = rows_m.mean(0); Mu[i] = rows_u.mean(0)
        sdm.append(np.mean(rows_m.std(0, ddof=1)))
        if np.isfinite(rows_u).all():
            sdu.append(np.mean(rows_u.std(0, ddof=1)))
    if not sdm:
        raise SystemExit("INVALID: no within-cell SD for M3. No verdict is issued.")
    return Mm, Mu, float(np.mean(sdm)), (float(np.mean(sdu)) if sdu else float("nan"))


def main():
    have = len(glob.glob(f"{OUT}/sc*.json"))   # the verdict file lives here too
    need = len(SCENES) * len(CONDS) * len(SEEDS)
    print(f"runs present: {have}/{need}")
    if have < need:
        raise SystemExit("INVALID: the matrix is incomplete. The frozen table disposes "
                         "of claims that were TESTED; an untested claim has not failed.")
    print()
    V = {}

    print("P1  opacity at the true surface (M1) -- higher = the surface is still there")
    M1, sd1 = cells("probe_m1")
    print(f"  within-cell run-to-run sd = {sd1:.4f}  (substantive threshold)")
    print("  means: " + "  ".join(f"{c}={M1[:, j].mean():.4f}" for j, c in enumerate(CONDS)))
    a = paired(M1, 0, 1, sd1, "correct > shuffled_space")
    b = paired(M1, 1, 2, sd1, "shuffled_space > wrong")
    c_ = paired(M1, 0, 2, sd1, "correct > wrong  (diagnostic only)")
    V["P1"] = bool(a[2] and a[0] > 0 and b[2] and b[0] > 0)

    print("\nP2  opacity in the 10 m behind the true surface (M2) -- mass in empty space")
    M2, sd2 = cells("probe_m2")
    print(f"  within-cell run-to-run sd = {sd2:.4f}")
    print("  means: " + "  ".join(f"{c}={M2[:, j].mean():.4f}" for j, c in enumerate(CONDS)))
    d = paired(M2, 2, 1, sd2, "wrong > shuffled_space")
    V["P2"] = bool(d[2] and d[0] > 0)

    print("\nP3  spatial coherence of the depletion (M3)")
    Mm, Mu, sd3m, sd3u = m3()
    print(f"  within-cell sd: matched={sd3m:.4f}  unmatched={sd3u:.4f}")
    print(f"  matched   : shuffled={Mm[:, 0].mean():.4f}  wrong={Mm[:, 1].mean():.4f}")
    print(f"  unmatched : shuffled={Mu[:, 0].mean():.4f}  wrong={Mu[:, 1].mean():.4f}")
    e = paired(Mm, 1, 0, sd3m, "wrong > shuffled  (MATCHED, decisive)")
    V["P3"] = bool(e[2] and e[0] > 0)
    # the unmatched variant is a DIAGNOSTIC; if it is undefined that must not abort
    # the decisive test, and it must not be silently reported as agreement either
    if np.isfinite(Mu).all() and np.isfinite(sd3u):
        f_ = paired(Mu, 1, 0, sd3u, "wrong > shuffled  (unmatched, diagnostic)")
        V["P3_unmatched_agrees"] = bool((e[0] > 0) == (f_[0] > 0))
    else:
        print("  unmatched diagnostic UNDEFINED (no pixels above the fixed threshold)")
        V["P3_unmatched_agrees"] = None

    # ---- the frozen decision table of PRE_ANALYSIS_PLAN_M.md §4, applied literally ----
    if not V["P1"]:
        disp = ("P1 failed -> §5.1(A) AND §5.1(B) are both RETRACTED; "
                "report as a negative result")
    elif V["P1"] and V["P2"] and V["P3"]:
        disp = "§5.1 upgraded from an interpretation to a MEASURED mechanism"
    elif V["P1"] and V["P2"] and not V["P3"]:
        disp = ("claim surface relocation ONLY; explicitly RETRACT the coherence "
                "claim that mutually consistent targets are satisfied simultaneously")
    else:
        disp = (f"P1 held but P2 did not (P3={V['P3']}). The frozen table has no row "
                f"for this combination, so no positive disposition is available: "
                f"report the measured quantities and claim neither (A) nor (B).")
    print("\n" + "=" * 72)
    for k, v in V.items():
        print(f"  {k:<22} "
              f"{'UNDEFINED' if v is None else ('SUPPORTED' if v else 'NOT SUPPORTED')}")
    print(f"\n  DISPOSITION: {disp}")
    json.dump({"verdict": V, "disposition": disp,
               "sd": {"m1": sd1, "m2": sd2, "m3_matched": sd3m, "m3_unmatched": sd3u},
               "M1": M1.tolist(), "M2": M2.tolist(),
               "M3_matched": Mm.tolist(), "M3_unmatched": Mu.tolist()},
              open(f"{OUT}/verdict_M.json", "w"), indent=1)


main()
