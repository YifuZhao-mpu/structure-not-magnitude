#!/usr/bin/env python3
"""
Figure for the §5.1 mechanism measurement (PRE_ANALYSIS_PLAN_M.md).

Left  : accumulated opacity as a function of depth RELATIVE to the analytic surface,
        averaged over foreground pixels. This is the picture the M1/M2 scalars summarise.
Right : M1 per scene, paired -- the unit of replication is the scene.

Same palette and conventions as make_figures.py (Okabe-Ito subset, single axis,
every series directly labelled and carrying a distinct marker).
"""
import json, glob, os, collections, math, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "paper-stage", "figures")
SRC = os.path.join(ROOT, "outputs", "probe_M")
CONDS = ["correct", "shuffled_space", "wrong"]
LAB = {"correct": "correct", "shuffled_space": "shuffled-space",
       "wrong": "wrong"}
TICK = {"correct": "correct", "shuffled_space": "shuffled-\nspace", "wrong": "wrong"}
COL = {"correct": "#D55E00", "shuffled_space": "#009E73", "wrong": "#0072B2"}
MK = {"correct": "s", "shuffled_space": "^", "wrong": "o"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a8a", "axes.linewidth": 0.8,
    "grid.color": "#dcdcdc", "grid.linewidth": 0.6,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def main():
    prof = collections.defaultdict(list)
    m1 = collections.defaultdict(list)
    off = None
    for f in glob.glob(f"{SRC}/sc*.json"):
        r = json.load(open(f))
        if "probe_prof" not in r:
            continue
        sc = os.path.basename(r["data"].rstrip("/"))
        off = np.array(r["probe_prof_off"])
        prof[r["cond"]].append(np.array(r["probe_prof"]))
        m1[(sc, r["cond"])].append(r["probe_m1"])
    scenes = sorted({s for s, _ in m1})
    print(f"figure from {len(scenes)} scenes, {sum(len(v) for v in prof.values())} runs")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    end = {}
    for c in CONDS:
        P = np.stack(prof[c]); mu = P.mean(0)
        se = P.std(0, ddof=1) / math.sqrt(len(P))
        ax.plot(off, mu, color=COL[c], lw=1.8, marker=MK[c], ms=3.4, markevery=3, zorder=3)
        ax.fill_between(off, mu - se, mu + se, color=COL[c], alpha=0.25, lw=0)
        end[c] = mu[-1]
        ax.annotate(LAB[c], (off[-1], mu[-1]), fontsize=7.6, color=COL[c],
                    xytext=(5, {"correct": 0, "shuffled_space": 7, "wrong": -8}[c]),
                    textcoords="offset points", va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.8))
    ax.axvline(0, color="#7a7a7a", lw=1.0, ls="--", zorder=1)
    ax.annotate("analytic surface", (0, 0.5), fontsize=7.2, color="#5a5a5a",
                rotation=90, ha="right", va="center",
                xytext=(-3, 0), textcoords="offset points")
    # the load-bearing feature is the gap that never closes
    ax.annotate("", xy=(11.5, end["correct"]), xytext=(11.5, end["wrong"]),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=0.9,
                                shrinkA=0, shrinkB=0))
    ax.annotate(f"{end['correct'] - end['wrong']:.2f} of the opacity never\n"
                "accumulates anywhere on the ray:\nthe structure is eroded, and the\n"
                "camera sees through it",
                (11.5, 0.42), fontsize=7.4, color="#333", ha="center", va="center")
    ax.plot([11.5, 11.5], [0.60, end["wrong"] - 0.012], color="#9a9a9a", lw=0.7, ls=":")
    ax.set_xlabel("Depth relative to the analytic surface (m)")
    ax.set_ylabel("Accumulated opacity")
    ax.set_title("The foreground is eroded, not displaced", fontsize=8.8)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.set_ylim(0, 1.06); ax.set_xlim(off[0] - 0.5, off[-1] + 7.5)

    ax = axes[1]
    for i, c in enumerate(CONDS):
        v = [st.mean(m1[(s, c)]) for s in scenes]
        ax.scatter([i] * len(v), v, s=26, color=COL[c], alpha=0.8,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot([i - 0.26, i + 0.26], [st.mean(v)] * 2, color=COL[c], lw=2.6, zorder=4)
    for s in scenes:                      # pairing lines: the unit is the scene
        ax.plot(range(3), [st.mean(m1[(s, c)]) for c in CONDS],
                color="#b8b8b8", lw=0.5, zorder=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([TICK[c] for c in CONDS], fontsize=7.8)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Opacity at the analytic surface")
    ax.set_title("Worse in 10 of 10 scenes", fontsize=8.8)
    ax.grid(axis="y"); ax.set_axisbelow(True)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig7_mechanism_measured.{ext}"))
    print("  wrote fig7_mechanism_measured.pdf / .png")


main()
