#!/usr/bin/env python3
"""
Publication figures for the multi-scene study.

Palette: Okabe-Ito subset, ordered so the worst adjacent colour-vision-deficiency
separation is dE 9.6 (validated with a script, not eyeballed). Every series also
carries a distinct marker and is directly labelled, so identity is never colour-alone.
Single axis everywhere; no dual-scale plots.

Run with an env that has matplotlib (e.g. conda `base`); no torch/gsplat needed.
"""
import json, glob, os, math, collections, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "paper-stage", "figures")
os.makedirs(OUT, exist_ok=True)

PAL = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
MARK = ["o", "s", "^", "D", "v", "P"]
GATE_A = ["none", "correct", "shuffled_space", "resampled_mag", "sign_flipped", "wrong"]
GATE_B = ["native", "native_naive", "native_oracle", "native_x2"]
ALL_COND = GATE_A + GATE_B          # Table 2 averages over ALL conditions
PRETTY = {"none": "rgb-only", "correct": "correct", "shuffled_space": "shuffled-space",
          "resampled_mag": "resampled-mag", "sign_flipped": "sign-flipped", "wrong": "wrong"}
MET = [("psnr", "PSNR"), ("depth_mae", "Global\ndepth MAE"),
       ("depth_mae_silhouette", "Silhouette\nMAE"), ("mae_car", "Box\nMAE"), ("mae_pole", "Pole\nMAE")]
ALLM = ("depth_mae", "depth_mae_silhouette", "mae_wall", "mae_pole", "mae_car", "psnr")

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a8a", "axes.linewidth": 0.8,
    "grid.color": "#dcdcdc", "grid.linewidth": 0.6,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load():
    cells = collections.defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, "outputs", "multiscene_v3", "*.json")):
        b = os.path.basename(f)[:-5]; sc = b.split("_")[0]
        cells[(sc, b[len(sc) + 1: b.rindex("_s")])].append(json.load(open(f)))
    return cells


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def fig_rank_chart(cm, scenes, tie_note=True):
    ranks = {}
    for k, _ in MET:
        vals = {c: st.mean([cm[(s, c)][k] for s in scenes]) for c in GATE_A}
        order = sorted(GATE_A, key=lambda c: -vals[c] if k == "psnr" else vals[c])
        ranks[k] = {c: i + 1 for i, c in enumerate(order)}
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    x = np.arange(len(MET))
    for i, c in enumerate(GATE_A):
        y = [ranks[k][c] for k, _ in MET]
        lw, z, a = (2.4, 5, 1.0) if c in ("none", "wrong") else (1.3, 3, 0.72)
        ax.plot(x, y, color=PAL[i], marker=MARK[i], markersize=6, linewidth=lw,
                zorder=z, alpha=a, markeredgecolor="white", markeredgewidth=0.9)
        ax.annotate(PRETTY[c], (x[-1] + 0.10, y[-1]), color=PAL[i], va="center",
                    fontsize=8.2, fontweight="bold" if c in ("none", "wrong") else "normal")
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in MET])
    ax.set_yticks(range(1, 7)); ax.invert_yaxis()
    ax.set_ylabel("Rank among the six conditions  (1 = best)")
    ax.set_xlim(-0.35, len(MET) - 1 + 1.55)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    save(fig, "fig6_rank_chart")


def fig_stratified(cm, scenes):
    chans = [("mae_wall", "Far wall"), ("mae_car", "Boxes"), ("mae_pole", "Thin pole")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharex=True)
    for ax, (k, name) in zip(axes, chans):
        mu = [st.mean([cm[(s, c)][k] for s in scenes]) for c in GATE_A]
        se = [st.stdev([cm[(s, c)][k] for s in scenes]) / math.sqrt(len(scenes)) for c in GATE_A]
        ax.bar(range(len(GATE_A)), mu, yerr=se, capsize=2.5, color=PAL,
               edgecolor="white", linewidth=1.4, error_kw=dict(lw=0.9, ecolor="#555"))
        ax.set_title(name)
        ax.set_xticks(range(len(GATE_A)))
        ax.set_xticklabels([PRETTY[c] for c in GATE_A], rotation=38, ha="right")
        ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Depth MAE (m)")
    save(fig, "fig4_stratified_error")


def fig_variance(cells, cm, scenes):
    keys = [("depth_mae", "Global"), ("depth_mae_silhouette", "Silhouette"),
            ("mae_wall", "Wall"), ("mae_pole", "Pole"), ("mae_car", "Box")]
    within = [st.mean([st.stdev([r[k] for r in v]) for v in cells.values() if len(v) > 1])
              for k, _ in keys]
    between = [st.mean([st.stdev([cm[(s, c)][k] for s in scenes]) for c in ALL_COND])
               for k, _ in keys]
    x = np.arange(len(keys)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(x - w / 2, within, w, label="within cell (run-to-run, fixed seed)",
           color=PAL[1], edgecolor="white", linewidth=1.4)
    ax.bar(x + w / 2, between, w, label="between scenes",
           color=PAL[0], edgecolor="white", linewidth=1.4)
    for xi, a, b in zip(x, within, between):
        ax.text(xi, max(a, b) * 1.18, f"{b / a:.1f}x", ha="center", fontsize=7.6, color="#4a4a4a")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in keys])
    ax.set_ylabel("Standard deviation (m, log scale)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.set_ylim(top=max(between) * 3)
    print("    fig2 ratios (must match Table 2): "
          + ", ".join(f"{n} {b/a:.1f}x" for (_, n), a, b in zip(keys, within, between)))
    save(fig, "fig2_variance_decomposition")


def fig_mechanism():
    """Top-down schematic of the error mechanism. The sensor baseline is EXAGGERATED
    for legibility -- at the real 0.75 m against ~13 m of range the two rays are almost
    parallel on the page. The caption states this; the geometry is otherwise to scale."""
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    cam = np.array([0.0, 0.0]); lid = np.array([0.0, 2.0])       # baseline exaggerated
    box_x, box_hw, box_hh, wall_x = 6.0, 1.0, 1.2, 12.0
    P = np.array([wall_x, 1.0])                                   # LiDAR return on wall

    ax.add_patch(plt.Rectangle((box_x - box_hw, -box_hh), 2 * box_hw, 2 * box_hh,
                               facecolor="#ececec", edgecolor="#9a9a9a", lw=1.0, zorder=2))
    ax.text(box_x, -0.72, "occluder", fontsize=8, ha="center", color="#6a6a6a")
    ax.plot([wall_x, wall_x], [-1.45, 2.6], color="#9a9a9a", lw=2.2, zorder=2)
    ax.text(wall_x + 0.25, 2.35, "far wall", fontsize=8, color="#5a5a5a")

    ax.plot([lid[0], P[0]], [lid[1], P[1]], color=PAL[2], lw=1.9, zorder=3)
    ax.plot(*P, marker="o", color=PAL[2], markersize=7.5,
            markeredgecolor="white", markeredgewidth=1.0, zorder=5)
    d = (P - cam) / np.linalg.norm(P - cam)
    hit = cam + d * ((box_x - box_hw - cam[0]) / d[0])
    ax.plot([cam[0], P[0]], [cam[1], P[1]], color=PAL[1], lw=1.7, ls=(0, (4, 2)), zorder=3)
    ax.plot(*hit, marker="X", color=PAL[1], markersize=9.5,
            markeredgecolor="white", markeredgewidth=1.0, zorder=5)

    for p_, nm, col, dy in [(cam, "camera", PAL[1], -0.05), (lid, "LiDAR", PAL[2], 0.05)]:
        ax.plot(*p_, marker="^", color=col, markersize=9.5,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)
        ax.text(p_[0] - 0.42, p_[1] + dy, nm, fontsize=8.2, ha="right", va="center",
                color=col, fontweight="bold")

    rail = -2.15
    for px, py in ((hit[0], hit[1]), (P[0], P[1])):
        ax.plot([px, px], [rail + 0.14, py - 0.22], color="#c4c4c4", lw=0.8,
                ls=(0, (2, 2)), zorder=1)
    ax.annotate("", xy=(hit[0], rail), xytext=(P[0], rail),
                arrowprops=dict(arrowstyle="<->", color="#6a6a6a", lw=1.1))
    ax.text((hit[0] + P[0]) / 2, rail - 0.46,
            "asserted as this pixel's depth", ha="center", fontsize=8,
            color="#4a4a4a", style="italic")

    ax.set_xlim(-3.4, 14.6); ax.set_ylim(-3.5, 3.0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(handles=[
        Line2D([], [], color=PAL[2], lw=1.9, marker="o", markersize=6,
               markeredgecolor="white", label="LiDAR ray — reaches the wall"),
        Line2D([], [], color=PAL[1], lw=1.7, ls=(0, (4, 2)), marker="X", markersize=7,
               markeredgecolor="white", label="camera ray — stopped by the occluder"),
    ], frameon=False, fontsize=7.9, loc="upper center",
        bbox_to_anchor=(0.5, 0.02), ncol=1, handlelength=2.6)
    save(fig, "fig1_mechanism")


def fig_c1(cm, scenes):
    """C1: `wrong` against three magnitude-matched controls, paired per scene."""
    ctrls = [("shuffled_space", "shuffled-space\n(exact match)"),
             ("resampled_mag", "resampled-mag"), ("sign_flipped", "sign-flipped")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    conds = ["correct", "shuffled_space", "resampled_mag", "sign_flipped", "wrong"]
    mu = [st.mean([cm[(s, c)]["depth_mae_silhouette"] for s in scenes]) for c in conds]
    se = [st.stdev([cm[(s, c)]["depth_mae_silhouette"] for s in scenes]) / math.sqrt(len(scenes))
          for c in conds]
    cols = [PAL[1], PAL[2], PAL[3], PAL[4], PAL[0]]
    ax.bar(range(len(conds)), mu, yerr=se, capsize=2.5, color=cols,
           edgecolor="white", linewidth=1.4, error_kw=dict(lw=0.9, ecolor="#555"))
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([PRETTY[c] for c in conds], rotation=38, ha="right")
    ax.set_ylabel("Boundary-band depth MAE (m)")
    ax.set_title("Same residual magnitudes,\ndifferent damage", fontsize=8.6)
    ax.grid(axis="y"); ax.set_axisbelow(True)

    ax = axes[1]
    for i, (c, lab) in enumerate(ctrls):
        d = [cm[(s, "wrong")]["depth_mae_silhouette"] - cm[(s, c)]["depth_mae_silhouette"]
             for s in scenes]
        ax.scatter([i] * len(d), d, s=26, color=PAL[i + 2], alpha=0.75,
                   edgecolor="white", linewidth=0.8, zorder=3)
        m_ = st.mean(d)
        ax.plot([i - 0.24, i + 0.24], [m_, m_], color=PAL[i + 2], lw=2.4, zorder=4)
    ax.axhline(0, color="#9a9a9a", lw=1.0, zorder=1)
    ax.set_xticks(range(len(ctrls)))
    ax.set_xticklabels([l for _, l in ctrls], fontsize=7.8)
    ax.set_ylabel("`wrong` − control, per scene (m)")
    ax.set_title("Structured error is worse in every scene", fontsize=8.6)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    save(fig, "fig3_structure_effect")


def fig_c2(cm, scenes):
    """C2: transfer conditions on the native baseline, including the budget control."""
    conds = ["native", "native_x2", "native_naive", "native_oracle"]
    lab = {"native": "native", "native_x2": "native ×2 (budget)",
           "native_naive": "+ unfiltered transfer", "native_oracle": "+ filtered transfer"}
    chans = [("depth_mae_silhouette", "Boundary band"), ("mae_pole", "Thin pole"), ("psnr", "PSNR")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    for ax, (k, name) in zip(axes, chans):
        mu = [st.mean([cm[(s, c)][k] for s in scenes]) for c in conds]
        se = [st.stdev([cm[(s, c)][k] for s in scenes]) / math.sqrt(len(scenes)) for c in conds]
        ax.bar(range(len(conds)), mu, yerr=se, capsize=2.5,
               color=[PAL[0], PAL[5], PAL[1], PAL[2]],
               edgecolor="white", linewidth=1.4, error_kw=dict(lw=0.9, ecolor="#555"))
        ax.set_title(name, fontsize=9)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([lab[c] for c in conds], rotation=30, ha="right", fontsize=7.2)
        ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Depth MAE (m)")
    axes[2].set_ylabel("dB")
    save(fig, "fig5_transfer")


if __name__ == "__main__":
    cells = load()
    scenes = sorted({s for s, _ in cells})
    cm = {k: {m: st.mean([r[m] for r in v]) for m in ALLM} for k, v in cells.items()}
    print(f"figures from {len(scenes)} scenes, {sum(len(v) for v in cells.values())} runs")
    fig_mechanism()
    fig_c1(cm, scenes)
    fig_stratified(cm, scenes)
    fig_variance(cells, cm, scenes)
    fig_c2(cm, scenes)
    fig_rank_chart(cm, scenes)
