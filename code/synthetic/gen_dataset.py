"""
GATE A dataset builder.

Produces, for one controlled configuration:
  - RGB images + exact camera depth along the trajectory
  - LiDAR returns with EXACT per-return co-visibility state
  - the supervision label variants that the causal test compares

The decisive design point (v0 got this wrong): every label variant is defined on the
SAME set of camera pixels. v0's "oracle" condition DROPPED contaminated labels, which
confounds label correctness with supervision support. Here `correct`, `wrong` and
`none` differ only in what value sits at an identical pixel set.

Coherence ablations preserve the signed residual distribution and the affected pixel
set, and destroy only structure:
  coherent        - the real thing: spatially + cross-view coherent, always "too far"
  shuffled_space  - same residuals, reassigned to other contaminated pixels in-frame
  shuffled_view   - residuals resampled independently per view
  sign_flipped    - magnitudes kept, signs randomised

If `coherent` is no worse than these, the harm is "just noise" and Claim 1 dies.
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import default_scene, scene_params, build_scene
from capture import trajectory, rig_params, covisibility

INF = 1e9


def build(out, n_views=24, baseline=(0.0, 0.0, 0.75), W=320, H=200,
          seed=0, n_az=900, n_el=48, wall_dist=38.0, scene_seed=None):
    """`scene_seed=None` reproduces the original single hand-built scene byte-for-byte.
    An integer draws an INDEPENDENT scene+rig, which is what makes scene the unit of
    replication instead of the training seed."""
    rng = np.random.default_rng(seed)
    sp = scene_params(scene_seed, wall_dist)
    rp = rig_params(scene_seed, baseline)
    sc = build_scene(sp)
    rigs = trajectory(n=n_views, rig=rp, W=W, H=H)
    os.makedirs(out, exist_ok=True)
    os.makedirs(f"{out}/rgb", exist_ok=True)

    # scene_params / rig_params are persisted so every downstream consumer rebuilds
    # THIS scene from the dataset instead of independently rebuilding the default one
    # (add_native.py used to do exactly that -- silent corruption once scenes vary).
    meta = {"n_views": n_views, "baseline": list(rp["baseline"]), "W": W, "H": H,
            "fx": float(rigs[0].f), "seed": seed, "scene_seed": scene_seed,
            "wall_dist_requested": float(wall_dist),
            "scene_params": sp, "rig_params": rp}
    per_view, stats = [], {"visible": 0, "occluded": 0}
    all_resid = []

    for i, r in enumerate(rigs):
        rgb, tcam, pid, _ = r.render(sc)
        np.save(f"{out}/rgb/{i:04d}.npy", rgb.astype(np.float32))
        np.save(f"{out}/rgb/{i:04d}_depth.npy", np.where(tcam < INF, tcam, 0).astype(np.float32))
        # 图元 id: 0=路面 1=远墙 2=车L 3=车R 4=细杆, -1=天空
        # 用于 rank-3 要求的分层评估(前景精度不变的前提下, 背景完整性与细结构保持)
        np.save(f"{out}/rgb/{i:04d}_pid.npy", pid.astype(np.int16))

        pts, ppid, _ = r.lidar_returns(sc, n_az=n_az, n_el=n_el)
        cv = covisibility(r, sc, pts)
        inim = cv["inim"]
        state = cv["state"][inim]
        uv = cv["uv"][inim]
        naive = cv["naive_depth"][inim]     # what naive projection asserts
        true_ = cv["true_depth"][inim]      # what the camera really sees
        occ = (state == "occluded")
        stats["visible"] += int((state == "visible").sum())
        stats["occluded"] += int(occ.sum())

        # residuals carried by the contaminated returns (always positive: "too far")
        resid = naive - true_
        if occ.any():
            all_resid.append(resid[occ])

        lab = {}
        lab["correct"] = true_.copy()                     # right label everywhere
        lab["wrong"] = np.where(occ, naive, true_)        # naive projection
        # --- coherence ablations: same affected pixels, same residual magnitudes ---
        ridx = np.where(occ)[0]
        rmag = resid[ridx]
        s_space = true_.copy()
        if len(ridx):
            s_space[ridx] = true_[ridx] + rng.permutation(rmag)
        lab["shuffled_space"] = s_space
        # 原名 shuffled_view 有误导: 它只是【同一视角内】有放回重采样残差,
        # 并未构造"保持空间结构、只破坏跨视角一致性"的干预 —— 本数据按视角独立生成,
        # 没有跨视角的点对应关系, 无法实现该干预。改名以免over-claim,
        # 并据此撤回"符号 vs 跨视角相干"的可加机制分解主张。
        s_res = true_.copy()
        if len(ridx):
            s_res[ridx] = true_[ridx] + rng.choice(rmag, size=len(ridx), replace=True)
        lab["resampled_mag"] = s_res
        # sign_flipped 旧实现会产生非正深度(b075 5112 个 / w24 1786 个),
        # 那不是物理可实现的测量噪声。改为: 只对翻转后仍为正的点翻转, 其余保持正号。
        s_sign = true_.copy()
        if len(ridx):
            sgn = rng.choice([-1.0, 1.0], size=len(ridx))
            cand = true_[ridx] + rmag * sgn
            bad = cand <= 0.1
            sgn[bad] = 1.0                     # 会导致非正深度的, 退回正向
            s_sign[ridx] = true_[ridx] + rmag * sgn
        lab["sign_flipped"] = s_sign

        # Physical-validity gate. An earlier version checked only isfinite(), which a
        # 1e9 no-hit sentinel passes; 3,588 such labels reached training and, through
        # the permutation pool, produced billion-metre targets in the magnitude-matched
        # controls. Depths must be positive and within sensor range, and the controls
        # must remain an exact permutation of the contaminated residuals.
        MAXD = 200.0
        for _name, _v in lab.items():
            if not np.all(np.isfinite(_v)):
                raise ValueError(f"{out} view {i}: non-finite label in {_name}")
            if _v.size and (_v.min() <= 0 or _v.max() > MAXD):
                raise ValueError(f"{out} view {i}: {_name} outside (0,{MAXD}] "
                                 f"-> [{_v.min():.4g},{_v.max():.4g}]")
        if len(ridx):
            a = np.sort(lab["shuffled_space"][ridx] - true_[ridx])
            b = np.sort(resid[ridx])
            if not np.allclose(a, b, atol=1e-6):
                raise ValueError(f"{out} view {i}: shuffled_space is not an exact "
                                 f"permutation of the residuals "
                                 f"(max dev {np.abs(a-b).max():.4g} m)")

        np.savez_compressed(
            f"{out}/sup_{i:04d}.npz",
            uv=uv.astype(np.float32), occ=occ,
            # LiDAR 回波的 3D 世界坐标 —— 用于【标准】初始化。
            # 先前 init_gaussians 用 camera 'correct' 深度反投影, 这既用了真值深度,
            # 又把被相机遮挡的回波放到了 camera-visible 前景位置, 是特权初始化。
            xyz=pts[inim].astype(np.float32),
            pid=ppid[inim].astype(np.int16),
            **{k: v.astype(np.float32) for k, v in lab.items()},
        )
        per_view.append({"view": i, "n_sup": int(inim.sum()), "n_occ": int(occ.sum())})

        c2w = np.eye(4); c2w[:3, :3] = r.R; c2w[:3, 3] = r.eye
        meta.setdefault("poses", []).append(c2w.tolist())
        meta.setdefault("lidar_origins", []).append(r.lidar_o.tolist())

    n = stats["visible"] + stats["occluded"]
    ar = np.concatenate(all_resid) if all_resid else np.zeros(0)
    meta["stats"] = {**stats, "occluded_frac": stats["occluded"] / max(n, 1),
                     "resid_median": float(np.median(ar)) if len(ar) else 0.0,
                     "resid_mean": float(ar.mean()) if len(ar) else 0.0,
                     "resid_min": float(ar.min()) if len(ar) else 0.0,
                     "resid_max": float(ar.max()) if len(ar) else 0.0}
    meta["per_view"] = per_view
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gateA/b075")
    ap.add_argument("--baseline", type=float, nargs=3, default=[0.0, 0.0, 0.75])
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wall-dist", type=float, default=38.0)
    ap.add_argument("--n-az", type=int, default=900)
    ap.add_argument("--n-el", type=int, default=48)
    ap.add_argument("--scene-seed", type=int, default=None,
                    help="None = original single scene; int = independent scene+rig")
    a = ap.parse_args()
    m = build(a.out, n_views=a.views, baseline=tuple(a.baseline), seed=a.seed,
              wall_dist=a.wall_dist, n_az=a.n_az, n_el=a.n_el, scene_seed=a.scene_seed)
    s = m["stats"]
    print(f"{a.out}: baseline={a.baseline}  views={a.views}")
    print(f"  supervision points: {s['visible']+s['occluded']:,}")
    print(f"  camera-occluded   : {s['occluded']:,} ({100*s['occluded_frac']:.2f}%)")
