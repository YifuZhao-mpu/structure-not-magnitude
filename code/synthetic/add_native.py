"""
GATE B 前置：为已有数据集补上 NATIVE LiDAR 射线监督。

Native 监督的含义：LiDAR 回波约束的是【它自己那条射线上】的表面位置，而不是
先投影成相机 z-depth。所以我们在 LiDAR 原点架一个虚拟相机，渲染 hit-distance
(沿射线距离)，再与真实 range = ||X - o_lidar|| 比较 —— 两者语义精确匹配。

这是 SplatAD/URF 一路的做法，也是 Claim 2 必须打败的强基线。注意 gsplat 的
"d"/"Ed" 模式渲染的正是 along-ray distance，不是 z-depth，所以无需改 CUDA。

虚拟 LiDAR 相机: W=400,H=200,f=200 -> 覆盖 yaw±45°, el±26.6°,
足以包住本场景 LiDAR 的 yaw±40°, el -22°~+8°。
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import default_scene, build_scene
from capture import trajectory, covisibility

LW, LH, LF = 400, 200, 200.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--wall-dist", type=float, default=38.0)
    ap.add_argument("--n-az", type=int, default=900)
    ap.add_argument("--n-el", type=int, default=48)
    a = ap.parse_args()

    meta = json.load(open(f"{a.data}/meta.json"))
    # Rebuild THIS dataset's scene from its OWN metadata. This script used to
    # independently rebuild default_scene(--wall-dist); once scenes vary per dataset
    # that silently computes native supervision against a DIFFERENT scene than the
    # one the camera supervision was generated from.
    if "scene_params" in meta:
        sc = build_scene(meta["scene_params"])
        rigs = trajectory(n=meta["n_views"], rig=meta["rig_params"],
                          W=meta["W"], H=meta["H"])
    else:                                   # datasets built before scene_params existed
        sc = default_scene(wall_dist=a.wall_dist)
        rigs = trajectory(n=meta["n_views"], baseline=tuple(meta["baseline"]),
                          W=meta["W"], H=meta["H"])

    tot = kept = 0
    for i, r in enumerate(rigs):
        pts, ppid, t = r.lidar_returns(sc, n_az=a.n_az, n_el=a.n_el)       # t = true range along LiDAR ray
        # 每个回波是否【被相机遮挡】——rank-3 要区分"丢弃"与"降级为 native 约束"
        cv = covisibility(r, sc, pts)
        occ_all = (cv["state"] == "occluded")
        # project into the virtual LiDAR camera (same orientation as the rig)
        v = pts - r.lidar_o
        cam = v @ r.R                             # into rig axes
        z = cam[:, 2]
        ok = z > 1e-3
        u = np.full(len(pts), -1.0); vv = np.full(len(pts), -1.0)
        u[ok] = cam[ok, 0] / z[ok] * LF + LW / 2
        vv[ok] = cam[ok, 1] / z[ok] * LF + LH / 2
        inim = ok & (u >= 0) & (u < LW) & (vv >= 0) & (vv < LH)
        # Same correspondence fix on the native side: several returns can land in one
        # virtual-LiDAR pixel with different ranges, and one rendered depth cannot
        # satisfy them. Use the range along that pixel's own centre ray.
        iu = np.clip(np.floor(u).astype(np.int64), 0, LW - 1)
        iv = np.clip(np.floor(vv).astype(np.int64), 0, LH - 1)
        xc = (iu + 0.5 - LW / 2) / LF
        yc = (iv + 0.5 - LH / 2) / LF
        from capture import _unit as _u
        d_pix = _u(np.stack([xc, yc, np.ones_like(xc)], 1) @ r.R.T)
        t_pix, _ = sc.trace(r.lidar_o, d_pix)
        # Where the sampled pixel-centre ray misses, there is no valid target. The
        # previous fallback substituted the range along a DIFFERENT ray, which is the
        # very defect the pixel-centre correction exists to remove. Drop instead.
        hit = np.isfinite(t_pix) & (t_pix < 1e8)
        rng = t_pix
        inim = inim & hit
        tot += len(pts); kept += int(inim.sum())
        np.savez_compressed(
            f"{a.data}/nat_{i:04d}.npz",
            uv=np.stack([u[inim], vv[inim]], 1).astype(np.float32),
            rng=rng[inim].astype(np.float32),
            occ=occ_all[inim],
            pid=ppid[inim].astype(np.int16))

    meta["lidar_cam"] = {"W": LW, "H": LH, "f": LF}
    meta["native_kept"] = kept
    json.dump(meta, open(f"{a.data}/meta.json", "w"), indent=1)
    print(f"{a.data}: native 监督 {kept:,}/{tot:,} 回波落入虚拟 LiDAR 相机 "
          f"({100*kept/tot:.1f}%)")


if __name__ == "__main__":
    main()
