#!/usr/bin/env python3
"""
R-4 real-data corroboration: build nuScenes scenes in the same layout the synthetic
GATE B trainer consumes, so the two experiments share a trainer and are comparable.

Only one synthetic claim is testable here -- that transferring returns to the camera
WITHOUT a visibility filter harms a native-ray baseline -- because that comparison needs
no ground-truth visibility. Frozen design: paper-stage/PRE_ANALYSIS_PLAN_R4.md.

Layout per scene (mirrors data/multiscene_v3/):
  meta.json                  poses, intrinsics, virtual-LiDAR camera, train/test split
  rgb/{i:04d}.npy            CAM_FRONT image, float32 HWC in [0,1]
  nat_{i:04d}.npz            uv, rng, xyz -- native supervision + world points (init)
  cam_{i:04d}.npz            uv, d     -- every in-image return transferred as camera depth
  eval_{i:04d}.npz           uv, rng, fg -- held-out frames only: geometric eval + box mask
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image

NUSC = os.environ.get("NUSCENES_ROOT", "")
if not os.path.isdir(NUSC):
    raise SystemExit("set NUSCENES_ROOT to an official nuScenes installation "
                     f"(tried {NUSC!r}); these scenes are rebuilt from it, not shipped")
IDX = "data/nusc_index"
CAM_FRONT = 0
# virtual LiDAR camera: pinhole at the LiDAR origin, +-45 deg H, +-25 deg V.
# A single pinhole cannot cover 360 deg; returns outside this frustum take no part in
# native supervision, which the paper states as a scope limit.
LW, LH, LF = 512, 239, 256.0
RMIN, RMAX = 1.0, 80.0          # drop ego-vehicle returns; match the synthetic range cap
# camera axes (right, down, forward) expressed in the LiDAR frame (x fwd, y left, z up)
R_CAM2LID = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])


def in_boxes(pts_w, centres, sizes, rots):
    """True where a world point falls inside any annotated 3D box (nuScenes w,l,h)."""
    m = np.zeros(len(pts_w), bool)
    for c, s, q in zip(centres, sizes, rots):
        R = quat_to_R(q)
        local = (pts_w - c) @ R                      # world -> box frame
        half = np.array([s[1], s[0], s[2]]) / 2.0    # (w,l,h) -> (x=l, y=w, z=h)
        m |= np.all(np.abs(local) <= half, axis=1)
    return m


def build(scene_npz, out, scale=4, stride=4):
    d = np.load(scene_npz, allow_pickle=True)
    n = len(d["lid_paths"])
    os.makedirs(f"{out}/rgb", exist_ok=True)
    test = list(range(0, n, stride))
    train = [i for i in range(n) if i not in test]
    bo = np.concatenate([[0], np.cumsum(d["box_nper"])])

    meta = {"n_frames": int(n), "scale": scale, "train": train, "test": test,
            "lidar_cam": {"W": LW, "H": LH, "f": LF},
            "scene": str(d["scene"]), "poses": [], "lidar_origins": [], "lidar_poses": []}
    stats = {"nat": 0, "cam": 0, "pts": 0}

    for i in range(n):
        p = np.fromfile(os.path.join(NUSC, str(d["lid_paths"][i])), dtype=np.float32)
        p = p.reshape(-1, 5)[:, :3]
        rng = np.linalg.norm(p, axis=1)
        keep = (rng > RMIN) & (rng < RMAX)
        p, rng = p[keep], rng[keep]
        stats["pts"] += len(p)

        l2w = d["l2w"][i]
        w = (l2w[:3, :3] @ p.T).T + l2w[:3, 3]

        # ---- native: project sensor-frame points into the virtual LiDAR camera ----
        pl = p @ R_CAM2LID                      # == R_CAM2LID.T @ p, per point
        zl = pl[:, 2]
        okl = zl > 0.1
        uvl = np.full((len(p), 2), -1.0)
        uvl[okl] = pl[okl, :2] / zl[okl, None] * LF + np.array([LW / 2, LH / 2])
        nat = okl & (uvl[:, 0] >= 0) & (uvl[:, 0] < LW) & (uvl[:, 1] >= 0) & (uvl[:, 1] < LH)

        # ---- camera transfer: every in-image return, labelled with its own distance ----
        ci = i * 6 + CAM_FRONT
        c2w = d["c2w"][ci]
        K = d["Ks"][ci].copy() / scale
        K[2, 2] = 1.0
        W, H = 1600 // scale, 900 // scale
        w2c = np.linalg.inv(c2w)
        pcam = (w2c[:3, :3] @ w.T).T + w2c[:3, 3]
        zc = pcam[:, 2]
        okc = zc > 0.1
        uvc = np.full((len(p), 2), -1.0)
        uvc[okc] = (pcam[okc, :2] / zc[okc, None]) * np.array([K[0, 0], K[1, 1]]) \
                   + np.array([K[0, 2], K[1, 2]])
        cam = okc & (uvc[:, 0] >= 0) & (uvc[:, 0] < W) & (uvc[:, 1] >= 0) & (uvc[:, 1] < H)
        dcam = np.linalg.norm(w - c2w[:3, 3], axis=1)     # along-ray distance, matches RGB-Ed

        # ---- physical-validity gate (a 1e9 sentinel is finite; check the range) ----
        for nm, v in (("rng", rng[nat]), ("dcam", dcam[cam])):
            if v.size and (not np.all(np.isfinite(v)) or v.min() <= 0 or v.max() > RMAX + 1):
                raise ValueError(f"{out} frame {i}: {nm} outside (0,{RMAX}]")

        np.savez_compressed(f"{out}/nat_{i:04d}.npz",
                            uv=uvl[nat].astype(np.float32), rng=rng[nat].astype(np.float32),
                            xyz=w[nat].astype(np.float32))   # world points, for initialisation
        np.savez_compressed(f"{out}/cam_{i:04d}.npz",
                            uv=uvc[cam].astype(np.float32), d=dcam[cam].astype(np.float32),
                            xyz=w[cam].astype(np.float32))
        stats["nat"] += int(nat.sum()); stats["cam"] += int(cam.sum())

        if i in test:
            fg = in_boxes(w[nat], d["box_centers"][bo[i]:bo[i+1]],
                          d["box_sizes"][bo[i]:bo[i+1]], d["box_rots"][bo[i]:bo[i+1]])
            np.savez_compressed(f"{out}/eval_{i:04d}.npz",
                                uv=uvl[nat].astype(np.float32),
                                rng=rng[nat].astype(np.float32), fg=fg)

        im = Image.open(os.path.join(NUSC, str(d["img_paths"][ci]))).resize((W, H), Image.BILINEAR)
        np.save(f"{out}/rgb/{i:04d}.npy", (np.asarray(im, np.float32) / 255.0))

        vcam = l2w.copy(); vcam[:3, :3] = l2w[:3, :3] @ R_CAM2LID
        meta["poses"].append(c2w.tolist())
        meta["lidar_poses"].append(vcam.tolist())
        meta["lidar_origins"].append(l2w[:3, 3].tolist())
        if i == 0:
            meta.update(W=W, H=H, fx=float(K[0, 0]), fy=float(K[1, 1]),
                        cx=float(K[0, 2]), cy=float(K[1, 2]))

    meta["stats"] = {**stats,
                     "nat_frac": stats["nat"] / max(stats["pts"], 1),
                     "cam_frac": stats["cam"] / max(stats["pts"], 1)}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--out", default="data/nusc_r4")
    a = ap.parse_args()
    import glob
    files = sorted(glob.glob(f"{IDX}/*.npz"))[:a.scenes]
    print(f"{'scene':14}{'frames':>7}{'native%':>9}{'camera%':>9}{'pts/sweep':>11}")
    for f in files:
        name = os.path.basename(f)[:-4]
        o = f"{a.out}/{name}"
        if os.path.exists(f"{o}/meta.json"):
            print(f"{name:14}  (skip)"); continue
        m = build(f, o)
        s = m["stats"]
        print(f"{name:14}{m['n_frames']:>7}{100*s['nat_frac']:>9.1f}{100*s['cam_frac']:>9.1f}"
              f"{s['pts']//m['n_frames']:>11,}")
