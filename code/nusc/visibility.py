"""
GATE C 的可见性估计器 —— 三态, 不是二值。

为什么必须三态 (Codex 的 identification 批评):
  真实数据上, "相机确实看到了这个回波" 与 "相机前方有个 LiDAR 没采到的遮挡物"
  在证据上无法区分。两个场景可以给出完全相同的累积点云与标注框, 而真值可见性不同。
  任何把 unknown 强行归入 visible 或 occluded 的做法, 都是在用缺失证据做判断。

  → 因此输出 visible / occluded / UNKNOWN 三态, 并且:
     - 训练时只对 visible 施加强监督
     - occluded 不监督
     - unknown 降权(或按消融完全不监督), 且必须单独报告其占比

三条独立判据 (不是"求共识", 而是各自负责不同的失效模式):
  A. 累积 LiDAR z-buffer  —— 能发现被采样到的遮挡物
  B. 标注框遮挡           —— 能发现被标注的前景物体(含 LiDAR 漏采的)
  C. 局部深度不连续       —— 边界邻域一律判 unknown(而非猜测)
"""
import os, sys, json, argparse
import numpy as np

NUSC = os.environ["NUSCENES_ROOT"]
VIS, OCC, UNK = 0, 1, 2


def load_pc(path):
    return np.fromfile(f"{NUSC}/{path}", dtype=np.float32).reshape(-1, 5)[:, :3]


def estimate(pts_cam, K, W, H, cell=8, rel=0.06, absd=0.4,
             disc_rel=0.15, accum_cam=None):
    """
    pts_cam : 本帧回波在【相机坐标系】下的坐标 [N,3]
    accum_cam: 累积多帧回波在同一相机系下的坐标 [M,3] (遮挡证据来源, 可为 None)
    返回 state[N] in {VIS,OCC,UNK}, uv[N,2], z[N]
    """
    z = pts_cam[:, 2]
    uv = np.full((len(pts_cam), 2), -1.0)
    ok = z > 0.5
    uv[ok] = (K @ pts_cam[ok].T).T[:, :2] / z[ok, None]
    u, v = uv[:, 0], uv[:, 1]
    inim = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    state = np.full(len(pts_cam), UNK, np.int8)
    if not inim.any():
        return state, uv, z, inim

    gw, gh = int(np.ceil(W/cell)), int(np.ceil(H/cell))
    src = accum_cam if accum_cam is not None else pts_cam[inim]
    sz = src[:, 2]
    sok = sz > 0.5
    suv = np.full((len(src), 2), -1.0)
    suv[sok] = (K @ src[sok].T).T[:, :2] / sz[sok, None]
    sin = sok & (suv[:, 0] >= 0) & (suv[:, 0] < W) & (suv[:, 1] >= 0) & (suv[:, 1] < H)
    scell = (suv[sin, 1]//cell).astype(int)*gw + (suv[sin, 0]//cell).astype(int)
    near = np.full(gw*gh, np.inf)
    np.minimum.at(near, scell, sz[sin])

    idx = np.where(inim)[0]
    c = (v[idx]//cell).astype(int)*gw + (u[idx]//cell).astype(int)
    zn = near[c]
    gap = z[idx] - zn
    # A: 有明确更近的表面 -> occluded
    occ = gap > np.maximum(absd, rel*z[idx])
    # C: 邻域深度不连续 -> unknown (不猜)
    grid = near.reshape(gh, gw)
    disc = np.zeros_like(grid, bool)
    fin = np.isfinite(grid)
    for dy, dx in ((0,1),(1,0),(0,-1),(-1,0)):
        sh = np.roll(np.roll(grid, dy, 0), dx, 1)
        shfin = np.roll(np.roll(fin, dy, 0), dx, 1)
        # BUG FIX: 空 cell 是 inf, inf-finite=inf 会让【任何邻接空格的 cell】
        # 都被判成深度不连续。稀疏 LiDAR 下几乎每个 cell 都邻接空格,
        # 结果是 ~100% unknown, 估计器完全失效。只在两侧都有观测时才比较。
        both = fin & shfin
        g1 = np.where(both, grid, 0.0); g2 = np.where(both, sh, 0.0)
        d = np.where(both, np.abs(g1-g2)/np.maximum(np.minimum(g1, g2), 1e-3), 0.0)
        disc |= d > disc_rel
    disc &= fin
    near_edge = disc.reshape(-1)[c]

    st = np.full(len(idx), VIS, np.int8)
    st[near_edge] = UNK          # 边界邻域: 证据不足, 不判
    st[occ] = OCC                # 明确被遮挡
    state[idx] = st
    return state, uv, z, inim


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene-0002")
    ap.add_argument("--index", default="data/nusc_index")
    ap.add_argument("--accum", type=int, default=5, help="累积 sweep 数(遮挡证据)")
    a = ap.parse_args()
    d = np.load(f"{a.index}/{a.scene}.npz", allow_pickle=True)
    lid, l2w = d["lid_paths"], d["l2w"]
    Ks, c2w, cam_ids, imgs = d["Ks"], d["c2w"], d["cam_ids"], d["img_paths"]
    W, H = 1600, 900
    tot = np.zeros(3, np.int64)
    n_f = 0
    for fi in range(0, len(lid), 5):          # 抽样若干帧
        pts_w = (l2w[fi][:3, :3] @ load_pc(lid[fi]).T + l2w[fi][:3, 3:4]).T
        acc = [pts_w]
        for k in range(max(0, fi-a.accum), min(len(lid), fi+a.accum+1)):
            if k != fi:
                acc.append((l2w[k][:3, :3] @ load_pc(lid[k]).T + l2w[k][:3, 3:4]).T)
        acc = np.concatenate(acc)
        for ci in range(6):
            j = fi*6+ci
            if j >= len(Ks): continue
            Tinv = np.linalg.inv(c2w[j])
            pc = (Tinv[:3, :3] @ pts_w.T + Tinv[:3, 3:4]).T
            ac = (Tinv[:3, :3] @ acc.T + Tinv[:3, 3:4]).T
            st, uv, z, m = estimate(pc, Ks[j], W, H, accum_cam=ac)
            for s in (VIS, OCC, UNK):
                tot[s] += int((st[m] == s).sum())
            n_f += 1
    n = tot.sum()
    print(f"{a.scene}: {n_f} 相机帧, {n:,} 个在图回波")
    for s, nm in [(VIS,"visible"),(OCC,"occluded"),(UNK,"UNKNOWN")]:
        print(f"  {nm:9}: {tot[s]:10,}  {100*tot[s]/n:5.2f}%")
    print(f"\n  → unknown 占比必须单独报告; 把它归入任一侧都是用缺失证据做判断")
