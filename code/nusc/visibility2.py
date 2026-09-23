"""
可见性判据 v2 —— 基于【深度间隙】而非【到最近点的距离】。

v1 的系统性缺陷(与 Codex 对 Gate A 检测器的批评同源):
  "同一 cell 内的两点可能位于各自都可见的不同射线上"。
  一个 8px cell 在远处覆盖很大物理范围, 斜面(尤其路面)在 cell 内的深度跨度
  本就可能超过阈值 -> 点越密, 被误判为 occluded 的越多。实测累积 5 帧后
  occluded 率虚高到 30%+, 而单帧真实值约 0.6-0.9%。

v2 判据: 真实遮挡的物理特征是【前景与背景之间存在空隙】, 而斜面是连续的。
  对 cell 内排序后的深度序列找最大间隙 g:
    - g < gap_thr           -> 深度连续(斜面/同一表面)  -> 该 cell 内全部 visible
    - g >= gap_thr          -> 存在前后景分离 -> 间隙之后的点 occluded,
                               间隙附近 tol 范围内的点 unknown(证据不足)
  这直接对应遮挡语义, 且对点密度不敏感。
"""
import numpy as np
VIS, OCC, UNK = 0, 1, 2


def estimate_v2(pts_cam, K, W, H, cell=16, gap_rel=0.10, gap_abs=1.0,
                tol_rel=0.25, accum_cam=None, min_pts=3):
    z = pts_cam[:, 2]
    ok = z > 0.5
    uv = np.full((len(pts_cam), 2), -1.0)
    uv[ok] = (K @ pts_cam[ok].T).T[:, :2] / z[ok, None]
    u, v = uv[:, 0], uv[:, 1]
    inim = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    state = np.full(len(pts_cam), UNK, np.int8)
    if not inim.any():
        return state, uv, z, inim

    gw = int(np.ceil(W/cell)); gh = int(np.ceil(H/cell))
    # 证据点集(可含累积帧)
    src = accum_cam if accum_cam is not None else pts_cam
    sz = src[:, 2]; sok = sz > 0.5
    suv = np.full((len(src), 2), -1.0)
    suv[sok] = (K @ src[sok].T).T[:, :2] / sz[sok, None]
    sin = sok & (suv[:,0] >= 0) & (suv[:,0] < W) & (suv[:,1] >= 0) & (suv[:,1] < H)
    sc = (suv[sin,1]//cell).astype(np.int64)*gw + (suv[sin,0]//cell).astype(np.int64)
    sd = sz[sin]

    # 按 cell 分组排序, 找每个 cell 的最大深度间隙与其位置
    order = np.lexsort((sd, sc))
    sc_s, sd_s = sc[order], sd[order]
    # cell 边界
    newc = np.r_[True, sc_s[1:] != sc_s[:-1]]
    starts = np.where(newc)[0]
    ends = np.r_[starts[1:], len(sc_s)]
    front = np.full(gw*gh, np.inf)   # 前景层的最大深度(间隙前沿)
    has_gap = np.zeros(gw*gh, bool)
    nearest = np.full(gw*gh, np.inf)
    for s, e in zip(starts, ends):
        c = sc_s[s]; seg = sd_s[s:e]
        nearest[c] = seg[0]
        if len(seg) < min_pts:
            continue
        d = np.diff(seg)
        i = int(np.argmax(d))
        g = d[i]
        thr = max(gap_abs, gap_rel*seg[i])
        if g >= thr:
            has_gap[c] = True
            front[c] = seg[i]        # 间隙之前的最后一个点

    idx = np.where(inim)[0]
    c = (v[idx]//cell).astype(np.int64)*gw + (u[idx]//cell).astype(np.int64)
    zi = z[idx]
    st = np.full(len(idx), VIS, np.int8)
    hg = has_gap[c]
    # 只在有间隙的 cell 上计算, 否则 front=inf 会产生 inf-inf=nan
    if hg.any():
        fr = front[c][hg]
        tol = tol_rel*np.maximum(fr, 1e-3)
        zh = zi[hg]
        sub = np.full(len(fr), VIS, np.int8)
        sub[zh > fr + tol] = OCC
        sub[(zh > fr - tol) & (zh <= fr + tol)] = UNK
        st[hg] = sub
    state[idx] = st
    return state, uv, z, inim
