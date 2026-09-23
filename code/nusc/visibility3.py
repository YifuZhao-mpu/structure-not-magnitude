"""
可见性判据 v3 —— 射线级, 不再用像素 cell。

v1/v2 共同的根本缺陷(Codex 用反例证伪):
  遮挡是【射线级】关系, 而 cell 只是像素网格。同一 cell 内的点可能位于
  完全不同的射线上、各自都可见。v2 的"深度间隙"判据因此在
  "同 cell 两个空间分离的表面"上给出 UNK/UNK/OCC/OCC —— 四个点其实全部可见。
  且证据不足时 v2 默认落回 VIS。

v3 定义: 点 Y 遮挡点 X  <=>  Y 落在 X 的视线方向附近(角度容差内) 且 Y 显著更近。
  用单位方向向量上的 KD-tree 做角度邻域查询, 远处自动获得更严格的空间约束。

三态由【证据量】决定, 不再有"默认可见":
  - 邻域内存在显著更近的点            -> OCCLUDED (正面证据)
  - 邻域内点数 >= min_evidence 且无更近点 -> VISIBLE (有一定证据表明无遮挡)
  - 否则                              -> UNKNOWN (证据不足, 不猜)

稀疏 LiDAR 本质上无法证明"无遮挡", 所以 VISIBLE 永远是有条件的判断,
UNKNOWN 比例必须单独报告。
"""
import numpy as np
from scipy.spatial import cKDTree

VIS, OCC, UNK = 0, 1, 2


def estimate_v3(pts_cam, K, W, H, pix_tol=1.0, rel=0.06, absd=0.4,
                min_evidence=3, accum_cam=None, **_):
    """
    pix_tol: 角度容差以【像素】给出, 内部按焦距转成角度。
    固定角度容差无法适配不同相机: f=100 时相邻像素角距 0.573 度,
    f=1266(nuScenes) 时仅 0.045 度 —— 同一个 0.35 度阈值在前者找不到任何邻居,
    在后者却覆盖 7.7 像素。容差必须反映采样密度。

    ★ 容差选取原则(对照解析真值实测得出):
      角度容差必须【显著小于 LiDAR 仰角分辨率】, 同时不小于方位角分辨率。
      一旦容差逼近仰角间距, 相邻 beam 打在斜面(如路面)不同距离处就会被
      误判为遮挡 —— 实测遮挡精度在 0.251度->0.501度 之间崩塌 20 倍
      (40.5% -> 2.0%), 而合成 LiDAR 仰角分辨率正是 0.625 度。
      合成场景(方位 0.089度/仰角 0.625度, f=228.5): pix_tol=1.0 即 0.251 度。
      nuScenes(方位 ~0.33度=7.3px / 仰角 ~1.25度=27px, f~1266): pix_tol 取 10-15。
    """
    z = pts_cam[:, 2]
    ok = z > 0.5
    uv = np.full((len(pts_cam), 2), -1.0)
    uv[ok] = (K @ pts_cam[ok].T).T[:, :2] / z[ok, None]
    u, v = uv[:, 0], uv[:, 1]
    inim = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    state = np.full(len(pts_cam), UNK, np.int8)
    if not inim.any():
        return state, uv, z, inim

    src = accum_cam if accum_cam is not None else pts_cam
    sr = np.linalg.norm(src, axis=1)
    sok = sr > 0.5
    src, sr = src[sok], sr[sok]
    if len(src) == 0:
        return state, uv, z, inim
    sdir = src / sr[:, None]                      # 单位方向
    tree = cKDTree(sdir)

    idx = np.where(inim)[0]
    q = pts_cam[idx]
    qr = np.linalg.norm(q, axis=1)
    qdir = q / qr[:, None]
    # 角度容差 -> 单位球上的弦长
    ang_tol = np.arctan(pix_tol / float(K[0, 0]))    # 像素 -> 角度
    chord = 2.0 * np.sin(ang_tol / 2.0)
    nbrs = tree.query_ball_point(qdir, r=chord)

    st = np.full(len(idx), UNK, np.int8)
    for i, nb in enumerate(nbrs):
        if not nb:
            continue
        d = sr[nb]
        thr = max(absd, rel * qr[i])
        closer = d < qr[i] - thr
        if closer.any():
            st[i] = OCC                    # 正面证据: 视线上有更近的表面
        elif len(nb) >= min_evidence:
            st[i] = VIS                    # 邻域有足够采样且无更近点
        # else 保持 UNK: 证据不足
    state[idx] = st
    return state, uv, z, inim
