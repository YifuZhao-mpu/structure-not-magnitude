"""
GATE C 条件表所需的【竞争过滤器】。Codex 要求: 缺 RePLAy 等已有强过滤器,
"无法证明方法超出已知过滤"。

全部在相机坐标系下工作, 输入 LiDAR 回波, 输出 accept 掩码。
先用解析真值审核各自的精度/召回, 确认实现正确, 再进入训练。

- zbuffer_epipolar : RePLAy 所依据的对极/视线几何的【离散近似】(非其解析解的完整复现)。
- hpr_katz    : Katz et al. SIGGRAPH 2007 的 Hidden Point Removal, 球面翻转 + 凸包。
                LidaRF 已将其作为基线。
- random_drop : 随机丢弃, 匹配接受数 —— 排除"只是减少了监督量"。
- edge_drop   : 按深度不连续丢弃, 匹配接受数 —— 排除"只是避开了难像素"。
"""
import numpy as np
from scipy.spatial import cKDTree, ConvexHull


def zbuffer_epipolar(pts_cam, K, W, H, lidar_origin_cam, rel=0.06, absd=0.4,
                     pix_tol=1.0):
    """
    像素级 z-buffer 遮挡判定(RePLAy 所依据的几何, 但为离散实现)。

    ⚠️ 命名诚实性: 这【不是】RePLAy 的完整复现。RePLAy(2024) 给出的是
    虚拟 LiDAR 相机与 RGB 相机之间对极遮挡的【解析解】; 此处是其所依据的
    同一几何关系的离散近似。先前版本沿对极线搜索"更近的点"是错误的 ——
    投影落在对极线上并不蕴含遮挡, 遮挡要求候选点投影到【相同像素】且更近;
    对极约束的作用是高效定位候选, 而非替代视线判定。该错误导致接受率仅 6.77%
    (丢弃 93% 的点)。
    """
    z = pts_cam[:, 2]
    ok = z > 0.5
    uv = np.full((len(pts_cam), 2), -1.0)
    uv[ok] = (K @ pts_cam[ok].T).T[:, :2] / z[ok, None]
    inim = ok & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    acc = np.ones(len(pts_cam), bool)
    if not inim.any():
        return acc
    idx = np.where(inim)[0]
    tree = cKDTree(uv[inim])
    zi = z[inim]
    nb = tree.query_ball_point(uv[inim], r=pix_tol)
    for j, lst in enumerate(nb):
        if not lst:
            continue
        thr = max(absd, rel * zi[j])
        if (zi[lst] < zi[j] - thr).any():
            acc[idx[j]] = False
    return acc


def hpr_katz(pts_cam, K, W, H, param=3.0):
    """Katz et al. 2007: 球面翻转后取凸包, 凸包上的点即为可见。"""
    acc = np.zeros(len(pts_cam), bool)
    P = pts_cam
    r = np.linalg.norm(P, axis=1)
    ok = r > 1e-6
    if ok.sum() < 5:
        return np.ones(len(pts_cam), bool)
    Pf = P[ok]; rf = r[ok]
    R = rf.max() * (10 ** param)
    flipped = Pf + 2 * (R - rf)[:, None] * (Pf / rf[:, None])
    try:
        hull = ConvexHull(np.vstack([flipped, np.zeros((1, 3))]))
        vis = np.array([v for v in hull.vertices if v < len(flipped)])
    except Exception:
        return np.ones(len(pts_cam), bool)
    sub = np.zeros(len(Pf), bool); sub[vis] = True
    acc[np.where(ok)[0]] = sub
    return acc


def random_drop(n, n_accept, seed=0):
    """随机接受 n_accept 个 —— 匹配接受数的对照。"""
    rng = np.random.default_rng(seed)
    acc = np.zeros(n, bool)
    acc[rng.choice(n, size=min(n_accept, n), replace=False)] = True
    return acc


def edge_drop(pts_cam, K, W, H, n_accept, cell=8):
    """
    按局部深度不连续排序, 丢弃最"靠近边缘"的点直到匹配接受数。
    用于排除"只是避开了难像素"这一平凡解释。
    """
    z = pts_cam[:, 2]
    ok = z > 0.5
    uv = np.full((len(pts_cam), 2), -1.0)
    uv[ok] = (K @ pts_cam[ok].T).T[:, :2] / z[ok, None]
    inim = ok & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    score = np.zeros(len(pts_cam))
    if inim.any():
        tree = cKDTree(uv[inim]); zi = z[inim]; idx = np.where(inim)[0]
        nb = tree.query_ball_point(uv[inim], r=cell)
        for j, lst in enumerate(nb):
            if len(lst) > 1:
                score[idx[j]] = np.abs(zi[lst] - zi[j]).max()
    order = np.argsort(score)               # 深度变化小的先接受
    acc = np.zeros(len(pts_cam), bool)
    acc[order[:n_accept]] = True
    return acc
