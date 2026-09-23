"""
GATE B: native LiDAR 射线监督 vs native + oracle 相机端转移。

问题（Claim 2 的前置门槛）：native 监督已经正确用上了每一个回波 —— 约束它自己
射线上的表面位置。那么"额外把回波转移成相机像素监督"还有没有增量？

条件：
  native          仅 LiDAR 射线监督 (hit-distance vs 真实 range)
  native_oracle   native + 仅对【真值共视】回波的相机端监督
  native_naive    native + 对【全部】回波的相机端监督（含被遮挡的错误标签）
  native_x2       native, 但 LiDAR 监督权重加倍 —— 排除"只是监督更强/预算更大"

KILL: 若 native_oracle 打不过 native -> Claim 2 终止，不必再建可见性估计器。
"""
import os, sys, json, math, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gateA import load, init_gaussians

INF = 1e9


PID = {"road": 0, "wall": 1, "carL": 2, "carR": 3, "pole": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cond", required=True,
                    choices=["native", "native_oracle", "native_naive", "native_x2",
                             "native_los", "native_los_oracle", "native_los_naive"])
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--nat-w", type=float, default=0.5)
    ap.add_argument("--cam-w", type=float, default=0.5)
    ap.add_argument("--los-w", type=float, default=0.5)
    ap.add_argument("--los-margin", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    from gsplat.rendering import rasterization

    meta, rgb, dep, sup = load(a.data)
    n = meta["n_views"]; W, H = meta["W"], meta["H"]; f = meta["fx"]
    LC = meta["lidar_cam"]; LW, LH, LF = LC["W"], LC["H"], LC["f"]
    test = list(range(0, n, 4)); train = [i for i in range(n) if i not in test]

    K = torch.tensor([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], dtype=torch.float32, device=dev)[None]
    LK = torch.tensor([[LF, 0, LW/2], [0, LF, LH/2], [0, 0, 1]], dtype=torch.float32, device=dev)[None]
    poses = torch.tensor(np.array(meta["poses"]), dtype=torch.float32, device=dev)
    viewmats = torch.linalg.inv(poses)
    # LiDAR 相机: 同朝向, 原点在 lidar_o
    lp = poses.clone()
    lp[:, :3, 3] = torch.tensor(np.array(meta["lidar_origins"]), dtype=torch.float32, device=dev)
    lviewmats = torch.linalg.inv(lp)
    imgs = torch.tensor(rgb, dtype=torch.float32, device=dev)
    gtd = torch.tensor(dep, dtype=torch.float32, device=dev)

    nat = []
    for i in range(n):
        d = np.load(f"{a.data}/nat_{i:04d}.npz")
        nat.append(dict(uv=torch.tensor(d["uv"], device=dev),
                        rng=torch.tensor(d["rng"], device=dev)))
    cam = []
    for i in range(n):
        s = sup[i]
        occ = torch.tensor(s["occ"], device=dev)
        cam.append(dict(uv=torch.tensor(s["uv"], dtype=torch.float32, device=dev),
                        correct=torch.tensor(s["correct"], dtype=torch.float32, device=dev),
                        wrong=torch.tensor(s["wrong"], dtype=torch.float32, device=dev),
                        occ=occ))

    # Ground-truth sky is a non-black colour; without an explicit background the
    # renderer composites against black, so the RGB loss can only be reduced by
    # putting opacity along analytically EMPTY rays (sky is ~25% of pixels). That is
    # a competing explanation for any appearance/geometry trade-off, so supply it.
    BG = torch.tensor([[0.45, 0.55, 0.72]], dtype=torch.float32, device=dev)
    means, scales, quats, opac, cols = init_gaussians(meta, sup, dev, seed=a.seed, train_idx=train)
    opt = torch.optim.Adam([
        {"params": [means], "lr": 1.6e-3}, {"params": [scales], "lr": 5e-3},
        {"params": [quats], "lr": 1e-3}, {"params": [opac], "lr": 5e-2},
        {"params": [cols], "lr": 2.5e-3}])

    def render(vm, Ks, w, h, mode, eval3d=False):
        bg = BG if mode.startswith("RGB") else None
        return rasterization(means=means, quats=quats/quats.norm(dim=-1, keepdim=True),
                             scales=scales.exp(), opacities=torch.sigmoid(opac), colors=cols,
                             viewmats=vm, Ks=Ks, width=w, height=h, render_mode=mode,
                             packed=False, with_eval3d=eval3d, backgrounds=bg)

    nat_w = a.nat_w * (2.0 if a.cond == "native_x2" else 1.0)
    t0 = time.time()
    for it in range(a.iters):
        i = train[np.random.randint(len(train))]
        out, _, _ = render(viewmats[i:i+1], K, W, H, "RGB-Ed", True)
        img = out[0, ..., :3]; d_pred = out[0, ..., 3]
        loss = F.l1_loss(img, imgs[i])

        # --- native LiDAR ray supervision: hit-distance at the LiDAR origin ---
        lo, _, _ = render(lviewmats[i:i+1], LK, LW, LH, "Ed", eval3d=True)
        hd = lo[0, ..., 0]
        s = nat[i]
        u = s["uv"][:, 0].long().clamp(0, LW-1); v = s["uv"][:, 1].long().clamp(0, LH-1)
        pred_rng = hd[v, u]
        loss = loss + nat_w * (pred_rng - s["rng"]).abs().mean()

        # --- line-of-sight / free-space term (SplatAD & URF use this; an Ed+range
        #     objective alone is an UNDERCONSTRAINED baseline and beating it proves
        #     nothing). Penalise any surface appearing BEFORE the true return, i.e.
        #     spurious occupancy along the LiDAR path.
        if a.cond.startswith("native_los"):
            # 真 LOS: 惩罚【回波之前】的累积不透明度, 而非对期望距离再做 hinge。
            #
            # 旧实现 relu(R - m - E[d]) 是无效的 (Codex 反例, 已数值复核):
            #   R=10m, 权重在 5m/15m 各半 -> E[d]=10=R
            #   => |E[d]-R| = 0 且 relu(R-m-E[d]) = 0, 而回波前占据 50%。
            #   它只是对同一个标量期望再加一次非对称惩罚, 不含自由空间信息。
            #
            # 真判据基于 R 之前的不透明度。gsplat 的 far_plane 是标量, 故按 range
            # 分桶, 每次迭代随机取一桶, 以 far_plane = lo - margin 渲染 alpha:
            # 该 alpha 即"桶内射线在其回波之前累积到的不透明度", 应趋近 0。
            rng_all = s["rng"]
            lo_all = float(rng_all.min().item())
            hi_all = float(rng_all.max().item())
            NBUCKET = 6          # 注意: 不能用 K, 会遮蔽外层的相机内参张量 K
            edges = np.linspace(lo_all, hi_all, NBUCKET + 1)
            k = np.random.randint(NBUCKET)
            blo, bhi = edges[k], edges[k + 1]
            sel = (rng_all >= blo) & (rng_all < bhi)
            fp = float(blo - a.los_margin)
            if sel.any() and fp > 0.5:
                _, al, _ = rasterization(
                    means=means, quats=quats/quats.norm(dim=-1, keepdim=True),
                    scales=scales.exp(), opacities=torch.sigmoid(opac), colors=cols,
                    viewmats=lviewmats[i:i+1], Ks=LK, width=LW, height=LH,
                    render_mode="RGB", packed=False, far_plane=fp)
                a_front = al[0, ..., 0][v[sel], u[sel]]
                loss = loss + a.los_w * a_front.mean()

        # --- optional camera-side transfer ---
        if a.cond in ("native_oracle", "native_naive",
                      "native_los_oracle", "native_los_naive"):
            c = cam[i]
            u2 = c["uv"][:, 0].long().clamp(0, W-1); v2 = c["uv"][:, 1].long().clamp(0, H-1)
            if a.cond.endswith("oracle"):
                m = ~c["occ"]                     # 只转移真值共视的回波
                tgt, u2, v2 = c["correct"][m], u2[m], v2[m]
            else:
                tgt = c["wrong"]                  # 全部转移, 含错误标签
            if len(tgt):
                loss = loss + a.cam_w * (d_pred[v2, u2] - tgt).abs().mean()

        opt.zero_grad(); loss.backward(); opt.step()

    res = {"cond": a.cond, "data": a.data, "seed": a.seed, "iters": a.iters,
           "nat_w": a.nat_w, "cam_w": a.cam_w, "los_w": a.los_w,
           "train_time_s": round(time.time()-t0, 1)}
    with torch.no_grad():
        # loaded HERE, after training: allocating it earlier perturbs the CUDA
        # allocator layout, and gsplat's atomicAdd backward is accumulation-order
        # sensitive, which silently shifts the very numbers this metric reports on.
        gpid = torch.tensor(np.stack([np.load(f"{a.data}/rgb/{i:04d}_pid.npy")
                                      for i in range(n)]), device=dev)
        errs, sil, ps = [], [], []
        strat = {"wall": [], "pole": [], "car": []}
        for i in test:
            out, _, _ = render(viewmats[i:i+1], K, W, H, "RGB-Ed", True)
            img = out[0, ..., :3]; d = out[0, ..., 3]
            g = gtd[i]; m = g > 0; e = (d-g).abs()
            errs.append(e[m].mean().item())
            pm = gpid[i]
            for nm, selm in [("wall", pm == PID["wall"]), ("pole", pm == PID["pole"]),
                             ("car", (pm == PID["carL"]) | (pm == PID["carR"]))]:
                m2 = m & selm
                if m2.any():
                    strat[nm].append(e[m2].mean().item())
            # Boundary band from ANALYTIC primitive identity, not a depth-step
            # threshold: on a receding ground plane perspective alone exceeds 0.5 m
            # between adjacent rows, so the old band was ~81% non-boundary pixels.
            step = torch.zeros_like(g, dtype=torch.bool)
            step[:, 1:] |= pm[:, 1:] != pm[:, :-1]
            step[1:] |= pm[1:] != pm[:-1]
            band = F.max_pool2d(step.float()[None, None], 9, 1, 4)[0, 0] > 0
            mm = m & band
            if mm.any(): sil.append(e[mm].mean().item())
            ps.append(-10*math.log10(max(((img-imgs[i])**2).mean().item(), 1e-10)))
        res.update(depth_mae=float(np.mean(errs)),
                   depth_mae_silhouette=float(np.mean(sil)), psnr=float(np.mean(ps)),
                   **{f"mae_{k}": (float(np.mean(v)) if v else None)
                      for k, v in strat.items()})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
