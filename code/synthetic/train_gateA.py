"""
GATE A trainer: 3DGS on the analytic synthetic scene, one label condition per run.

The comparison that matters is ACROSS conditions at a fixed seed and fixed everything
else. Only the depth-label VALUES differ between conditions; the supervised pixel set,
the count, the schedule and the loss weight are identical. That is what isolates
"structured wrongness" from "noise" and from "less supervision".

Evaluation is against the analytic scene, not against a reconstruction, so there is
no circularity: held-out-view depth error, error restricted to silhouette bands, and
pole retention are all measured against closed-form geometry.
"""
import os, sys, json, math, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import default_scene
from capture import trajectory

INF = 1e9
PID = {"road": 0, "wall": 1, "carL": 2, "carR": 3, "pole": 4}


def load(ds):
    meta = json.load(open(f"{ds}/meta.json"))
    n = meta["n_views"]
    rgb = np.stack([np.load(f"{ds}/rgb/{i:04d}.npy") for i in range(n)])
    dep = np.stack([np.load(f"{ds}/rgb/{i:04d}_depth.npy") for i in range(n)])
    sup = [np.load(f"{ds}/sup_{i:04d}.npz") for i in range(n)]
    return meta, rgb, dep, sup


def init_gaussians(meta, sup, device, n_init=60000, seed=0, train_idx=None, adaptive_scale=False):
    """
    标准 LiDAR 初始化: 直接用回波的 3D 坐标, 且只用【训练帧】。

    先前实现是特权初始化, 有三个问题(Codex 指出, 已复核):
      1) 用 camera 'correct' 真值深度反投影 —— 真实系统拿不到
      2) 遍历全部 sup, 含【测试视角】 —— 泄漏测试几何
      3) 被相机遮挡的回波被放到 camera-visible 前景位置 —— 抹掉了要研究的错配
    因此它不能被称作"保守下界"。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    idxs = range(len(sup)) if train_idx is None else train_idx
    pts = [sup[i]["xyz"] for i in idxs]
    pts = np.concatenate(pts)
    sel = torch.randperm(len(pts), generator=g)[:n_init].numpy()
    means = torch.tensor(pts[sel], dtype=torch.float32, device=device)
    N = len(means)
    if adaptive_scale:
        # 3DGS 原文做法: 初始尺度按最近邻距离设定。
        # 固定 0.06 会使不同点数的比较失去公平性 —— 实测 229 点时它是最近邻距的
        # 0.10 倍(高斯过小), 60000 点时是 1.85 倍(严重重叠), 相差 18 倍。
        from scipy.spatial import cKDTree
        P = pts[sel]
        dd, _ = cKDTree(P).query(P, k=2)
        nn = np.clip(dd[:, 1], 1e-3, None)
        scales = torch.tensor(np.log(nn * 0.5)[:, None].repeat(3, 1),
                              dtype=torch.float32, device=device)
    else:
        scales = torch.full((N, 3), math.log(0.06), device=device)
    quats = torch.zeros(N, 4, device=device); quats[:, 0] = 1.0
    opac = torch.full((N,), 0.0, device=device)
    cols = torch.full((N, 3), 0.5, device=device)
    for t in (means, scales, quats, opac, cols):
        t.requires_grad_(True)
    return means, scales, quats, opac, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cond", required=True,
                    choices=["none", "correct", "wrong", "shuffled_space",
                             "resampled_mag", "sign_flipped"])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--depth-w", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    # §5.1 mechanism probe. OFF by default: it runs strictly AFTER every existing
    # metric is computed, so results with and without it are bit-identical.
    ap.add_argument("--probe", default="", metavar="DIR",
                    help="write depth-sliced opacity probe to DIR (see PRE_ANALYSIS_PLAN_M.md)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    from gsplat.rendering import rasterization

    meta, rgb, dep, sup = load(a.data)
    n = meta["n_views"]; W, H = meta["W"], meta["H"]; f = meta["fx"]
    # held-out every 4th view
    test = list(range(0, n, 4)); train = [i for i in range(n) if i not in test]

    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]],
                     dtype=torch.float32, device=dev)[None]
    poses = torch.tensor(np.array(meta["poses"]), dtype=torch.float32, device=dev)
    viewmats = torch.linalg.inv(poses)
    imgs = torch.tensor(rgb, dtype=torch.float32, device=dev)
    gtd = torch.tensor(dep, dtype=torch.float32, device=dev)

    # Ground-truth sky is a non-black colour; without an explicit background the
    # renderer composites against black, so the RGB loss can only be reduced by
    # putting opacity along analytically EMPTY rays (sky is ~25% of pixels). That is
    # a competing explanation for any appearance/geometry trade-off, so supply it.
    BG = torch.tensor([[0.45, 0.55, 0.72]], dtype=torch.float32, device=dev)
    means, scales, quats, opac, cols = init_gaussians(meta, sup, dev, seed=a.seed, train_idx=train)
    opt = torch.optim.Adam([
        {"params": [means], "lr": 1.6e-4 * 10}, {"params": [scales], "lr": 5e-3},
        {"params": [quats], "lr": 1e-3}, {"params": [opac], "lr": 5e-2},
        {"params": [cols], "lr": 2.5e-3}])

    # supervision tensors per view
    sup_t = []
    for i in range(n):
        s = sup[i]
        sup_t.append(dict(
            uv=torch.tensor(s["uv"], dtype=torch.float32, device=dev),
            d=torch.tensor(s[a.cond] if a.cond != "none" else s["correct"],
                           dtype=torch.float32, device=dev),
            occ=torch.tensor(s["occ"], device=dev)))

    t0 = time.time()
    for it in range(a.iters):
        i = train[np.random.randint(len(train))]
        out, alpha, info = rasterization(
            means=means, quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=scales.exp(), opacities=torch.sigmoid(opac), colors=cols,
            viewmats=viewmats[i:i+1], Ks=K, width=W, height=H,
            render_mode="RGB-Ed", packed=False, with_eval3d=True,
                backgrounds=BG)
        img = out[0, ..., :3]; d_pred = out[0, ..., 3]
        loss = F.l1_loss(img, imgs[i])
        if a.cond != "none":
            s = sup_t[i]
            u = s["uv"][:, 0].long().clamp(0, W - 1)
            v = s["uv"][:, 1].long().clamp(0, H - 1)
            pred = d_pred[v, u]
            loss = loss + a.depth_w * (pred - s["d"]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()

    # ---------------- evaluation against ANALYTIC truth ----------------
    res = {"cond": a.cond, "data": a.data, "seed": a.seed,
           "iters": a.iters, "depth_w": a.depth_w,
           "train_time_s": round(time.time() - t0, 1),
           "n_gauss": int(means.shape[0])}
    with torch.no_grad():
        # loaded HERE, after training: allocating it earlier perturbs the CUDA
        # allocator layout, and gsplat's atomicAdd backward is accumulation-order
        # sensitive, which silently shifts the very numbers this metric reports on.
        gpid = torch.tensor(np.stack([np.load(f"{a.data}/rgb/{i:04d}_pid.npy")
                                      for i in range(n)]), device=dev)
        errs, sil_errs, psnrs = [], [], []
        strat = {"wall": [], "pole": [], "car": []}
        for i in test:
            out, _, _ = rasterization(
                means=means, quats=quats / quats.norm(dim=-1, keepdim=True),
                scales=scales.exp(), opacities=torch.sigmoid(opac), colors=cols,
                viewmats=viewmats[i:i+1], Ks=K, width=W, height=H,
                render_mode="RGB-Ed", packed=False, with_eval3d=True,
                backgrounds=BG)
            img = out[0, ..., :3]; d = out[0, ..., 3]
            g = gtd[i]; m = g > 0
            e = (d - g).abs()
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
            band = torch.nn.functional.max_pool2d(
                step.float()[None, None], 9, 1, 4)[0, 0] > 0
            mm = m & band
            if mm.any():
                sil_errs.append(e[mm].mean().item())
            mse = ((img - imgs[i]) ** 2).mean().item()
            psnrs.append(-10 * math.log10(max(mse, 1e-10)))
        res.update(depth_mae=float(np.mean(errs)),
                   depth_mae_silhouette=float(np.mean(sil_errs)),
                   psnr=float(np.mean(psnrs)),
                   **{f"mae_{k}": (float(np.mean(v)) if v else None)
                      for k, v in strat.items()})

        # ---------- §5.1 mechanism probe (opt-in, post-hoc, non-perturbing) ----------
        if a.probe:
            # gsplat's far_plane culls a Gaussian when its CENTRE's camera-space z
            # exceeds the plane (ProjectionEWA3DGSFused.cu: `mean_c.z > far_plane`).
            # Verified off-axis: with two Gaussians on one ray at z=8/16 (along-ray
            # t=9.765/19.530), alpha steps at far_plane 9 and in (12,17], never at
            # 19.53. So the slice variable is z, while the analytic depth gtd and the
            # "Ed" render are ALONG-RAY distance -- convert per pixel.
            uu, vv = torch.meshgrid(torch.arange(W, device=dev),
                                    torch.arange(H, device=dev), indexing="xy")
            xc = (uu + 0.5 - W / 2) / f; yc = (vv + 0.5 - H / 2) / f
            cos_th = 1.0 / torch.sqrt(xc ** 2 + yc ** 2 + 1.0)      # z = t * cos(theta)
            planes = torch.arange(1.5, 45.01, 1.5, device=dev)      # 30 slices, in z
            os.makedirs(a.probe, exist_ok=True)
            PROF_OFF = np.arange(-5.0, 20.01, 1.0)
            prof = np.zeros(len(PROF_OFF)); prof_n = np.zeros(len(PROF_OFF))
            s1 = s2 = ssat = 0.0; nfg = 0; nview = 0
            for i in test:
                A = torch.empty(len(planes), H, W, device=dev)
                for k, fp in enumerate(planes):
                    _, al, _ = rasterization(
                        means=means, quats=quats / quats.norm(dim=-1, keepdim=True),
                        scales=scales.exp(), opacities=torch.sigmoid(opac), colors=cols,
                        viewmats=viewmats[i:i+1], Ks=K, width=W, height=H,
                        render_mode="RGB-Ed", packed=False, with_eval3d=True,
                        backgrounds=BG, far_plane=float(fp))
                    A[k] = al[0, ..., 0]
                A = torch.cummax(A, dim=0).values   # alpha is monotone in far_plane;
                                                    # enforce it so interpolation is sane
                Af = A.reshape(len(planes), -1)

                def alpha_at(zq):
                    """profile interpolated at per-pixel depth zq (VALID RANGE ONLY --
                    outside it alpha_at returns the endpoint, which is a different
                    depth than asked for, so callers must mask)"""
                    k = torch.clamp(torch.searchsorted(
                        planes.contiguous(), zq.reshape(-1).contiguous()), 1, len(planes) - 1)
                    z0 = planes[k - 1]; z1 = planes[k]
                    a0 = Af.gather(0, (k - 1)[None])[0]; a1 = Af.gather(0, k[None])[0]
                    w = ((zq.reshape(-1) - z0) / (z1 - z0)).clamp(0, 1)
                    return (a0 + w * (a1 - a0)).reshape(H, W)

                # the plan writes the offsets in ALONG-RAY metres, so they are added to
                # t and the SUM is converted -- adding them to z would give a window of
                # delta/cos(theta), which at the frame corner is 12.4 m for a nominal 10
                def zq_of(off):
                    return (gtd[i] + off) * cos_th

                pm = gpid[i]
                fg = (gtd[i] > 0) & ((pm == PID["carL"]) | (pm == PID["carR"]) |
                                     (pm == PID["pole"]))
                if not fg.any():
                    continue
                # hard gate: a query outside the slice grid would silently be answered at
                # the grid endpoint instead. Fail loudly rather than report a wrong depth.
                for off in (0.5, 10.0):
                    q = zq_of(off)[fg]
                    if not bool(((q > planes[0]) & (q < planes[-1])).all()):
                        raise ValueError(f"probe query at +{off} m leaves the slice grid "
                                         f"[{planes[0]:.1f},{planes[-1]:.1f}]: "
                                         f"{q.min():.2f}..{q.max():.2f}")
                a_near = alpha_at(zq_of(0.5))       # opacity accumulated AT the surface
                a_far = alpha_at(zq_of(10.0))       # ... and 10 m of ray behind it
                # count-weighted: the plan's unit is the foreground PIXEL SET, and the
                # per-view foreground count varies by several fold
                s1 += a_near[fg].sum().item(); s2 += (a_far - a_near)[fg].sum().item()
                ssat += A[-1][fg].sum().item(); nfg += int(fg.sum()); nview += 1
                # descriptive profile: mean opacity as a function of depth RELATIVE to
                # the true surface. Not a test -- P1/P2/P3 are unchanged -- it is the
                # picture the three scalars summarise, and it costs nothing because the
                # slices are already rendered. Queries off the grid are DROPPED, not
                # clamped (2495 of 151410 foreground queries at +20 m sit beyond 45 m).
                for k, off in enumerate(PROF_OFF):
                    zq = zq_of(off)
                    ok = fg & (zq > planes[0]) & (zq < planes[-1])
                    if ok.any():
                        prof[k] += alpha_at(zq)[ok].sum().item(); prof_n[k] += int(ok.sum())
                # full-frame map kept for the M3 coherence test, which needs the
                # per-pixel DIFFERENCE against the matched `correct` run. float32: at
                # float16 a true difference of 0.0501 stores as 0.0498 and falls the
                # wrong side of the 0.05 depletion threshold
                np.save(f"{a.probe}/anear_{i:04d}.npy", a_near.cpu().numpy())
            res.update(probe_prof_off=PROF_OFF.tolist(),
                       probe_prof=(prof / np.maximum(prof_n, 1)).tolist(),
                       probe_prof_n=prof_n.tolist(),
                       probe_m1=s1 / nfg, probe_m2=s2 / nfg,
                       probe_alpha_full=ssat / nfg,
                       probe_fg_px=nfg, probe_views=nview)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
