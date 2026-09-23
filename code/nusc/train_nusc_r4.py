#!/usr/bin/env python3
"""
R-4 real-data corroboration trainer. Mirrors code/synthetic/train_gateB.py so the real
and synthetic experiments share a training and evaluation structure.

Conditions (frozen, paper-stage/PRE_ANALYSIS_PLAN_R4.md):
  native        LiDAR-ray supervision only
  native_naive  + every in-image return transferred as a camera depth label
  native_x2     native weight doubled, nothing transferred (loss-weight control)

There is no oracle condition: real data has no ground-truth visibility. That is the
reason only one of the three synthetic claims can be tested here.

Evaluation uses HELD-OUT LiDAR sweeps -- geometric evidence that needs no visibility
ground truth -- stratified into returns inside / outside an annotated 3D box.
"""
import os, sys, json, math, time, argparse
import numpy as np
import torch
import torch.nn.functional as F


def load(ds):
    meta = json.load(open(f"{ds}/meta.json"))
    n = meta["n_frames"]
    rgb = np.stack([np.load(f"{ds}/rgb/{i:04d}.npy") for i in range(n)])
    nat = [np.load(f"{ds}/nat_{i:04d}.npz") for i in range(n)]
    cam = [np.load(f"{ds}/cam_{i:04d}.npz") for i in range(n)]
    ev = {i: np.load(f"{ds}/eval_{i:04d}.npz") for i in meta["test"]}
    return meta, rgb, nat, cam, ev


def init_gaussians(nat, train_idx, dev, n_init=60000, seed=0):
    """Standard LiDAR initialisation from TRAINING frames only."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = np.concatenate([nat[i]["xyz"] for i in train_idx])
    sel = torch.randperm(len(pts), generator=g)[:n_init].numpy()
    P = pts[sel]
    means = torch.tensor(P, dtype=torch.float32, device=dev)
    from scipy.spatial import cKDTree
    dd, _ = cKDTree(P).query(P, k=2)
    nn = np.clip(dd[:, 1], 1e-2, 2.0)
    scales = torch.tensor(np.log(nn * 0.5)[:, None].repeat(3, 1), dtype=torch.float32, device=dev)
    quats = torch.zeros(len(P), 4, device=dev); quats[:, 0] = 1.0
    opac = torch.full((len(P),), 0.0, device=dev)
    cols = torch.full((len(P), 3), 0.5, device=dev)
    for t in (means, scales, quats, opac, cols):
        t.requires_grad_(True)
    return means, scales, quats, opac, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cond", required=True,
                    choices=["native", "native_naive", "native_x2"])
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--nat-w", type=float, default=0.5)
    ap.add_argument("--cam-w", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    from gsplat.rendering import rasterization

    meta, rgb, nat, cam, ev = load(a.data)
    W, H = meta["W"], meta["H"]
    LC = meta["lidar_cam"]; LW, LH, LF = LC["W"], LC["H"], LC["f"]
    train, test = meta["train"], meta["test"]

    K = torch.tensor([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]],
                     dtype=torch.float32, device=dev)[None]
    LK = torch.tensor([[LF, 0, LW / 2], [0, LF, LH / 2], [0, 0, 1]],
                      dtype=torch.float32, device=dev)[None]
    viewmats = torch.linalg.inv(torch.tensor(np.array(meta["poses"]), dtype=torch.float32, device=dev))
    lviewmats = torch.linalg.inv(torch.tensor(np.array(meta["lidar_poses"]), dtype=torch.float32, device=dev))
    imgs = torch.tensor(rgb, dtype=torch.float32, device=dev)

    T = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
    natT = [dict(uv=T(d["uv"]), rng=T(d["rng"])) for d in nat]
    camT = [dict(uv=T(d["uv"]), d=T(d["d"])) for d in cam]

    means, scales, quats, opac, cols = init_gaussians(nat, train, dev, seed=a.seed)
    opt = torch.optim.Adam([
        {"params": [means], "lr": 1.6e-3}, {"params": [scales], "lr": 5e-3},
        {"params": [quats], "lr": 1e-3}, {"params": [opac], "lr": 5e-2},
        {"params": [cols], "lr": 2.5e-3}])

    def render(vm, Ks, w, h, mode):
        return rasterization(means=means, quats=quats / quats.norm(dim=-1, keepdim=True),
                             scales=scales.exp(), opacities=torch.sigmoid(opac), colors=cols,
                             viewmats=vm, Ks=Ks, width=w, height=h, render_mode=mode,
                             packed=False, with_eval3d=True)

    nat_w = a.nat_w * (2.0 if a.cond == "native_x2" else 1.0)
    t0 = time.time()
    for it in range(a.iters):
        i = train[np.random.randint(len(train))]
        out, _, _ = render(viewmats[i:i+1], K, W, H, "RGB-Ed")
        img = out[0, ..., :3]; d_cam = out[0, ..., 3]
        loss = F.l1_loss(img, imgs[i])

        lout, _, _ = render(lviewmats[i:i+1], LK, LW, LH, "Ed")
        d_lid = lout[0, ..., 0]
        s = natT[i]
        if len(s["rng"]):
            u = s["uv"][:, 0].long().clamp(0, LW - 1); v = s["uv"][:, 1].long().clamp(0, LH - 1)
            loss = loss + nat_w * (d_lid[v, u] - s["rng"]).abs().mean()

        if a.cond == "native_naive":
            c = camT[i]
            if len(c["d"]):
                u2 = c["uv"][:, 0].long().clamp(0, W - 1); v2 = c["uv"][:, 1].long().clamp(0, H - 1)
                loss = loss + a.cam_w * (d_cam[v2, u2] - c["d"]).abs().mean()

        opt.zero_grad(); loss.backward(); opt.step()

    res = {"cond": a.cond, "data": a.data, "seed": a.seed, "iters": a.iters,
           "nat_w": a.nat_w, "cam_w": a.cam_w, "train_time_s": round(time.time() - t0, 1)}
    with torch.no_grad():
        allm, fgm, bgm, ps = [], [], [], []
        for i in test:
            lout, _, _ = render(lviewmats[i:i+1], LK, LW, LH, "Ed")
            d_lid = lout[0, ..., 0]
            e = ev[i]
            if len(e["rng"]):
                u = torch.tensor(e["uv"][:, 0], device=dev).long().clamp(0, LW - 1)
                v = torch.tensor(e["uv"][:, 1], device=dev).long().clamp(0, LH - 1)
                err = (d_lid[v, u] - torch.tensor(e["rng"], device=dev)).abs()
                fg = torch.tensor(e["fg"], device=dev)
                allm.append(err.mean().item())
                if fg.any(): fgm.append(err[fg].mean().item())
                if (~fg).any(): bgm.append(err[~fg].mean().item())
            out, _, _ = render(viewmats[i:i+1], K, W, H, "RGB-Ed")
            mse = ((out[0, ..., :3] - imgs[i]) ** 2).mean().item()
            ps.append(-10 * math.log10(max(mse, 1e-10)))
        res.update(range_mae=float(np.mean(allm)),
                   range_mae_fg=float(np.mean(fgm)) if fgm else None,
                   range_mae_bg=float(np.mean(bgm)) if bgm else None,
                   psnr=float(np.mean(ps)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
