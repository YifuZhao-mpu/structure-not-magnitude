"""
rank-3: 被相机遮挡的 LiDAR 回波, 应当【降级为沿自身射线的表面约束】,
        而不是连同其几何信息一起丢弃。

现有做法(LidaRF 式拒绝 / z-buffer / HPR / edge-drop) 都是把被遮挡回波整个剔除。
但那些回波仍是有效的表面测量 —— 它们只是不能充当【相机像素】的深度标签。

三条件(其余一切相同):
  discard   : 被遮挡回波从 native 与 camera 两侧【同时】剔除   <- 现有做法
  demote    : 被遮挡回波保留在 native 侧, 仅从 camera 侧剔除   <- 本文主张
  keep_all  : 被遮挡回波在两侧都保留(naive)                    <- 已知有害

Codex 对该缺口的验收要求:
  "improved background completeness and thin-structure retention
   at the same foreground accuracy"
故评估按图元分层: 背景(远墙) / 细结构(细杆) / 前景(车)。
"""
import os, sys, json, math, argparse, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gateA import load, init_gaussians

PID = {"road": 0, "wall": 1, "carL": 2, "carR": 3, "pole": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cond", required=True, choices=["discard", "demote", "keep_all"])
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--nat-w", type=float, default=0.5)
    ap.add_argument("--cam-w", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    from gsplat.rendering import rasterization

    meta, rgb, dep, sup = load(a.data)
    n = meta["n_views"]; W, H, f = meta["W"], meta["H"], meta["fx"]
    LC = meta["lidar_cam"]; LW, LH, LF = LC["W"], LC["H"], LC["f"]
    test = list(range(0, n, 4)); train = [i for i in range(n) if i not in test]
    pids = np.stack([np.load(f"{a.data}/rgb/{i:04d}_pid.npy") for i in range(n)])

    K = torch.tensor([[f,0,W/2],[0,f,H/2],[0,0,1]], dtype=torch.float32, device=dev)[None]
    LK = torch.tensor([[LF,0,LW/2],[0,LF,LH/2],[0,0,1]], dtype=torch.float32, device=dev)[None]
    poses = torch.tensor(np.array(meta["poses"]), dtype=torch.float32, device=dev)
    vm = torch.linalg.inv(poses)
    lp = poses.clone(); lp[:,:3,3] = torch.tensor(np.array(meta["lidar_origins"]), dtype=torch.float32, device=dev)
    lvm = torch.linalg.inv(lp)
    imgs = torch.tensor(rgb, dtype=torch.float32, device=dev)
    gtd = torch.tensor(dep, dtype=torch.float32, device=dev)
    gpid = torch.tensor(pids, device=dev)

    nat, cam = [], []
    for i in range(n):
        d = np.load(f"{a.data}/nat_{i:04d}.npz")
        occ_n = torch.tensor(d["occ"], device=dev)
        keep_n = ~occ_n if a.cond == "discard" else torch.ones_like(occ_n)
        nat.append(dict(uv=torch.tensor(d["uv"], device=dev)[keep_n],
                        rng=torch.tensor(d["rng"], device=dev)[keep_n]))
        s = sup[i]
        occ_c = torch.tensor(s["occ"], device=dev)
        keep_c = torch.ones_like(occ_c) if a.cond == "keep_all" else ~occ_c
        cam.append(dict(uv=torch.tensor(s["uv"], dtype=torch.float32, device=dev)[keep_c],
                        tgt=torch.tensor(s["wrong"] if a.cond=="keep_all" else s["correct"],
                                         dtype=torch.float32, device=dev)[keep_c],
                        n_orig=len(occ_c)))

    means, scales, quats, opac, cols = init_gaussians(meta, sup, dev, seed=a.seed, train_idx=train)
    opt = torch.optim.Adam([{"params":[means],"lr":1.6e-3},{"params":[scales],"lr":5e-3},
        {"params":[quats],"lr":1e-3},{"params":[opac],"lr":5e-2},{"params":[cols],"lr":2.5e-3}])
    def R(v,Ks,w,h,mode,e3=False,**kw):
        return rasterization(means=means,quats=quats/quats.norm(dim=-1,keepdim=True),
            scales=scales.exp(),opacities=torch.sigmoid(opac),colors=cols,viewmats=v,Ks=Ks,
            width=w,height=h,render_mode=mode,packed=False,with_eval3d=e3,**kw)

    t0=time.time()
    for it in range(a.iters):
        i = train[np.random.randint(len(train))]
        out,_,_ = R(vm[i:i+1],K,W,H,"RGB-Ed",True)
        img=out[0,...,:3]; dpred=out[0,...,3]
        loss = F.l1_loss(img, imgs[i])
        lo,_,_ = R(lvm[i:i+1],LK,LW,LH,"Ed",True); hd=lo[0,...,0]
        s=nat[i]
        if len(s["rng"]):
            u=s["uv"][:,0].long().clamp(0,LW-1); v=s["uv"][:,1].long().clamp(0,LH-1)
            loss = loss + a.nat_w*(hd[v,u]-s["rng"]).abs().mean()
        c=cam[i]
        if len(c["tgt"]):
            u2=c["uv"][:,0].long().clamp(0,W-1); v2=c["uv"][:,1].long().clamp(0,H-1)
            # 固定【原始点数】为分母, 避免 filtering 与 reweighting 混淆(Codex 要求)
            loss = loss + a.cam_w*((dpred[v2,u2]-c["tgt"]).abs().sum()/c["n_orig"])
        opt.zero_grad(); loss.backward(); opt.step()

    res=dict(cond=a.cond,data=a.data,seed=a.seed,iters=a.iters,train_time_s=round(time.time()-t0,1))
    with torch.no_grad():
        acc={k:[] for k in ["all","wall","pole","car"]}; ps=[]
        for i in test:
            out,_,_=R(vm[i:i+1],K,W,H,"RGB-Ed",True)
            d=out[0,...,3]; g=gtd[i]; pm=gpid[i]; m=g>0
            e=(d-g).abs()
            acc["all"].append(e[m].mean().item())
            for nm,sel in [("wall",pm==PID["wall"]),("pole",pm==PID["pole"]),
                           ("car",(pm==PID["carL"])|(pm==PID["carR"]))]:
                mm=m&sel
                if mm.any(): acc[nm].append(e[mm].mean().item())
            ps.append(-10*math.log10(max(((out[0,...,:3]-imgs[i])**2).mean().item(),1e-10)))
        for k,v in acc.items(): res[f"mae_{k}"]=float(np.mean(v)) if v else None
        res["psnr"]=float(np.mean(ps))
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    json.dump(res,open(a.out,"w"),indent=1); print(json.dumps(res))

if __name__=="__main__":
    main()
