"""
rank-4: 渲染深度门控的【自证偏误】(self-confirming bias)。

LidaRF 式做法: 若深度目标显著落在【当前渲染深度】之后, 判为被遮挡而拒绝。
隐患: 训练早期重建不成熟, 渲染深度不可靠 -> 错误拒绝【有效】表面
      -> 这些表面永远得不到监督 -> 重建继续不好。模型自身的错误被用来判定标签的错误。

门控条件:
  no_gate     : 不拒绝任何目标
  render_gate : 用当前渲染深度拒绝 (LidaRF 式, 存在自证偏误)
  geom_gate   : 用几何判据拒绝 (z-buffer, 不依赖当前重建)
  oracle_gate : 用解析真值可见性拒绝 (上界)

初始化:
  good : 标准 LiDAR 点初始化
  poor : 故意打乱(按 Codex 要求 "deliberately poor initialization")

关键测量: 门控的 precision/recall 对照【独立真值标签】, 随迭代变化。
"""
import os, sys, json, math, argparse, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gateA import load, init_gaussians
from filters import zbuffer_epipolar

PID = {"road":0,"wall":1,"carL":2,"carR":3,"pole":4}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--gate",required=True,choices=["no_gate","render_gate","geom_gate","oracle_gate","render_gate_curr","render_gate_warm"])
    ap.add_argument("--init",default="good",choices=["good","poor","sparse","poor_pos","poor_scale"])
    ap.add_argument("--iters",type=int,default=8000)
    ap.add_argument("--cam-w",type=float,default=0.5)
    ap.add_argument("--nat-w",type=float,default=0.5)
    ap.add_argument("--gate-thr",type=float,default=1.0)
    ap.add_argument("--n-init",type=int,default=0,help="0=按 init 预设")
    ap.add_argument("--adaptive-scale",action="store_true")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev="cuda"
    from gsplat.rendering import rasterization
    meta,rgb,dep,sup=load(a.data)
    n=meta["n_views"]; W,H,f=meta["W"],meta["H"],meta["fx"]
    LC=meta["lidar_cam"]; LW,LH,LF=LC["W"],LC["H"],LC["f"]
    test=list(range(0,n,4)); train=[i for i in range(n) if i not in test]
    pids=np.stack([np.load(f"{a.data}/rgb/{i:04d}_pid.npy") for i in range(n)])
    K=torch.tensor([[f,0,W/2],[0,f,H/2],[0,0,1]],dtype=torch.float32,device=dev)[None]
    LK=torch.tensor([[LF,0,LW/2],[0,LF,LH/2],[0,0,1]],dtype=torch.float32,device=dev)[None]
    poses=torch.tensor(np.array(meta["poses"]),dtype=torch.float32,device=dev)
    vm=torch.linalg.inv(poses)
    lp=poses.clone(); lp[:,:3,3]=torch.tensor(np.array(meta["lidar_origins"]),dtype=torch.float32,device=dev)
    lvm=torch.linalg.inv(lp)
    imgs=torch.tensor(rgb,dtype=torch.float32,device=dev); gtd=torch.tensor(dep,dtype=torch.float32,device=dev)
    gpid=torch.tensor(pids,device=dev)

    # 预计算几何门控(不依赖重建)
    geom_keep=[]
    for i in range(n):
        s=sup[i]
        Rm=poses[i][:3,:3].cpu().numpy(); eye=poses[i][:3,3].cpu().numpy()
        pc=(s["xyz"]-eye)@Rm
        Kn=np.array([[f,0,W/2],[0,f,H/2],[0,0,1]])
        geom_keep.append(torch.tensor(zbuffer_epipolar(pc,Kn,W,H,(np.array(meta["lidar_origins"][i])-eye)@Rm),device=dev))

    nat,cam=[],[]
    for i in range(n):
        d=np.load(f"{a.data}/nat_{i:04d}.npz")
        nat.append(dict(uv=torch.tensor(d["uv"],device=dev),rng=torch.tensor(d["rng"],device=dev)))
        s=sup[i]
        cam.append(dict(uv=torch.tensor(s["uv"],dtype=torch.float32,device=dev),
                        tgt=torch.tensor(s["wrong"],dtype=torch.float32,device=dev),   # naive 目标(含错标签)
                        occ=torch.tensor(s["occ"],device=dev),                          # 真值
                        n_orig=len(s["occ"])))

    # sparse init: 匹配【真实 nuScenes】的初始化稀疏度。
    # 实测 nuScenes 单帧 LiDAR 在 1600x900 相机中覆盖率仅 0.214%(每 467 像素 1 点),
    # 而本合成场景为 55.93% —— 高出 261 倍。即本文的 "good init" 远优于真实系统。
    # 故按真实覆盖率降采样, 检验 render_gate 在【真实稀疏度】下是否同样失效。
    n_init = a.n_init if a.n_init>0 else 60000
    if a.n_init==0 and a.init=="sparse":
        n_init = max(int(60000*0.214/55.93), 200)      # 按覆盖率比例降采样
    means,scales,quats,opac,cols=init_gaussians(meta,sup,dev,seed=a.seed,train_idx=train,n_init=n_init,adaptive_scale=a.adaptive_scale)
    # Codex Q1: 原 poor init 同时改了【位置】与【尺度】, 两个因素混在一起。
    #   实际扰动 RMS = sqrt(3)*2 ≈ 3.46 m; 尺度 0.06 -> 0.5 (放大 8 倍)。
    #   大尺度高斯易形成错误的前景"雾层", 随后"比预测远 1m 即拒绝"自然拒绝一切。
    # 若【仅放大尺度】即重现全部失效, 则测到的是"过大 splat x 过早门控"的相互作用,
    # 与跨传感器可见性无关。故拆成三条。
    if a.init in ("poor","poor_pos","poor_scale"):
        with torch.no_grad():
            g=torch.Generator(device=dev).manual_seed(a.seed)
            if a.init in ("poor","poor_pos"):
                means += torch.randn(means.shape,generator=g,device=dev)*2.0
            if a.init in ("poor","poor_scale"):
                scales.fill_(math.log(0.5))
    opt=torch.optim.Adam([{"params":[means],"lr":1.6e-3},{"params":[scales],"lr":5e-3},
        {"params":[quats],"lr":1e-3},{"params":[opac],"lr":5e-2},{"params":[cols],"lr":2.5e-3}])
    def R(v,Ks,w,h,mode,e3=False):
        return rasterization(means=means,quats=quats/quats.norm(dim=-1,keepdim=True),scales=scales.exp(),
            opacities=torch.sigmoid(opac),colors=cols,viewmats=v,Ks=Ks,width=w,height=h,
            render_mode=mode,packed=False,with_eval3d=e3)

    gate_log=[]
    t0=time.time()
    for it in range(a.iters):
        i=train[np.random.randint(len(train))]
        out,_,_=R(vm[i:i+1],K,W,H,"RGB-Ed",True)
        img=out[0,...,:3]; dpred=out[0,...,3]
        loss=F.l1_loss(img,imgs[i])
        s=nat[i]
        u=s["uv"][:,0].long().clamp(0,LW-1); v=s["uv"][:,1].long().clamp(0,LH-1)
        lo,_,_=R(lvm[i:i+1],LK,LW,LH,"Ed",True)
        loss=loss+a.nat_w*(lo[0,...,0][v,u]-s["rng"]).abs().mean()
        c=cam[i]
        u2=c["uv"][:,0].long().clamp(0,W-1); v2=c["uv"][:,1].long().clamp(0,H-1)
        pred=dpred[v2,u2]
        if a.gate=="no_gate":      keep=torch.ones_like(c["occ"])
        elif a.gate=="oracle_gate":keep=~c["occ"]
        elif a.gate=="geom_gate":  keep=geom_keep[i]
        elif a.gate=="render_gate_warm":
            # 对照 Q2: 预热期完全不门控, 之后再启用渲染深度门控。
            # 若 render_gate 的危害仅因"早期重建不成熟", 预热应能消除它。
            keep=torch.ones_like(c["occ"]) if it < a.iters*0.3 else (c["tgt"] <= pred.detach()+a.gate_thr)
        elif a.gate=="render_gate_curr":
            # 对照 Q2: LidaRF 式 near-to-far curriculum —— 阈值随训练由宽收紧,
            # 且早期只接受近处目标。若危害因此消失, 我批评的就是稻草人版本。
            prog=it/max(a.iters-1,1)
            thr=a.gate_thr*(8.0-7.0*prog)                     # 阈值 8x -> 1x
            near=c["tgt"] <= (5.0+45.0*prog)                  # 由近及远放开
            keep=near & (c["tgt"] <= pred.detach()+thr)
        else:                      keep=(c["tgt"] <= pred.detach()+a.gate_thr)   # 渲染深度门控
        if it%400==0:
            rej=~keep; occ=c["occ"]
            tp=int((rej&occ).sum()); fp=int((rej&~occ).sum()); fn=int((~rej&occ).sum())
            gate_log.append(dict(it=it,rej=int(rej.sum()),
                prec=100*tp/max(tp+fp,1), rec=100*tp/max(tp+fn,1),
                wrong_rej=100*fp/max(int((~occ).sum()),1)))   # 有效表面被错误拒绝的比例
        if keep.any():
            loss=loss+a.cam_w*((pred[keep]-c["tgt"][keep]).abs().sum()/c["n_orig"])
        opt.zero_grad(); loss.backward(); opt.step()

    res=dict(gate=a.gate,init=a.init,n_init=n_init,adaptive=a.adaptive_scale,data=a.data,seed=a.seed,iters=a.iters,
             train_time_s=round(time.time()-t0,1),gate_log=gate_log)
    with torch.no_grad():
        acc={k:[] for k in ["all","wall","pole","car"]}; ps=[]
        for i in test:
            out,_,_=R(vm[i:i+1],K,W,H,"RGB-Ed",True)
            d=out[0,...,3]; g=gtd[i]; pm=gpid[i]; m=g>0; e=(d-g).abs()
            acc["all"].append(e[m].mean().item())
            for nm,sel in [("wall",pm==PID["wall"]),("pole",pm==PID["pole"]),("car",(pm==PID["carL"])|(pm==PID["carR"]))]:
                mm=m&sel
                if mm.any(): acc[nm].append(e[mm].mean().item())
            ps.append(-10*math.log10(max(((out[0,...,:3]-imgs[i])**2).mean().item(),1e-10)))
        for k,v in acc.items(): res[f"mae_{k}"]=float(np.mean(v)) if v else None
        res["psnr"]=float(np.mean(ps))
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    json.dump(res,open(a.out,"w"),indent=1)
    print(json.dumps({k:v for k,v in res.items() if k!="gate_log"}))

if __name__=="__main__": main()
