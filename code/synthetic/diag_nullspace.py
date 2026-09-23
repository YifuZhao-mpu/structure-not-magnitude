"""
native 基线为何救不回来？假设：native 监督存在零空间 —— 高斯可以在 LiDAR 射线上
给出正确的期望命中距离，同时在相机射线上给出错误深度。

若成立: 训练后 LiDAR 视角 range 误差【小】而相机视角深度误差【大】。
这不是 bug，而是 alpha 合成期望深度不等于表面位置的直接后果 (Codex 的 D3 警告),
且正是"不同射线原点的约束并不等价"的实证。
"""
import sys, json, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'code/synthetic')
from train_gateA import load, init_gaussians
from gsplat.rendering import rasterization
dev='cuda'; ds='data/gateA/w24'
meta,rgb,dep,sup=load(ds)
n=meta["n_views"]; W,H,f=meta["W"],meta["H"],meta["fx"]
LC=meta["lidar_cam"]; LW,LH,LF=LC["W"],LC["H"],LC["f"]
test=list(range(0,n,4)); train=[i for i in range(n) if i not in test]
K=torch.tensor([[f,0,W/2],[0,f,H/2],[0,0,1]],dtype=torch.float32,device=dev)[None]
LK=torch.tensor([[LF,0,LW/2],[0,LF,LH/2],[0,0,1]],dtype=torch.float32,device=dev)[None]
poses=torch.tensor(np.array(meta["poses"]),dtype=torch.float32,device=dev)
vm=torch.linalg.inv(poses)
lp=poses.clone(); lp[:,:3,3]=torch.tensor(np.array(meta["lidar_origins"]),dtype=torch.float32,device=dev)
lvm=torch.linalg.inv(lp)
imgs=torch.tensor(rgb,dtype=torch.float32,device=dev); gtd=torch.tensor(dep,dtype=torch.float32,device=dev)
nat=[dict(uv=torch.tensor(np.load(f"{ds}/nat_{i:04d}.npz")["uv"],device=dev),
          rng=torch.tensor(np.load(f"{ds}/nat_{i:04d}.npz")["rng"],device=dev)) for i in range(n)]
means,scales,quats,opac,cols=init_gaussians(meta,sup,dev,seed=0)
opt=torch.optim.Adam([{"params":[means],"lr":1.6e-3},{"params":[scales],"lr":5e-3},
    {"params":[quats],"lr":1e-3},{"params":[opac],"lr":5e-2},{"params":[cols],"lr":2.5e-3}])
def R(v,Ks,w,h,mode,e3=False):
    return rasterization(means=means,quats=quats/quats.norm(dim=-1,keepdim=True),scales=scales.exp(),
        opacities=torch.sigmoid(opac),colors=cols,viewmats=v,Ks=Ks,width=w,height=h,
        render_mode=mode,packed=False,with_eval3d=e3)
for it in range(8000):
    i=train[np.random.randint(len(train))]
    out,_,_=R(vm[i:i+1],K,W,H,"RGB+ED"); loss=F.l1_loss(out[0,...,:3],imgs[i])
    lo,_,_=R(lvm[i:i+1],LK,LW,LH,"Ed",True); hd=lo[0,...,0]
    s=nat[i]; u=s["uv"][:,0].long().clamp(0,LW-1); v=s["uv"][:,1].long().clamp(0,LH-1)
    pr=hd[v,u]
    loss=loss+0.5*(pr-s["rng"]).abs().mean()+0.5*torch.relu(s["rng"]-0.5-pr).mean()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    le=[];ce=[]
    for i in test:
        lo,_,_=R(lvm[i:i+1],LK,LW,LH,"Ed",True); hd=lo[0,...,0]
        s=nat[i]; u=s["uv"][:,0].long().clamp(0,LW-1); v=s["uv"][:,1].long().clamp(0,LH-1)
        le.append((hd[v,u]-s["rng"]).abs().mean().item())
        o,_,_=R(vm[i:i+1],K,W,H,"RGB+ED"); g=gtd[i]; m=g>0
        ce.append((o[0,...,3]-g).abs()[m].mean().item())
print("native_los 训练 8000 步后（held-out 视角）:")
print(f"  LiDAR 视角 range  MAE = {np.mean(le):.4f} m   <- 被直接监督的量")
print(f"  相机视角 depth    MAE = {np.mean(ce):.4f} m   <- 被评估的量")
print(f"  比值 = {np.mean(ce)/np.mean(le):.1f}x")
print()
print("→ "+("✅ 零空间假说成立: native 监督在自己射线上拟合良好, 却未能约束相机射线"
      if np.mean(ce)>3*np.mean(le) else "❌ 假说不成立: 两个视角误差相当, 问题另有原因"))
