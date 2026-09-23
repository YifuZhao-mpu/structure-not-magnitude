"""对照解析真值共视性, 审核全部过滤器。实现不正确的过滤器会让训练对比失去意义。"""
import sys, numpy as np
sys.path.insert(0,'code/synthetic'); sys.path.insert(0,'code/nusc')
from scene import default_scene
from capture import trajectory, covisibility
from filters import zbuffer_epipolar, hpr_katz, random_drop, edge_drop
from visibility3 import estimate_v3, VIS, OCC, UNK

def run(wall=24.0, baseline=0.75, n_views=8):
    sc=default_scene(wall_dist=wall); rigs=trajectory(n=n_views,baseline=(0,0,baseline))
    agg={}
    for r in rigs:
        pts,_,_=r.lidar_returns(sc); cv=covisibility(r,sc,pts); inim=cv["inim"]
        truth_occ=(cv["state"]=="occluded")
        pc=(pts-r.eye)@r.R
        K=np.array([[r.f,0,r.W/2],[0,r.f,r.H/2],[0,0,1]])
        lo=(r.lidar_o-r.eye)@r.R
        st,_,_,_=estimate_v3(pc,K,r.W,r.H,pix_tol=1.0,min_evidence=3)
        cand={
          "ours_v3(接受VIS)": st==VIS,
          "ours_v3(接受VIS+UNK)": st!=OCC,
          "z-buffer(RePLAy几何)": zbuffer_epipolar(pc,K,r.W,r.H,lo),
          "HPR(Katz)": hpr_katz(pc,K,r.W,r.H),
          "no_filter": np.ones(len(pc),bool),
        }
        n_acc=int((cand["ours_v3(接受VIS)"]&inim).sum())
        cand["random_drop(匹配接受数)"]=random_drop(len(pc),n_acc)
        cand["edge_drop(匹配接受数)"]=edge_drop(pc,K,r.W,r.H,n_acc)
        for k,acc in cand.items():
            a=acc&inim
            d=agg.setdefault(k,dict(acc=0,acc_occ=0,tot=0,tot_occ=0))
            d["acc"]+=int(a.sum()); d["acc_occ"]+=int((a&truth_occ).sum())
            d["tot"]+=int(inim.sum()); d["tot_occ"]+=int((inim&truth_occ).sum())
    return agg

for wall,tag in [(24.0,"w24 真实残差"),(38.0,"b075 放大残差")]:
    agg=run(wall)
    print(f"--- {tag} ---")
    print(f"{'过滤器':24} {'接受率%':>8} {'★错误接受%':>11} {'遮挡去除率%':>11}")
    print("-"*60)
    for k,d in agg.items():
        fa=100*d["acc_occ"]/max(d["acc"],1)
        rem=100*(1-d["acc_occ"]/max(d["tot_occ"],1))
        print(f"{k:24} {100*d['acc']/d['tot']:7.2f}% {fa:10.3f}% {rem:10.2f}%")
    print(f"  (真值遮挡率 {100*list(agg.values())[0]['tot_occ']/list(agg.values())[0]['tot']:.2f}%)\n")
print("★错误接受% = 接受集中实为遮挡的比例(越低越好); 遮挡去除率 = 真遮挡被剔除的比例(越高越好)")
