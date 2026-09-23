"""
Codex 要求的审核: practical visibility 的【错误接受率】。
"先审核错误接受率; 不过关就停止扩大 GATE C。"

合成场景的共视性是解析计算的真值, 因此可以直接量化估计器的错误:
  错误接受率 = 被判 VISIBLE 但真值为 occluded 的比例
             = 会被当作可靠监督、实际却是错标签的那部分
这是决定方法能否工作的关键数字: 它直接进入训练损失。
"""
import sys, os, json, numpy as np
sys.path.insert(0, 'code/nusc'); sys.path.insert(0, 'code/synthetic')
from scene import default_scene
from capture import trajectory, covisibility
from visibility2 import estimate_v2
from visibility3 import estimate_v3
VIS, OCC, UNK = 0, 1, 2

def audit(ds, wall, baseline, n_views=12):
    sc = default_scene(wall_dist=wall)
    rigs = trajectory(n=n_views, baseline=(0.0, 0.0, baseline))
    out = {}
    for name, fn, kw in [("v2 (cell 间隙)", estimate_v2, dict(cell=16)),
                         ("v3 (射线级)", estimate_v3, dict(pix_tol=1.0, min_evidence=3))]:
        TP=FP=TN=FN=0; nunk=0; ntot=0; unk_occ=0
        for r in rigs:
            pts,_,_ = r.lidar_returns(sc)
            cv = covisibility(r, sc, pts)
            inim = cv["inim"]
            truth = cv["state"][inim]                  # 'visible'/'occluded'
            pc = (pts - r.eye) @ r.R                   # 相机系
            K = np.array([[r.f,0,r.W/2],[0,r.f,r.H/2],[0,0,1]])
            st,uv,z,m = fn(pc, K, r.W, r.H, **kw)
            st = st[inim]
            t_occ = (truth == "occluded")
            ntot += len(st); nunk += int((st==UNK).sum())
            unk_occ += int(((st==UNK)&t_occ).sum())
            TP += int(((st==OCC)&t_occ).sum())          # 正确识别遮挡
            FP += int(((st==OCC)&~t_occ).sum())         # 误判为遮挡(丢弃了好数据)
            FN += int(((st==VIS)&t_occ).sum())          # ★错误接受: 判可见实为遮挡
            TN += int(((st==VIS)&~t_occ).sum())
        acc = TP+FP+FN+TN
        out[name] = dict(
            n=ntot, unk=100*nunk/ntot,
            false_accept=100*FN/max(TN+FN,1),           # 接受集中错标签占比
            occ_recall=100*TP/max(TP+FN+unk_occ,1),
            occ_prec=100*TP/max(TP+FP,1),
            accepted=100*(TN+FN)/ntot)
    return out

print("估计器审核 —— 对照解析真值共视性 (合成场景)\n")
for wall,b,tag in [(24.0,0.75,"w24 (真实残差)"), (38.0,0.75,"b075 (放大残差)")]:
    print(f"--- {tag} ---")
    res = audit(None, wall, b)
    print(f"{'判据':16} {'UNKNOWN%':>9} {'接受率%':>8} {'★错误接受%':>11} {'遮挡召回%':>9} {'遮挡精度%':>9}")
    print("-"*70)
    for k,v in res.items():
        print(f"{k:16} {v['unk']:8.2f}% {v['accepted']:7.2f}% {v['false_accept']:10.3f}% "
              f"{v['occ_recall']:8.2f}% {v['occ_prec']:8.2f}%")
    print()
print("★错误接受% = 被判 VISIBLE 却真实被遮挡的比例 —— 这些错标签会直接进入训练损失")
