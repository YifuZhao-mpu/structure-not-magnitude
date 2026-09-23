"""
可见性估计器的回归测试。先有测试, 再改实现。

T1/T2 是 Codex 用来证伪 v2 的反例, 必须通过。
"""
import sys, numpy as np
sys.path.insert(0,'code/nusc')
from visibility2 import estimate_v2
from visibility3 import estimate_v3, VIS, OCC, UNK
NAME={VIS:"VIS",OCC:"OCC",UNK:"UNK"}
W=H=64
K=np.array([[100.,0,32],[0,100.,32],[0,0,1]])

def mk(uv_z):
    """由 (u,v,z) 造相机系点"""
    P=[]
    for u,v,z in uv_z:
        P.append([(u-K[0,2])/K[0,0]*z,(v-K[1,2])/K[1,1]*z,z])
    return np.array(P)

def run(fn,pts,**kw):
    st,uv,z,inim=fn(pts,K,W,H,**kw)
    return [NAME[s] for s in st]

def t1(fn):
    """
    同 cell 两个空间分离、互不遮挡的小表面(每面仅 2 点)。
    关键要求是【不得误判为 OCCLUDED】—— 采样稀疏时判 UNKNOWN 是诚实的;
    判 OCC 则是 v2 的致命假阳性。
    """
    # cell=64 -> 整幅图为一个 cell, 四点必然同 cell。
    # 两个表面在图像上左右分离(u=10,11 与 u=54,55), 各自射线不同, 互不遮挡。
    pts=mk([(10,32,10.),(11,32,10.),(54,32,30.),(55,32,30.)])
    got=run(fn,pts,cell=64)
    return got, all(g!="OCC" for g in got)

def t1b(fn):
    """
    同上但每面 4 点。要求: 不得判 OCC, 且多数点判 VIS。
    边缘点邻域内同伴少于 min_evidence 时判 UNK 是正确的保守行为。
    """
    pts=mk([(8,32,10.),(9,32,10.),(10,32,10.),(11,32,10.),
            (52,32,30.),(53,32,30.),(54,32,30.),(55,32,30.)])
    got=run(fn,pts,cell=64)
    return got, all(g!="OCC" for g in got) and sum(g=="VIS" for g in got)>=len(got)//2

def t2(fn):
    """证据不足(单点)时不应武断判 VISIBLE"""
    pts=mk([(32,32,10.)])
    got=run(fn,pts,cell=64)
    return got, got[0]=="UNK"

def t3(fn):
    """真实遮挡: 同一射线上近点挡住远点 -> 远点应判 OCC"""
    pts=mk([(32,32,5.),(32,32,25.),(32.3,32,5.),(32.3,32,25.)])
    got=run(fn,pts,cell=64)
    return got, got[1]=="OCC" or got[3]=="OCC"

if __name__=="__main__":
    tests=[("T1  同cell两个互不遮挡表面/稀疏(不得判OCC)",t1),
           ("T1b 同上但证据充足(应判VIS)",t1b),
           ("T2 证据不足单点(不应判VIS)",t2),
           ("T3 真实遮挡(远点应判OCC)",t3)]
    for label,fn in [("v2 (cell 间隙)",estimate_v2),("v3 (射线级)",estimate_v3)]:
        print(f"=== {label} ===")
        for nm,t in tests:
            got,ok=t(fn)
            print(f"  {'PASS' if ok else 'FAIL'}  {nm}")
            print(f"        得到: {got}")
        print()
