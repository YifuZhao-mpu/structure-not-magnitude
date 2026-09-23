"""
GATE A 的正式判定。判定规则在跑实验之前就写死了（见 EXPERIMENT_PLAN.md）:

  'wrong'（真实遮挡造成的结构化错误标签）必须在轮廓带 depth MAE 上
  劣于 shuffled_space / shuffled_view / sign_flipped —— 这三者的残差幅度分布
  与 'wrong' 匹配，只有结构被破坏。

  若不劣 -> 损害来自幅度/噪声而非结构 -> Claim 1 失败。

按 seed 配对做检验（同一 seed 共享初始化与视图采样序列），并与实测的
CUDA 非确定性噪声地板对照，避免把噪声读成信号。
"""
import json, glob, math, statistics as st
from collections import defaultdict

NOISE_SIL = 0.01237   # 实测: 8 次完全相同配置重复的轮廓 MAE sd
NOISE_ALL = 0.00295

rows = [json.load(open(f)) for f in glob.glob("outputs/gateA/*.json")]
by = defaultdict(dict)
for r in rows:
    by[r["cond"]][r["seed"]] = r

def paired(a, b, key):
    seeds = sorted(set(by[a]) & set(by[b]))
    d = [by[a][s][key] - by[b][s][key] for s in seeds]
    n = len(d)
    if n < 2: return None
    m, sd = st.mean(d), st.stdev(d)
    t = m / (sd / math.sqrt(n)) if sd > 0 else float('inf')
    return dict(n=n, mean=m, sd=sd, t=t, cohen=m / sd if sd > 0 else float('inf'))

print("=" * 88)
print("GATE A 判定 — Claim 1")
print("=" * 88)
conds = ["correct", "wrong", "shuffled_space", "shuffled_view", "sign_flipped", "none"]
print(f"\n{'condition':16} {'n':>2} {'轮廓 MAE':>18} {'全局 MAE':>18} {'PSNR':>14}")
print("-" * 76)
for c in conds:
    if c not in by: continue
    v = list(by[c].values())
    f = lambda k: f"{st.mean([x[k] for x in v]):.4f}±{(st.stdev([x[k] for x in v]) if len(v)>1 else 0):.4f}"
    print(f"{c:16} {len(v):2} {f('depth_mae_silhouette'):>18} {f('depth_mae'):>18} {f('psnr'):>14}")

print(f"\n实测噪声地板（完全相同配置重复 8 次，仅 CUDA 非确定性）:")
print(f"  轮廓 MAE sd = {NOISE_SIL:.5f}   全局 MAE sd = {NOISE_ALL:.5f}")

print("\n" + "=" * 88)
print("核心检验：wrong  vs  幅度匹配的打乱对照（轮廓带 MAE，按 seed 配对）")
print("=" * 88)
print(f"{'对比':34} {'n':>2} {'Δ均值':>10} {'配对 sd':>9} {'t':>8} {'Cohen d':>9} {'/噪声地板':>10}")
print("-" * 88)
verdict = []
for b in ["shuffled_space", "shuffled_view", "sign_flipped"]:
    r = paired("wrong", b, "depth_mae_silhouette")
    if not r: continue
    ratio = r["mean"] / NOISE_SIL
    print(f"{'wrong - ' + b:34} {r['n']:2} {r['mean']:+10.4f} {r['sd']:9.4f} "
          f"{r['t']:8.2f} {r['cohen']:9.2f} {ratio:9.1f}x")
    verdict.append(r["mean"] > 0 and abs(r["t"]) > 2.5 and ratio > 3)

r = paired("wrong", "correct", "depth_mae_silhouette")
if r:
    print(f"\n{'(参考) wrong - correct':34} {r['n']:2} {r['mean']:+10.4f} {r['sd']:9.4f} "
          f"{r['t']:8.2f} {r['cohen']:9.2f} {r['mean']/NOISE_SIL:9.1f}x")

print("\n" + "=" * 88)
if verdict and all(verdict):
    print("✅ CLAIM 1 成立：wrong 在三项幅度匹配的对照上均显著更差。")
    print("   → 损害来自错误标签的【结构】(空间相干 + 跨视角相干 + 符号恒正)，而非幅度。")
    print("   → GATE A 通过，可进入 GATE B（oracle 头room 检验）。")
else:
    print("❌ CLAIM 1 不成立：wrong 未能稳定劣于幅度匹配的对照。")
    print("   → 损害无法归因于结构 → 按预案终止该路线。")
print("=" * 88)

# PSNR 是否会误导
if "none" in by and "wrong" in by:
    pn = st.mean([x["psnr"] for x in by["none"].values()])
    sn = st.mean([x["depth_mae_silhouette"] for x in by["none"].values()])
    sc = st.mean([x["depth_mae_silhouette"] for x in by["correct"].values()])
    print(f"\n附：仅看 PSNR 会得出相反结论")
    print(f"  none(无深度监督) PSNR={pn:.2f} 为全场最高，但轮廓 MAE={sn:.3f}，"
          f"是 correct({sc:.3f}) 的 {sn/sc:.1f} 倍")
    print(f"  → 几何指标必须分层报告；全局 PSNR 对几何质量不敏感。")
