"""Aggregate GATE A runs into the comparison that decides Claim 1."""
import json, glob, sys, statistics as st
from collections import defaultdict
rows = [json.load(open(f)) for f in glob.glob(sys.argv[1] if len(sys.argv) > 1 else "outputs/gateA/*.json")]
by = defaultdict(list)
for r in rows:
    by[(r["data"], r["cond"])].append(r)
print(f"{'dataset':22} {'condition':16} {'n':>2} {'depth MAE':>18} {'MAE @silhouette':>22} {'PSNR':>13}")
print("-" * 100)
def ms(v):
    return f"{st.mean(v):8.4f}" + (f" ±{st.stdev(v):6.4f}" if len(v) > 1 else "        ")
for (d, c), rs in sorted(by.items()):
    print(f"{d.split('/')[-1]:22} {c:16} {len(rs):2} "
          f"{ms([r['depth_mae'] for r in rs]):>18} "
          f"{ms([r['depth_mae_silhouette'] for r in rs]):>22} "
          f"{ms([r['psnr'] for r in rs]):>13}")
print()
print("DECISION RULE for Claim 1:")
print("  'wrong' must be WORSE than shuffled_space / shuffled_view / sign_flipped")
print("  on MAE @silhouette, at matched residual distribution.")
print("  If it is not -> the harm is noise, not structure -> Claim 1 fails.")
