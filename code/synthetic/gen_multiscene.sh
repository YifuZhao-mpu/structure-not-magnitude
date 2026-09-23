#!/usr/bin/env bash
# Build N independent scenes at the REALISTIC residual regime (wall_dist 24 m base,
# jittered +/-15% per scene). Scene, not training seed, becomes the unit of replication.
set +u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate vislidar
N=${1:-10}
for k in $(seq 0 $((N-1))); do
  d=$(printf "data/multiscene/sc%02d" $k)
  if [ -f "$d/meta.json" ] && [ -f "$d/nat_0023.npz" ]; then echo "[skip] $d"; continue; fi
  python code/synthetic/gen_dataset.py --out "$d" --wall-dist 24.0 \
      --baseline 0.0 0.0 0.75 --views 24 --seed 0 --scene-seed "$k" || exit 1
  python code/synthetic/add_native.py --data "$d" || exit 1
done
echo "=== ALL SCENES BUILT ==="
python - <<'PY'
import json,glob
print(f"{'scene':8} {'wall_d':>7} {'occ%':>7} {'resid_med':>10} {'fx':>8} {'n_sup':>9}")
for f in sorted(glob.glob('data/multiscene/sc*/meta.json')):
    m=json.load(open(f)); s=m['stats']
    print(f"{f.split('/')[-2]:8} {m['scene_params']['wall']['dist']:7.2f} "
          f"{100*s['occluded_frac']:7.3f} {s['resid_median']:10.3f} {m['fx']:8.2f} "
          f"{s['visible']+s['occluded']:9,}")
PY
