#!/bin/bash
# GATE A sweep. Conditions x seeds at a fixed baseline, then the baseline dose-response.
# Everything except the depth-label VALUES is held fixed across conditions.
set -u
PY=${PYTHON:-python}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ITERS=${ITERS:-3000}
SEEDS=${SEEDS:-"0 1 2"}
CONDS=${CONDS:-"none correct wrong shuffled_space shuffled_view sign_flipped"}
DATA=${DATA:-data/gateA/b075}
OUT=${OUT:-outputs/gateA}
mkdir -p "$OUT"
for s in $SEEDS; do
  for c in $CONDS; do
    f="$OUT/$(basename $DATA)_${c}_s${s}.json"
    if [ -f "$f" ]; then echo "SKIP $f"; continue; fi
    echo "[$(date +%H:%M:%S)] $DATA cond=$c seed=$s"
    $PY code/synthetic/train_gateA.py --data "$DATA" --cond "$c" \
        --seed "$s" --iters "$ITERS" --out "$f" || echo "FAILED $c s$s"
  done
done
echo "done"
