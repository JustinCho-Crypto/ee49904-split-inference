#!/usr/bin/env bash
# Re-run the main results: R1 (scenario sweep, both models) and R2 (live harness, ResNet-18).
# Run setup.sh first. Takes about 10-15 minutes on an 8-core laptop; close other programs.
# What should reproduce is the STRUCTURE (regime order, dominant term), not the milliseconds.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${SPLIT_VENV:-$HOME/split-inference-venv-313}"; PY="$VENV/bin/python"
KIT="${KIT_DIR:-$HERE/kit}/split-inference/starter"
OUT="${OUT:-$HERE/runs/reproduce_$(date +%Y%m%d_%H%M)}"
mkdir -p "$OUT/r1" "$OUT/r2"
cd "$KIT"
BW=1,2,5,10,20,50,100,200
"$PY" split_bench.py --models resnet18 --bandwidth $BW --rtt 20 --device-slowdown 15 --tail-speedup 38 \
      --repeats 20 --threads 4 --out "$OUT/r1/r1_resnet18_ds15_ts38.csv" | tee "$OUT/r1/r1_resnet18.log" | grep "best @"
"$PY" split_bench.py --models mobilenet_v3_large --bandwidth $BW --rtt 20 --device-slowdown 3.4 --tail-speedup 38 \
      --repeats 20 --threads 1 --out "$OUT/r1/r1_mobilenetv3_ds3.4_ts38_t1.csv" | tee "$OUT/r1/r1_mobilenetv3.log" | grep "best @"
SPLIT_VENV="$VENV" KIT="$KIT" OUT="$OUT/r2" bash "$HERE/scripts/run_r2.sh"
"$PY" "$HERE/scripts/parse_r2.py" "$OUT/r2"
echo "results in $OUT"
