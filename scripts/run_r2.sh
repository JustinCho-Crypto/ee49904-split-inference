#!/usr/bin/env bash
# R2: predicted vs measured on the live harness (1x/1x, fp32), unmodified kit.
# 3 cuts x 5 bandwidths, each selftest = 21 iterations (first discarded as warm-up -> mean of 20).
set -u
KIT="${KIT:-$(cd "$(dirname "$0")/.." && pwd)/kit/split-inference/starter}"
PY="${SPLIT_VENV:-$HOME/split-inference-venv-313}/bin/python"
OUT="${OUT:-$HOME/ee49904-split-inference/runs/r2}"
MODEL="${MODEL:-resnet18}"
CUTS="${CUTS:-input layer2 all_local}"
BWS="${BWS:-5 10 25 50 100}"
mkdir -p "$OUT"
cd "$KIT"
for cut in $CUTS; do
  for bw in $BWS; do
    log="$OUT/${MODEL}_${cut}_${bw}mbps.log"
    echo "== $MODEL $cut $bw Mb/s"
    "$PY" split_serve.py --selftest --model "$MODEL" --split "$cut" \
          --bandwidth "$bw" --rtt 20 --iters 21 > "$log" 2>&1
    grep -E "error|effective" "$log" | tail -2
  done
done
