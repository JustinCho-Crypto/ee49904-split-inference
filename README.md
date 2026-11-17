# EE49904 Term Project, Brief 2: Split Inference

Team: Jaeyong Lee (20210488), Eunhyun Cho (20210614) · KAIST EE49904 AI-Native Networks, Fall 2026
Extension: A (adaptive split policy)

This repository holds our scripts, run logs and CSVs. The kit itself (`split_bench.py`, `split_serve.py`, `splitlib.py`)
is the course's starter kit; `setup.sh` clones it at a fixed commit instead of copying it here.

## Environment (all results so far)

| | |
|---|---|
| Machine | MacBook Pro, Apple M1 Pro 8-core, macOS 27.0.1 |
| Python | 3.13.16 |
| torch / torchvision | 2.13.0 / 0.28.0 (the kit's verified versions), full list in `requirements-lock.txt` |
| Kit | `sunghyunc7/ee49904-term-project-2026f` at commit `e69d540` |
| Threads | 4 per process (ResNet-18), 1 (MobileNetV3) |

Versions matter: with torch 2.8.0 (Python 3.9) the MobileNetV3 boundary in R1 sat at a different bandwidth.

## Reproduce

```bash
bash setup.sh        # pinned venv (needs Python >= 3.10) + kit clone at the fixed commit
./reproduce_all.sh   # R1 for both models + R2 for ResNet-18, about 10-15 min
```

Absolute milliseconds will differ by machine. What should reproduce is the structure:
- **R1, ResNet-18** (phone 15x slower, server 38x faster, RTT 20 ms): all_local up to 20 Mb/s, input from 50 Mb/s, one boundary between 20 and 50 Mb/s.
- **R1, MobileNetV3-L** (phone 3.4x, 1 thread): all_local up to 50 Mb/s, features4_7 at 100 Mb/s, input at 200 Mb/s.
- **R2, ResNet-18**: measured latency above predicted at every point, and the error grows with bandwidth, because fixed costs the model does not price (RTT excess, serialization, compute on the harness) become a larger share as the transfer shrinks.

## Runs

| Run | What | Script / command | Output |
|---|---|---|---|
| R1 | Scenario sweep, optimal cut and boundaries | `split_bench.py` (see `reproduce_all.sh`) | `runs/r1/` |
| R2 | Predicted vs. measured, live harness 1x/1x, 3 cuts x 5 bandwidths | `scripts/run_r2.sh`, `scripts/parse_r2.py` | `runs/r2/`, `runs/r2/r2_summary.csv` |
| R3 | `--channels-last` (NHWC) stage profile and optimal cut | `split_bench.py --channels-last` | `runs/r3/` |
| Ext A | Adaptive split policy prototype on the scenario model | `scripts/ext_a_sim.py runs/r1/<model>.csv --out runs/ext_a` | `runs/ext_a/` |

Scenario factors and their sources are in our execution plan (Oct 15). Still to come: per-iteration tails (p50/p99), R2 for MobileNetV3, R4, and Extension A on the live harness.

## AI use
See `AI_USE_DISCLOSURE.md`.
