"""Parse R2 selftest logs into a CSV and an error-decomposition table."""
import csv, glob, os, re, sys
RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/ee49904-split-inference/runs/r2")
TENSOR = {"input": 602_112, "layer2": 401_408, "all_local": 0}  # fp32 bytes, ResNet-18
RTT_ASSUMED = 20.0
num = lambda pat, t: (float(m.group(1).replace(",", "")) if (m := re.search(pat, t)) else 0.0)
rows = []
for f in sorted(glob.glob(f"{RUN}/*_*mbps.log")):
    model, rest = os.path.basename(f).split("_", 1)
    cut, bw = rest.rsplit("_", 1); bw = float(bw.replace("mbps.log", ""))
    t = open(f).read()
    r = dict(model=model, cut=cut, bw_target=bw,
             bytes_sent=num(r"bytes sent\s+([\d,]+)", t),
             device_ms=num(r"device compute\s+([\d.]+)", t),
             ser_ms=num(r"serialization\s+([\d.]+)", t),
             rtt_ms=num(r"RTT \(measured\)\s+([\d.]+)", t),
             deser_ms=num(r"remote deserial\.\s+([\d.]+)", t),
             remote_ms=num(r"remote compute\s+([\d.]+)", t),
             transfer_ms=num(r"actual transfer\s+([\d.]+)", t),
             bw_eff=num(r"effective bandwidth ([\d.]+)", t) or None,
             measured_ms=num(r"measured end-to-end\s+([\d.]+)", t),
             predicted_ms=num(r"predicted \(bench\)\s+([\d.]+)", t),
             error_pct=num(r"error\s+([+-][\d.]+)", t))
    gap = r["measured_ms"] - r["predicted_ms"]
    if cut == "all_local":
        r.update(gap_ms=gap, gap_serdeser=0, gap_rtt=0, gap_transfer=0, gap_compute=gap)
    else:
        pred_transfer = TENSOR[cut] * 8 / (bw * 1e6) * 1e3
        pred_compute = r["predicted_ms"] - pred_transfer - RTT_ASSUMED
        r.update(gap_ms=gap,
                 gap_serdeser=r["ser_ms"] + r["deser_ms"],
                 gap_rtt=r["rtt_ms"] - RTT_ASSUMED,
                 gap_transfer=r["transfer_ms"] - pred_transfer,
                 gap_compute=r["device_ms"] + r["remote_ms"] - pred_compute)
    rows.append(r)
order = {"input": 0, "layer2": 1, "all_local": 2}
rows.sort(key=lambda r: (order.get(r["cut"], 9), r["bw_target"]))
with open(f"{RUN}/r2_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{'cut':<10}{'bw':>5}{'eff':>7}{'pred':>8}{'meas':>8}{'err%':>7} | {'gap':>6}{'ser+de':>7}{'RTT':>6}{'xfer':>6}{'comp':>6}")
for r in rows:
    eff = f"{r['bw_eff']:.1f}" if r["bw_eff"] else "-"
    print(f"{r['cut']:<10}{r['bw_target']:>5.0f}{eff:>7}{r['predicted_ms']:>8.1f}{r['measured_ms']:>8.1f}{r['error_pct']:>+7.1f} | "
          f"{r['gap_ms']:>6.1f}{r['gap_serdeser']:>7.1f}{r['gap_rtt']:>6.1f}{r['gap_transfer']:>6.1f}{r['gap_compute']:>6.1f}")
