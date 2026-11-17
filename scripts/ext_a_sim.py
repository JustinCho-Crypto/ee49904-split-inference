"""Extension A prototype: adaptive split policy on the scenario model, driven by bandwidth traces.

Per-cut costs come from an R1 split_bench.py CSV (device ms, remote ms, bytes), so the latency of a
cut at any bandwidth is   device + bytes*8/bw + RTT + remote   (all_local = device only).

Requests run back to back (closed loop). Simplification: the bandwidth in effect at the start of a
request holds for the whole request.

Policies
  oracle     : knows the current rate, picks the best cut for every request (lower bound)
  static     : one cut for the whole trace, the best one in hindsight
  adaptive   : sees only what the client can measure -- the effective bandwidth of its own finished
               offloaded requests (with measurement noise). EMA estimate + hysteresis margin.
               While it runs all_local it observes nothing, so after `probe_s` seconds without a
               measurement it offloads one request as a probe (its latency counts).

alpha and the margin are tuned on the DESIGN traces only, then scored on the HELD-OUT traces.

Usage: python3 ext_a_sim.py <r1.csv> [--rtt 20] [--seed 0] [--out DIR]
"""
import argparse, csv, math, os, random, statistics as st

LEVELS = [5, 10, 20, 50, 100, 200]          # Mb/s
DURATION = 60.0                              # s per trace


def load_costs(path, rtt):
    rows = [r for r in csv.DictReader(open(path)) if float(r["bandwidth_mbps"]) == 1.0]
    costs = {}
    for r in rows:
        dev, link, tot = float(r["device_ms"]), float(r["link_ms"]), float(r["total_ms"])
        costs[r["split"]] = dict(device=dev, remote=tot - dev - link, bytes=int(r["tx_bytes"]))
    model = rows[0]["model"]
    return model, costs


def latency(c, bw, rtt):
    if c["bytes"] == 0:
        return c["device"]
    return c["device"] + c["bytes"] * 8 / (bw * 1e6) * 1e3 + rtt + c["remote"]


# ---------------- traces: list of (start_s, bw) ----------------
def step_trace(period, rng):
    return [(i * period, rng.choice(LEVELS)) for i in range(int(DURATION / period))]


def fading_trace(rng, dt=0.1, sigma=0.15, mean=math.log(40), pull=0.05):
    x, out = mean, []
    for i in range(int(DURATION / dt)):
        x += pull * (mean - x) + rng.gauss(0, sigma)
        x = min(max(x, math.log(2)), math.log(200))
        out.append((i * dt, math.exp(x)))
    return out


def drop_trace(rng):
    out, t = [], 0.0
    while t < DURATION:
        if rng.random() < 0.15:
            d = rng.uniform(0.5, 2.0); out.append((t, 0.5)); t += d       # link nearly stalls
        else:
            out.append((t, rng.choice(LEVELS))); t += 1.0
    return out


def bw_at(trace, t):
    lo, hi = 0, len(trace) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if trace[mid][0] <= t: lo = mid
        else: hi = mid - 1
    return trace[lo][1]


# ---------------- policies ----------------
def run(trace, costs, rtt, policy, fixed=None, alpha=0.3, margin=0.1, probe_s=1.0,
        noise=0.1, rng=None):
    cuts = list(costs)
    t, lat, chosen = 0.0, [], []
    est, last_obs, cur, switches, probes = None, -1e9, "all_local", 0, 0
    while t < DURATION:
        bw = bw_at(trace, t)
        if policy == "oracle":
            cut = min(cuts, key=lambda c: latency(costs[c], bw, rtt))
        elif policy == "static":
            cut = fixed
        else:
            cut = cur
            if est is not None:
                best = min(cuts, key=lambda c: latency(costs[c], est, rtt))
                if latency(costs[best], est, rtt) < (1 - margin) * latency(costs[cur], est, rtt):
                    cut = best
            if cut == "all_local" and t - last_obs > probe_s:      # no information: probe once
                offload = [c for c in cuts if costs[c]["bytes"] > 0]
                ref = est if est is not None else LEVELS[len(LEVELS) // 2]
                cut = min(offload, key=lambda c: latency(costs[c], ref, rtt))
                probes += 1
            if cut != cur:
                switches += 1
            cur = cut
        L = latency(costs[cut], bw, rtt)
        if policy == "adaptive" and costs[cut]["bytes"] > 0:
            obs = bw * math.exp(rng.gauss(0, noise))
            est = obs if est is None else alpha * obs + (1 - alpha) * est
            last_obs = t + L / 1e3
            if cur == cut and latency(costs["all_local"], est, rtt) < latency(costs[cut], est, rtt):
                cur = "all_local"                                     # probe said: stay local
        lat.append(L); chosen.append(cut)
        t += L / 1e3
    return dict(lat=lat, switches=switches, probes=probes, chosen=chosen)


def pct(xs, q):
    s = sorted(xs); return s[min(len(s) - 1, int(q * len(s)))]


def evaluate(trace, costs, rtt, alpha, margin, seed):
    o = run(trace, costs, rtt, "oracle")
    statics = {c: run(trace, costs, rtt, "static", fixed=c) for c in costs}
    best_c = min(statics, key=lambda c: st.mean(statics[c]["lat"]))
    a = run(trace, costs, rtt, "adaptive", alpha=alpha, margin=margin, rng=random.Random(seed))
    return o, best_c, statics[best_c], a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("r1_csv"); ap.add_argument("--rtt", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=".")
    a = ap.parse_args()
    model, costs = load_costs(a.r1_csv, a.rtt)
    rng = random.Random(a.seed)
    design = {f"step_{p}s": step_trace(p, rng) for p in (0.1, 1.0, 10.0)}
    heldout = {"fading": fading_trace(rng), "drops": drop_trace(rng),
               **{f"step_{p}s_new": step_trace(p, rng) for p in (0.1, 1.0, 10.0)}}

    # tune on design traces only
    grid = [(al, m) for al in (0.2, 0.3, 0.5, 0.8) for m in (0.0, 0.05, 0.1, 0.2)]
    def design_regret(al, m):
        return st.mean(st.mean(evaluate(tr, costs, a.rtt, al, m, a.seed)[3]["lat"]) -
                       st.mean(evaluate(tr, costs, a.rtt, al, m, a.seed)[0]["lat"])
                       for tr in design.values())
    alpha, margin = min(grid, key=lambda g: design_regret(*g))

    rows = []
    for split, traces in (("design", design), ("held-out", heldout)):
        for name, tr in traces.items():
            o, best_c, s, ad = evaluate(tr, costs, a.rtt, alpha, margin, a.seed)
            mo, ms, ma = st.mean(o["lat"]), st.mean(s["lat"]), st.mean(ad["lat"])
            rows.append(dict(model=model, set=split, trace=name, alpha=alpha, margin=margin,
                             oracle_mean=round(mo, 1), static_cut=best_c, static_mean=round(ms, 1),
                             adaptive_mean=round(ma, 1), regret_adaptive=round(ma - mo, 1),
                             regret_static=round(ms - mo, 1),
                             adaptive_p95=round(pct(ad["lat"], .95), 1), static_p95=round(pct(s["lat"], .95), 1),
                             adaptive_p99=round(pct(ad["lat"], .99), 1), static_p99=round(pct(s["lat"], .99), 1),
                             switches=ad["switches"], probes=ad["probes"], requests=len(ad["lat"])))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"ext_a_{model}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{model}: tuned on design traces -> alpha={alpha}, margin={margin}")
    print(f"{'set':<9}{'trace':<14}{'oracle':>8}{'static(cut)':>20}{'adaptive':>9}{'regA':>7}{'regS':>7}{'p99 A/S':>14}{'sw':>5}{'probe':>6}")
    for r in rows:
        print(f"{r['set']:<9}{r['trace']:<14}{r['oracle_mean']:>8}{r['static_mean']:>9} ({r['static_cut']:>9}){r['adaptive_mean']:>9}"
              f"{r['regret_adaptive']:>7}{r['regret_static']:>7}{r['adaptive_p99']:>7}/{r['static_p99']:<6}{r['switches']:>5}{r['probes']:>6}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
