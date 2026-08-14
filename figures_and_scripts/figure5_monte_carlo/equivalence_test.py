#!/usr/bin/env python3
"""
!!! SUPERSEDED -- RETAINED ONLY FOR PROVENANCE OF A WITHDRAWN CLAIM !!!

This script tested whether the three navigation arms reach *equivalent* open-course ATE.
They did -- but only because, in the then-current wiring, the body velocity was rotated by
the raw gyro-integrated heading rather than by the gravity-anchored attitude readout. All
three arms therefore consumed an identical velocity signal AND an identical rotation, so
their trajectory errors coincided *by construction*. The resulting TOST equivalence
(p ~ 1e-16) measured a property of the code, not of the estimator, and the claim has been
withdrawn from the manuscript.

The wiring is fixed (snn_pose_cann.py, SLAM_COUPLE_HEADING, default 1): position is now
rotated by the attractor's attitude readout, so the arms genuinely differ and equivalence no
longer holds. Do not cite this script's output as a result. To reproduce the withdrawn
analysis, set SLAM_COUPLE_HEADING=0 and regenerate variance_results.json first.
"""
import os, sys, json, argparse
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "variance_results.json")


def tost_paired(a, b, margin, nboot=20000, seed=0):
    """Paired TOST + paired-bootstrap CI of the mean difference (a-b), in the input units."""
    diff = np.asarray(a) - np.asarray(b)          # paired per seed
    n = len(diff)
    md = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    # Two one-sided t-tests; equivalence p is the larger (worse) of the two tails.
    t_low = (md - (-margin)) / se                 # H0: (a-b) <= -margin
    t_high = (margin - md) / se                   # H0: (a-b) >= +margin
    p_tost = max(stats.t.sf(t_low, n - 1), stats.t.sf(t_high, n - 1))
    rng = np.random.default_rng(seed)
    bmeans = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    ci = np.percentile(bmeans, [2.5, 97.5])
    return dict(n=n, mean_diff=md, se=se, p_tost=p_tost, ci95=[ci[0], ci[1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--margin", type=float, default=0.5, help="equivalence margin, cm")
    ap.add_argument("--nboot", type=int, default=20000)
    args = ap.parse_args()

    res = json.load(open(args.json))["results"]
    modes = {
        "imu (dead-reckoning)": np.array([r["ate_imu"] for r in res]) * 100.0,
        "anchor-off (K_g=0)":   np.array([r["ate_ol"] for r in res]) * 100.0,
        "anchor-on (K_g=200)":  np.array([r["ate_cl"] for r in res]) * 100.0,
    }
    names = list(modes)
    print(f"n = {len(res)} seeds | equivalence margin = +/-{args.margin} cm")
    for k, v in modes.items():
        print(f"  {k:24s} mean ATE = {v.mean():.4f} cm")
    print()
    out = {"n": len(res), "margin_cm": args.margin, "pairs": {}}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = tost_paired(modes[names[i]], modes[names[j]], args.margin, args.nboot)
            key = f"{names[i]} vs {names[j]}"
            out["pairs"][key] = r
            print(f"  {key:48s}: mean diff = {r['mean_diff']:+.4f} cm | "
                  f"TOST p(+/-{args.margin}cm) = {r['p_tost']:.2e} | "
                  f"paired-boot 95% CI = [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] cm")
    outpath = os.path.join(HERE, "equivalence_test_results.json")
    json.dump(out, open(outpath, "w"), indent=2)
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
