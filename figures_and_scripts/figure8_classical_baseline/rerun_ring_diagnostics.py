#!/usr/bin/env python3
"""
Re-measure every IMU-only ring diagnostic on the CORRECTED harness (real PoseCANN, calibrated
velocity gain), at the DEPLOYED timestep dt = 0.02 s (50 Hz), which is the rate the estimator
actually runs at, and at the DEPLOYED anchor gain K_GRAVITY = 5 (matching the full-system
configuration; the earlier strong-anchor K=200 runs are the limit in which the ring exactly
reproduces the CF). Frequencies above the 25 Hz Nyquist therefore alias -- exactly the vibration
model the paper states for the deployed system.

Replaces:
  * Fig 2B  vibration envelope        (was: unanchored "33 deg no-reference floor" = w*T/sqrt(3))
  * SI Fig S2 bump drift              (was: unanchored "60.8 deg"  = 1.5/sqrt(2), RMS of the GT sinusoid)
  * Fig 3A  classical baselines       (ring curve was measured on the starved ring)
  * Fig 3B  transient-outlier sweep   (same)
"""
import os, sys, json, time
os.environ['JAX_PLATFORMS'] = 'cpu'; os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from ring_harness import calibrate_gain, run_ring_real, rmse_deg, wrap
from imu_filters import make_streams, run_cf, run_ekf
from snn_slam_system import imu_rate_average

DT = 0.02
G = 9.81


def make_streams_sin(dt, dur, seed, amp_vib=2.0, freq=115.0, gyro_bias=0.035,
                     A=1.5, f_true=0.1):
    """Bump-drift stream: sinusoidal true pitch + persistent gyro bias + wingbeat vibration."""
    rng = np.random.default_rng(seed)
    n = int(dur / dt); t = np.arange(n) * dt
    phi = rng.uniform(0.0, 2.0 * np.pi)                       # wingbeat phase vs the sample clock
    true_th = wrap(A * np.sin(2 * np.pi * f_true * t))
    true_w = A * 2 * np.pi * f_true * np.cos(2 * np.pi * f_true * t)
    # wingbeat delivered through the DEPLOYED IMU front-end (1 kHz IMU, mean rate per step)
    gyro = true_w + gyro_bias + imu_rate_average(freq, amp_vib, dt, n, phase0=phi) + rng.normal(0, 0.05, n)
    vib_a = imu_rate_average(freq, 0.1 * amp_vib, dt, n, phase0=phi)
    ax = G * np.sin(true_th) + vib_a + rng.normal(0, 0.08, n)
    az = G * np.cos(true_th) - vib_a + rng.normal(0, 0.08, n)
    return dict(t=t, true_th=true_th, gyro=gyro, th_accel=np.arctan2(ax, az),
                valid=np.ones(n, bool))


def main():
    t0 = time.time()
    sc, ratio = calibrate_gain(DT)
    out = {"dt": DT, "cal_scale": sc, "cal_gain": 0.045 * sc, "travel_ratio": ratio,
           "note": "unanchored ring now path-integrates (travel ratio ~1); previously ~0.000"}
    print(f"calibrated gain {0.045*sc:.4f} (scale {sc:.3f}), travel ratio {ratio:.4f}\n")

    # ---- (A) vibration envelope: Fig 2B ------------------------------------------------
    # Wingbeat frequencies across the 10-260 Hz envelope, delivered through the DEPLOYED IMU
    # front-end (1 kHz IMU; the estimator consumes the mean rate over each 20 ms step). A tone is
    # attenuated by |sinc(f*DT)|, so high-frequency wingbeats are strongly suppressed without any
    # synthetic filter. Tones at exact multiples of 1/DT = 50 Hz integrate to zero over a step --
    # a REAL physical null -- so the grid deliberately avoids them to keep every point informative.
    freqs = [10, 30, 70, 115, 170, 260]
    amps = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]
    dur = 2.0; steps = int(dur / DT)
    env = []
    print("(A) vibration envelope  [ring_anch / ring_unanch / CF / EKF]  RMSE deg")
    for amp in amps:
        row = {"amp": amp, "per_freq": []}
        for f in freqs:
            # distinct seed per (freq, amp) -- avoids the amp 0.1/0.5 collision of int(f*10+amp)
            S = make_streams(f, amp, DT, steps, 0.5, 20240707 + f * 1000 + int(round(amp * 10)))
            tg = run_cf(S, DT)
            ra = rmse_deg(run_ring_real(S, DT, sc, True, tg), S['true_th'])
            ru = rmse_deg(run_ring_real(S, DT, sc, False), S['true_th'])
            cf = rmse_deg(tg, S['true_th']); ek = rmse_deg(run_ekf(S, DT), S['true_th'])
            row["per_freq"].append(dict(freq=f, ring_anch=ra, ring_unanch=ru, cf=cf, ekf=ek))
        for k in ("ring_anch", "ring_unanch", "cf", "ekf"):
            row[k] = float(np.mean([p[k] for p in row["per_freq"]]))
        env.append(row)
        print(f"  amp {amp:5.1f} | {row['ring_anch']:7.2f} {row['ring_unanch']:8.2f} "
              f"{row['cf']:7.2f} {row['ekf']:7.2f}")
    out["envelope"] = env

    # ---- (B) bump drift over 30 s: SI Fig S2 -------------------------------------------
    print("\n(B) bump drift, 30 s, sinusoidal truth + persistent gyro bias")
    S = make_streams_sin(DT, 30.0, seed=7)
    tg = run_cf(S, DT)
    ea = run_ring_real(S, DT, sc, True, tg)
    eu = run_ring_real(S, DT, sc, False)
    def running(e):
        err = wrap(e - S['true_th'])
        return np.degrees(np.sqrt(np.cumsum(err ** 2) / np.arange(1, len(err) + 1)))
    ra, ru = running(ea), running(eu)
    out["bump_drift"] = {"t": S['t'].tolist(), "running_anch": ra.tolist(),
                         "running_unanch": ru.tolist(),
                         "final_anch": float(ra[-1]), "final_unanch": float(ru[-1]),
                         "gt_rms_deg": float(np.degrees(np.sqrt(np.mean(S['true_th'] ** 2))))}
    for T in (2, 5, 10, 20, 30):
        i = int(T / DT) - 1
        print(f"   t={T:2d}s | anchored {ra[i]:6.2f} deg | unanchored {ru[i]:7.2f} deg")
    print(f"   (RMS of the ground-truth sinusoid itself = {out['bump_drift']['gt_rms_deg']:.2f} deg;")
    print(f"    the OLD figure reported unanchored 60.8 deg, i.e. exactly this stationary-bump value)")

    # ---- (C) transient-outlier sweep: Fig 3B -------------------------------------------
    print("\n(C) transient accel-outlier sweep (anchored ring vs CF vs EKF)")
    seeds = [11, 23, 37, 51, 67]; dur = 2.0; steps = int(dur / DT)
    outl = []
    for prob in (0.0, 0.02, 0.05, 0.08, 0.16, 0.40):
        rr, rc, re = [], [], []
        for sd in seeds:
            S = make_streams(115, 2.0, DT, steps, 0.5, sd,
                             outliers=(prob, np.deg2rad(40.0)) if prob > 0 else None)
            tg = run_cf(S, DT)
            rr.append(rmse_deg(run_ring_real(S, DT, sc, True, tg), S['true_th']))
            rc.append(rmse_deg(tg, S['true_th'])); re.append(rmse_deg(run_ekf(S, DT), S['true_th']))
        row = dict(rate=prob, ring=float(np.mean(rr)), cf=float(np.mean(rc)), ekf=float(np.mean(re)))
        outl.append(row)
        print(f"   rate {prob:4.2f} | ring {row['ring']:6.2f}  CF {row['cf']:6.2f}  EKF {row['ekf']:6.2f}")
    out["outliers"] = outl

    json.dump(out, open(os.path.join(_HERE, 'ring_diagnostics_corrected.json'), 'w'), indent=2)
    print(f"\n[{time.time()-t0:.0f}s] wrote ring_diagnostics_corrected.json")


if __name__ == '__main__':
    main()
