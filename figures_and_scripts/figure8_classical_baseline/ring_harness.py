#!/usr/bin/env python3
"""
ring_harness.py -- CORRECTED IMU-only ring-attractor harness.

WHY THIS EXISTS
---------------
The previous IMU-only diagnostics (figure2_vibration_phase/generate_vibration_phase.py,
figure3_bump_drift/analyze_bump_drift.py, and classical_baseline.run_ring) re-implemented the
ring by hand instead of driving the deployed PoseCANN. That replica had two faults:

  1. Bump amplitude ~40x too small. The real PoseCANN maintains peak ring activity ~5.3 via
     divisive normalisation; the replica initialised u_ring ~0.13 and used clip(u,0,1).
     Since I_vel ~ gain * omega * (W_asym @ r), a 40x smaller r starves the velocity drive
     while the recurrent restoring force still pins the bump.
  2. The cerebellar velocity gain was never trained (update_cerebellum is never called in
     those scripts), leaving it at its initialisation (0.045).

Consequence: the UNANCHORED bump did not rotate at all (travel ratio ~0.000). The reported
"33 deg no-reference tracking floor" was therefore just omega*T/sqrt(3), and the reported
"60.8 deg unanchored bump drift" was just the RMS of the ground-truth sinusoid (1.5/sqrt(2)).
Both were properties of the ground truth, not of the estimator.

THIS HARNESS
------------
Drives the real PoseCANN (its own normalisation, init and dynamics) and calibrates the
gyro-velocity gain so the unanchored bump path-integrates with unit gain -- which is what the
deployed system achieves online via update_cerebellum (learned gain 0.13-0.17; the calibrated
value lands at ~0.19, confirming the calibration is not arbitrary).

With a working integrator the unanchored ring drifts WITHOUT BOUND under a gyro bias, as the
SI theory predicts -- it does not sit at a constant floor.
"""
import os, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import jax.numpy as jnp
from jax import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', 'neuro-symbolic-slam'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, 'src'))
from snn_slam_system import SNNSLAMSystem

wrap = lambda a: (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
_BASE_W = None


def fresh_pose(k_gravity=0.0, cal_scale=1.0, seed=42):
    """A real PoseCANN, reset and initialised, with the velocity gain scaled by cal_scale."""
    global _BASE_W
    p = SNNSLAMSystem(random.PRNGKey(seed)).pose
    p.reset(1)
    p.initialize_pose(jnp.array([[1.0, 1.0]]), jnp.array([0.0]))
    if _BASE_W is None:
        _BASE_W = p.W_cereb_th_imu.copy()
    p.W_cereb_th_imu = _BASE_W * cal_scale
    p.K_GRAVITY = float(k_gravity)
    return p


def _travel_ratio(cal_scale, dt, w=0.5, dur=2.0):
    p = fresh_pose(0.0, cal_scale)
    est = []
    for _ in range(int(dur / dt)):
        p(jnp.array([[0.0, 0.0, w]]), theta_gravity=None, dt=dt)
        est.append(float(p.estimate_heading()[0]))
    e = np.unwrap(np.array(est))
    return (e[-1] - e[0]) / (w * dur)


def calibrate_gain(dt, tol=1e-3, iters=12):
    """Bisect the velocity-gain scale so the UNANCHORED bump tracks a constant rate at unit gain."""
    lo, hi = 1.0, 60.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _travel_ratio(mid, dt) < 1.0:
            lo = mid
        else:
            hi = mid
    sc = 0.5 * (lo + hi)
    return sc, _travel_ratio(sc, dt)


def run_ring_real(S, dt, cal_scale, anchored, theta_g=None, k_gravity=5.0):
    """Drive the real PoseCANN with the shared IMU stream. Returns decoded pitch per step.

    anchored=True  -> K_GRAVITY=k_gravity (default 5.0, the deployed full-system gain) and the
                      CF output theta_g is injected as the anchor target. Pass k_gravity=200
                      to probe the strong-anchor limit in which the ring reproduces the CF.
    anchored=False -> K_GRAVITY=0, ring path-integrates the gyro alone (genuine dead reckoning).
    """
    p = fresh_pose(float(k_gravity) if anchored else 0.0, cal_scale)
    n = len(S['gyro'])
    out = np.zeros(n)
    for i in range(n):
        tg = None
        if anchored and S['valid'][i] and theta_g is not None:
            tg = jnp.array([float(theta_g[i])])
        p(jnp.array([[0.0, 0.0, float(S['gyro'][i])]]), theta_gravity=tg, dt=dt)
        out[i] = float(p.estimate_heading()[0])
    return out


def rmse_deg(est, true_th, mask=None):
    e = wrap(np.asarray(est) - np.asarray(true_th))
    if mask is not None:
        e = e[mask]
    return float(np.degrees(np.sqrt(np.mean(e ** 2))))


if __name__ == '__main__':
    for dt in (0.02, 0.002):
        sc, r = calibrate_gain(dt)
        print(f"dt={dt}: calibrated scale {sc:.3f}  (gain {0.045*sc:.4f})  travel ratio {r:.4f}")
