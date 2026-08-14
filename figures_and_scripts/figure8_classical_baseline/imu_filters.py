#!/usr/bin/env python3
"""
imu_filters.py -- shared IMU stream generator + classical attitude filters used by the Fig 3 /
Fig 2B / Fig S2 ring diagnostics (rerun_ring_diagnostics.py).

The IMU front-end is NOT re-implemented here: `imu_rate_average` is imported from the deployed
system (snn_slam_system), so the diagnostics see exactly the IMU the deployed LiveEnvironment
generates. Sensors do not share the estimator's clock -- the MEMS IMU samples at
IMU_OVERSAMPLE/DT = 1 kHz, where the 115 Hz wingbeat is properly represented, and the estimator
consumes the MEAN rate over each 20 ms step (what integrating the IMU at its native rate gives).
A zero-mean tone is therefore attenuated by |sinc(f*DT)| (0.112 at 115 Hz); tones at exact
multiples of 1/DT integrate to zero over a step -- a real physical null, not a sampling artifact.
No synthetic anti-alias filter is involved.

  make_streams : pre-generate the shared IMU streams (true pitch, noisy gyro, accel-pitch
                 measurement, accel-valid mask; optional occlusion window and transient outliers)
  run_cf       : raw complementary filter (gyro integration nudged toward accel pitch)
  run_ekf      : scalar (1D) Kalman filter for pitch (gyro predict, accel-pitch update)

run_cf uses the deployed ALPHA_FUSE (never re-hardcoded). Note it is a *pure* CF baseline: it
omits the deployed accelerometer EMA pre-filter (alpha_acc) and the learned gyro-bias term, which
is intentional -- it is the textbook filter the ring is compared against, not a replica of
forward_step.
"""
import os, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, '..', '..', 'neuro-symbolic-slam')))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, '..', '..', 'neuro-symbolic-slam', 'src')))
from snn_slam_system import wrap_angle, ALPHA_FUSE, imu_rate_average, IMU_OVERSAMPLE

G = 9.81


def make_streams(freq, amp, dt, steps, true_omega, seed, occlusion=None, outliers=None):
    """Pre-generate the shared IMU streams so every estimator sees identical inputs.

    The wingbeat tone is delivered through the DEPLOYED IMU front-end (imu_rate_average): it is
    sampled at the 1 kHz IMU rate and averaged over each estimator step. The wingbeat phase
    relative to the sample clock is arbitrary, so a per-stream random phase is drawn (this does
    not affect the exact nulls, whose integral over a whole number of periods is zero for any
    phase). MEMS white noise is added at the estimator rate, matching the deployed model.

    `outliers`: (prob, magnitude_rad) injects transient accel-pitch spikes (linear-acceleration
    events violating the quasi-static assumption) to probe outlier rejection.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(steps) * dt
    phi = rng.uniform(0.0, 2.0 * np.pi)                                # wingbeat phase vs sample clock
    true_th = wrap_angle(true_omega * t)
    vib = imu_rate_average(freq, amp, dt, steps, phase0=phi)           # gyro vibration (rad/s)
    vib_acc = imu_rate_average(freq, 0.1 * amp, dt, steps, phase0=phi)  # accel vibration
    gyro = true_omega + vib + rng.normal(0, 0.05, steps)               # noisy angular rate
    ax = G * np.sin(true_th) + vib_acc + rng.normal(0, 0.08, steps)
    az = G * np.cos(true_th) - vib_acc + rng.normal(0, 0.08, steps)
    th_accel = np.arctan2(ax, az)
    valid = np.ones(steps, dtype=bool)
    if occlusion is not None:
        t0, t1 = occlusion
        valid[(t >= t0) & (t < t1)] = False                            # accel/gravity blocked
    if outliers is not None:
        prob, mag = outliers
        hit = rng.random(steps) < prob
        th_accel = th_accel + hit * rng.choice([-1, 1], steps) * mag
    return dict(t=t, true_th=true_th, gyro=gyro, th_accel=th_accel, valid=valid)


def run_cf(S, dt):
    """Raw complementary filter: gyro integration corrected toward accel pitch (deployed ALPHA_FUSE)."""
    th = 0.0
    out = np.zeros(len(S['t']))
    for i in range(len(S['t'])):
        th = wrap_angle(th + S['gyro'][i] * dt)
        if S['valid'][i]:
            th = wrap_angle(th + ALPHA_FUSE * wrap_angle(S['th_accel'][i] - th))
        out[i] = th
    return out


def run_ekf(S, dt, Q=1e-4, R=6e-2):
    """Scalar (1D) Kalman filter for pitch: gyro predict, accel-pitch update.

    NOTE: with fixed Q and R the Kalman gain converges to a constant steady-state value
    (K -> 0.040 for these settings); it is 'adaptive' only in its transient. Its outlier
    sensitivity relative to the CF is therefore a consequence of that steady-state gain being
    ~2x ALPHA_FUSE, not of any run-time adaptation. Reported as such.
    """
    th, P = 0.0, 1.0
    out = np.zeros(len(S['t']))
    for i in range(len(S['t'])):
        th = wrap_angle(th + S['gyro'][i] * dt)                    # predict
        P = P + Q
        if S['valid'][i]:                                          # update
            K = P / (P + R)
            th = wrap_angle(th + K * wrap_angle(S['th_accel'][i] - th))
            P = (1.0 - K) * P
        out[i] = th
    return out
