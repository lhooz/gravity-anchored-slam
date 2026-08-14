"""
simulate_sensory_deprivation.py — reference-dropout (visual-occlusion) diagnostic → Fig 2C data.

Convention: theta is the body's IN-PLANE PITCH ATTITUDE in a vertical (sagittal) plane. Gravity
is yaw-invariant but projects onto pitch as ax = g*sin(theta), az = g*cos(theta), so
arctan2(ax, az) is a valid absolute pitch reference.

Experiment. A persistent MEMS gyro bias is ALWAYS present. While the (idealized) absolute-attitude
reference is available it continuously corrects both arms, so both enter the occlusion window at
truth (no boundary discontinuity). During the 5 s occlusion:
  * UNANCHORED = pure gyro dead-reckoning: integrates the biased gyro and DRIFTS without bound.
  * ANCHORED   = the DEPLOYED complementary filter: gyro integration nudged toward the
    accelerometer gravity reference with weight ALPHA_FUSE. The accelerometer still sees gravity
    under occlusion, so the estimate stays BOUNDED (steady-state offset ~ bias * dt/ALPHA_FUSE).
The ring attractor (CANN) is the readout substrate; the contrast is measured ENTIRELY inside the
occlusion window (no ground truth is injected into either arm there).

Fidelity to the deployed system (all three were wrong in an earlier version of this script):
  * Runs at the DEPLOYED estimator step DT = 0.02 s. ALPHA_FUSE is a PER-STEP gain, so running at
    a finer dt with the same gain silently makes the filter faster than the one that is deployed.
  * The accelerometer carries ONLY what the deployed model gives it (white MEMS noise + the
    wingbeat tone). An earlier version added a large invented accel tilt-bias random walk that
    supplied ~93% of the reported residual; it exists nowhere in snn_slam_system.
  * The wingbeat reaches the estimator through the DEPLOYED IMU front-end (imu_rate_average:
    1 kHz IMU, mean rate per estimator step), not as a tone point-sampled at 50 Hz.
Reported over N_SEEDS seeds with a 95% CI — the earlier single-seed number was one lucky draw.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, '..', '..', 'neuro-symbolic-slam')))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, '..', '..', 'neuro-symbolic-slam', 'src')))

import figstyle
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from snn_slam_system import (wrap_angle, ALPHA_FUSE, imu_rate_average,
                            VIB_FREQ, VIB_GYRO_AMP, VIB_ACC_AMP, GYRO_BIAS_STD)
from snn_pose_cann import ring_readout, neural_field_update
from sparse_forest import DT

G = 9.81
N_SEEDS = 10
OCC0, OCC1 = 5.0, 10.0
ACC_MEMS_NOISE = 0.05     # m/s^2, deployed
GYRO_MEMS_NOISE = 0.02    # rad/s, white gyro noise
K_TRACK, SIG_TRACK = 60.0, 0.20   # current locking the ring bump onto the active estimate
K_VIS = 0.30              # per-step visual-correction gain when the reference is available


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def run_seed(seed, pose):
    """One seed-pinned run. Returns (t, true, unanchored, anchored) at the deployed step DT."""
    rng = np.random.default_rng(seed)
    W_ring, RING_TAU_U = pose.W_ring, pose.RING_TAU_U
    RING_N = W_ring.shape[0]
    angles = jnp.arange(RING_N, dtype=jnp.float32) * (2.0 * jnp.pi / RING_N)

    dt = float(DT)                       # <-- the DEPLOYED estimator step (ALPHA_FUSE is per-step)
    steps = int(15.0 / dt)
    t = np.arange(steps) * dt
    true_th = wrap_angle(0.8 * np.sin(0.5 * t))
    true_w = 0.4 * np.cos(0.5 * t)

    gyro_bias = float(rng.normal(0.0, GYRO_BIAS_STD))      # deployed per-trial MEMS bias
    phi = rng.uniform(0.0, 2 * np.pi)                       # wingbeat phase vs sample clock
    vib_g = imu_rate_average(VIB_FREQ, VIB_GYRO_AMP, dt, steps, phase0=phi)
    vib_a = imu_rate_average(VIB_FREQ, VIB_ACC_AMP, dt, steps, phase0=phi)
    gyro = true_w + gyro_bias + vib_g + rng.normal(0, GYRO_MEMS_NOISE, steps)
    ax = G * np.sin(true_th) + vib_a + rng.normal(0, ACC_MEMS_NOISE, steps)
    az = G * np.cos(true_th) + rng.normal(0, ACC_MEMS_NOISE, steps)
    th_accel = np.arctan2(ax, az)                           # absolute gravity reference
    vision = _wrap(true_th + rng.normal(0, 0.02, steps))    # idealized ABSOLUTE attitude reference

    def arm(use_gravity):
        diff0 = jnp.mod(angles + jnp.pi, 2 * jnp.pi) - jnp.pi
        u = 0.5 * jnp.exp(-(diff0 ** 2) / (2 * 0.15 ** 2))[None, :]
        u = u / (u.sum() + 1e-8)
        r = jnp.clip(u, 0, 1.0)
        est = float(true_th[0])
        out = np.zeros(steps)
        for i in range(steps):
            occluded = (t[i] >= OCC0) and (t[i] <= OCC1)
            est = float(wrap_angle(est + gyro[i] * dt))                       # gyro propagate
            if use_gravity:                                                    # DEPLOYED CF form
                est = float(wrap_angle(est + ALPHA_FUSE * wrap_angle(th_accel[i] - est)))
            if not occluded:                                                   # absolute ref present
                est = float(wrap_angle(est + K_VIS * wrap_angle(vision[i] - est)))
            d = angles[None, :] - jnp.array([est])[:, None]
            dw = jnp.mod(d + jnp.pi, 2 * jnp.pi) - jnp.pi
            I = K_TRACK * jnp.exp(-(dw ** 2) / (2 * SIG_TRACK ** 2))
            for _ in range(10):
                u = neural_field_update(u, r + 1e-8, W_ring, I, dt=dt / 10.0, tau=RING_TAU_U)
                r = jnp.clip(u, 0, 1.0)
            out[i] = float(ring_readout(r)[0])
        return out

    return t, true_th, arm(False), arm(True)


def main():
    print("=" * 70)
    print("  Fig 2C: reference-dropout diagnostic (deployed dt, deployed CF, deployed IMU)")
    print("=" * 70)
    from snn_slam_system import SNNSLAMSystem
    pose = SNNSLAMSystem(random.PRNGKey(42)).pose
    pose.reset(1)

    rm_un, rm_an = [], []
    # Keep EVERY seed's trace + signed error, not just one draw. The figure previously plotted seed 0
    # while annotating the n-seed mean; seed 0 happened to be the 4th-lowest of 10 (unanchored 1.75 deg
    # vs a 2.86 deg mean), so the displayed effect understated the reported statistic. Panel 2C now
    # renders the across-seed mean |error| with a 95% CI band, and the inset trace uses the MEDIAN seed.
    tr_un, tr_an, er_un, er_an = [], [], [], []
    t = tru = None
    for s in range(N_SEEDS):
        t, tru, un, an = run_seed(1000 + s, pose)
        m = (t >= OCC0) & (t <= OCC1)
        eu = np.degrees(_wrap(un - tru))            # signed pitch error, full time series (deg)
        ea = np.degrees(_wrap(an - tru))
        ru = float(np.sqrt(np.mean(eu[m] ** 2)))    # identical to the previous degrees(RMS(wrap(...)))
        ra = float(np.sqrt(np.mean(ea[m] ** 2)))
        rm_un.append(ru); rm_an.append(ra)
        tr_un.append(un); tr_an.append(an); er_un.append(eu); er_an.append(ea)
        print(f"  seed {s}: unanchored {ru:6.2f} deg | anchored {ra:5.2f} deg")

    rm_un, rm_an = np.array(rm_un), np.array(rm_an)
    tr_un, tr_an = np.array(tr_un), np.array(tr_an)
    er_un, er_an = np.array(er_un), np.array(er_an)
    ci = lambda a: 1.96 * a.std(ddof=1) / np.sqrt(len(a))
    mu, ma = rm_un.mean(), rm_an.mean()
    red = 100 * (1 - ma / mu)
    # representative seed = MEDIAN by unanchored window RMSE (was: seed 0, an arbitrary draw)
    rep = int(np.argsort(rm_un)[len(rm_un) // 2])
    print(f"\n  OCCLUSION-WINDOW RMSE over n={N_SEEDS} seeds")
    print(f"    unanchored {mu:.1f} +/- {ci(rm_un):.1f} deg")
    print(f"    anchored   {ma:.1f} +/- {ci(rm_an):.1f} deg   ({red:.0f}% reduction)")
    print(f"    representative (median) seed = {rep}  "
          f"(unanchored {rm_un[rep]:.2f} deg, anchored {rm_an[rep]:.2f} deg)")

    un, an = tr_un[rep], tr_an[rep]
    np.savez(os.path.join(_HERE, "sensory_deprivation_data.npz"),
             time=t, true_headings=tru, headings_unanchored=un, headings_anchored=an,
             # NEW: every seed, so the figure can show the across-seed mean +/- 95% CI rather than one draw
             err_unanchored_seeds=er_un, err_anchored_seeds=er_an,     # (n_seeds, steps), deg, signed
             traj_unanchored_seeds=tr_un, traj_anchored_seeds=tr_an,   # (n_seeds, steps), rad
             rep_seed=rep,
             rmse_unanchored_seeds=rm_un, rmse_anchored_seeds=rm_an,
             rmse_unanchored_mean=mu, rmse_anchored_mean=ma,
             rmse_unanchored_ci=ci(rm_un), rmse_anchored_ci=ci(rm_an), n_seeds=N_SEEDS)

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.axvspan(OCC0, OCC1, color=figstyle.OKABE['grey'], alpha=0.18, lw=0, label='Reference blocked')
    ax.plot(t, np.degrees(tru), color=figstyle.C_TRUE, ls='--', lw=1.6, label='True pitch')
    ax.plot(t, np.degrees(un), color=figstyle.C_IMU, lw=1.8, label='Unanchored (dead-reckoning)')
    ax.plot(t, np.degrees(an), color=figstyle.C_ANCH, lw=1.8, label='Gravity-anchored')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Pitch attitude (deg)')
    ax.set_xlim(0, t[-1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=8.5)
    ax.annotate(f'occlusion-window RMSE ($n={N_SEEDS}$)\n'
                f'unanchored {mu:.1f}$^\\circ$ vs anchored {ma:.1f}$^\\circ$',
                xy=(0.985, 0.97), xycoords='axes fraction', ha='right', va='top', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#d3d8dd', lw=0.8))
    fig.savefig(os.path.join(_HERE, 'snn_slam_sensory_deprivation.pdf'))   # local preview only
    fig.savefig(os.path.join(_HERE, 'snn_slam_sensory_deprivation.png'), dpi=200)
    plt.close(fig)
    print("DONE")


if __name__ == '__main__':
    main()
