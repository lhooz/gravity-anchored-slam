#!/usr/bin/env python3
"""
plot_monte_carlo.py — Publication-quality Monte Carlo SLAM trajectory variance figure

Loads the variance results from slam_variance.py and generates:
  Panel A: Position error (RMSE) over time (3 lines: IMU, Open-Loop SNN, Closed-Loop SNN) with 95% CI shading
  Panel B: Representative (median) 2D trajectory showing GT, IMU, OL, CL paths

Usage:
    python plot_monte_carlo.py [--data PATH_TO_NPZ] [--json PATH_TO_JSON]
"""
import os
os.environ['MPLBACKEND'] = 'Agg'

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Repo-relative paths (no hardcoded user-specific absolute paths)
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_NPZ = os.path.join(_HERE, 'variance_results_timeseries.npz')
_DEFAULT_JSON = os.path.join(_HERE, 'variance_results.json')
sys.path.insert(0, os.path.join(_HERE, '..'))
import figstyle

# Styling is owned entirely by figstyle.apply() (called in main), so this figure matches the rest
# of the suite. A previous local plt.rcParams.update() here set larger, inconsistent type sizes and
# a dotted grid; it was dead code (figstyle.apply ran after it and overrode the same keys) and is
# removed to avoid the appearance of a per-figure style fork.

# ── Color palette (colorblind-safe, Okabe-Ito) ──────────────────────────────
C_IMU  = figstyle.C_IMU    # Okabe-Ito vermillion
C_OL   = figstyle.C_OL     # Okabe-Ito orange
C_CL   = figstyle.C_CL     # Okabe-Ito green
C_GT   = figstyle.C_TRUE   # black
C_GRID = '#CCCCCC'


def main():
    parser = argparse.ArgumentParser(description='Plot Monte Carlo SLAM variance results')
    parser.add_argument('--data', type=str,
                        default=_DEFAULT_NPZ,
                        help='Path to the timeseries .npz file')
    parser.add_argument('--json', type=str,
                        default=_DEFAULT_JSON,
                        help='Path to the summary JSON file')
    args = parser.parse_args()
    figstyle.apply()

    # ── Load data ────────────────────────────────────────────────────────────
    d = np.load(args.data)
    imu_err = d['imu_err_ts'] * 100.0   # convert to cm
    ol_err  = d['ol_err_ts']  * 100.0
    cl_err  = d['cl_err_ts']  * 100.0
    n_seeds = imu_err.shape[0]
    n_steps = int(d['n_steps'])
    dt      = float(d['dt'])
    time_s  = np.arange(n_steps) * dt

    rep_gt  = d['rep_gt']
    rep_imu = d['rep_imu']
    rep_ol  = d['rep_ol']
    rep_cl  = d['rep_cl']

    # ── Statistics ───────────────────────────────────────────────────────────
    mean_imu, std_imu = np.mean(imu_err, axis=0), np.std(imu_err, axis=0, ddof=1)
    mean_ol,  std_ol  = np.mean(ol_err,  axis=0), np.std(ol_err,  axis=0, ddof=1)
    mean_cl,  std_cl  = np.mean(cl_err,  axis=0), np.std(cl_err,  axis=0, ddof=1)

    from scipy import stats
    _tcrit = float(stats.t.ppf(0.975, n_seeds - 1))   # Student-t (sample std, ddof=1) for n_seeds
    ci95 = lambda std: _tcrit * std / np.sqrt(n_seeds)

    # ── Create figure ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                                    gridspec_kw={'width_ratios': [1.3, 1], 'wspace': 0.32})

    # ─── Panel A: ATE over time with 95% CI ─────────────────────────────────
    ax1.plot(time_s, mean_imu, color=C_IMU, label='IMU dead-reckoning',   zorder=3)
    ax1.fill_between(time_s, mean_imu - ci95(std_imu), mean_imu + ci95(std_imu),
                     color=C_IMU, alpha=0.12, zorder=1)

    ax1.plot(time_s, mean_cl, color=C_CL, lw=2.4, label='SNN (anchor on)',  zorder=3)
    ax1.fill_between(time_s, mean_cl - ci95(std_cl), mean_cl + ci95(std_cl),
                     color=C_CL, alpha=0.12, zorder=1)
    # OL ~= CL on the random course; draw OL dashed ON TOP so both read.
    ax1.plot(time_s, mean_ol, color=C_OL, linestyle='--', dashes=(5, 2), lw=1.8,
             label='SNN (anchor off)',  zorder=6)
    ax1.fill_between(time_s, mean_ol - ci95(std_ol), mean_ol + ci95(std_ol),
                     color=C_OL, alpha=0.10, zorder=1)

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position error (cm)')
    ax1.set_title(f'Mean position error over time  (n = {n_seeds} trials)', loc='left')
    figstyle.panel(ax1, 'A')
    ax1.legend(loc='upper left', frameon=True)
    ax1.set_xlim(time_s[0], time_s[-1])
    ax1.set_ylim(bottom=0)

    # ─── Panel B: 2D trajectory map (representative/median trial) ────────────────────────────
    ax2.plot(rep_imu[:, 0], rep_imu[:, 1], color=C_IMU, lw=1.2, label='IMU', zorder=2)
    ax2.plot(rep_cl[:, 0],  rep_cl[:, 1],  color=C_CL,  lw=2.4, label='SNN (anchor on)', zorder=3)
    ax2.plot(rep_ol[:, 0],  rep_ol[:, 1],  color=C_OL,  lw=1.4, linestyle='--', dashes=(5, 2),
             label='SNN (anchor off)', zorder=4)
    ax2.plot(rep_gt[:, 0],  rep_gt[:, 1],  '--', color=C_GT,  lw=1.6, label='Ground Truth', zorder=5)

    # Start/end markers. Start is white-filled (not 'limegreen') so the only green in the panel is
    # the C_CL 'anchor on' series; End stays black X.
    ax2.plot(rep_gt[0, 0], rep_gt[0, 1], 'o', color='white', ms=8, zorder=6,
             markeredgecolor='k', markeredgewidth=1.0, label='Start')
    ax2.plot(rep_gt[-1, 0], rep_gt[-1, 1], 'X', color='k', ms=8, zorder=6, label='End')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Representative trial (ATE ≈ mean)', loc='left')
    figstyle.panel(ax2, 'B')
    ax2.legend(loc='upper right', fontsize=7.5, frameon=True)
    # Square window centred on the data BOUNDING-BOX MIDPOINT (not the mean) so no trajectory
    # point can ever be clipped, with padding for legend clearance; then force a square box so
    # panel B renders the same height as panel A (no vertical collapse from equal-aspect).
    _allp = np.concatenate([rep_gt, rep_imu, rep_ol, rep_cl], axis=0)
    _xmid = 0.5 * (_allp[:, 0].min() + _allp[:, 0].max())
    _ymid = 0.5 * (_allp[:, 1].min() + _allp[:, 1].max())
    _half = 0.5 * max(np.ptp(_allp[:, 0]), np.ptp(_allp[:, 1])) * 1.18  # 18% pad (legend clearance)
    ax2.set_xlim(_xmid - _half, _xmid + _half); ax2.set_ylim(_ymid - _half, _ymid + _half)
    # adjustable='datalim' (not set_box_aspect) lets the axes BOX fill the full subplot cell so panel
    # B shares panel A's top/bottom edges; equal aspect is preserved by expanding the (already square)
    # limits, which only grows the view, so no trajectory point is clipped.
    ax2.set_aspect('equal', adjustable='datalim')
    figstyle.no_grid(ax2)   # spatial map: a y-only grid would be misleading

    # ── Save (vector PDF + PNG, shared style) ─────────────────────────────────
    figstyle.save(fig, __file__, 'snn_slam_monte_carlo_drift')
    plt.close(fig)


if __name__ == '__main__':
    main()
