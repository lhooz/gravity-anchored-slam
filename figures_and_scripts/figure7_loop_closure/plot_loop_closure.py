#!/usr/bin/env python3
"""
plot_loop_closure.py — Figure 4: loop closure corrects accumulated GLOBAL drift on
revisit-rich courses, under a realistic event+ToF visual-odometry velocity-error model.

Two panels, both unambiguous (the earlier single-circle overlay was a classic loop-closure
visualization pitfall -- on a many-lap circuit the smooth-but-phase-drifted LC-OFF track can
look "cleaner" than the globally-corrected LC-ON track, even though LC-ON has the lower ATE).
We therefore lead with the quantitative views:
  A  ATE bars (IMU / LC-OFF / LC-ON), paired over seeds, 95% bootstrap CIs.
  B  Position error vs time (mean, 95% CI band) -- LC-ON is lowest at every horizon.
The spatial before/after map with the fired loop-closure edges is the separate
topological-map figure (plot_topomap.py).

Reads the compare-lc archive produced by:
  scripts/slam_variance.py --compare-lc --course circuit --seeds N --steps S
Inputs (this dir, or $SLAM_FIG_DATA): loop_closure_results.json + *_timeseries.npz
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.environ.get('SLAM_FIG_DATA', _HERE)   # override to render from a re-run dir before archiving
_JSON = os.path.join(_DATA, 'loop_closure_results.json')
_NPZ  = os.path.join(_DATA, 'loop_closure_results_timeseries.npz')
import sys; sys.path.insert(0, os.path.join(_HERE, '..'))
import figstyle

C_IMU = figstyle.OKABE['grey']   # gray (IMU dead-reckoning)
C_OFF = figstyle.C_OFF           # Okabe-Ito orange (loop closure OFF)
C_ON  = figstyle.C_ON            # Okabe-Ito green  (loop closure ON)
C_GT  = figstyle.C_TRUE          # near-black ground truth


def main():
    with open(_JSON) as f:
        data = json.load(f)
    s = data['summary']
    d = np.load(_NPZ)

    imu   = s['ate_imu_cm']
    off   = s['ate_cl_off_cm']
    on    = s['ate_cl_on_cm']
    delta = s['lc_delta_cm']
    lcs   = s['lcs']
    n     = s['n_seeds']

    figstyle.apply()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 3.7))

    # ---- Panel A: ATE bar chart with 95% bootstrap CI error bars ----
    labels = ['IMU\ndead-reckoning', 'SLAM\n(LC OFF)', 'SLAM\n(LC ON)']
    means  = [imu['mean'], off['mean'], on['mean']]
    cis    = [imu['ci95'], off['ci95'], on['ci95']]
    colors = [C_IMU, C_OFF, C_ON]
    yerr = np.array([[m - c[0] for m, c in zip(means, cis)],
                     [c[1] - m for m, c in zip(means, cis)]])
    xs = np.arange(3)
    axA.bar(xs, means, yerr=yerr, color=colors, capsize=6, width=0.62,
            error_kw={'elinewidth': 1.6, 'ecolor': '#2c3e50'}, zorder=3)
    axA.set_xticks(xs); axA.set_xticklabels(labels)
    axA.set_ylabel('Absolute trajectory error (cm)')
    axA.set_title(f'ATE on revisit-rich circuit (n={n})', loc='left')
    figstyle.panel(axA, 'A')
    for x, m in zip(xs, means):
        axA.text(x, m + yerr[1][int(x)] + 0.03 * max(means), f'{m:.1f}', ha='center', fontsize=10)
    # paired LC improvement annotation
    axA.annotate(f"loop closure: $-${delta['mean']:.1f} cm\n(95% CI [{delta['ci95'][0]:.1f}, {delta['ci95'][1]:.1f}])",
                 xy=(0.97, 0.97), xycoords='axes fraction', ha='right', va='top', fontsize=8.2,
                 bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#d3d8dd', lw=0.8))

    # ---- Panel B: mean position error vs time, LC ON vs OFF, with 95% CI bands ----
    dt = float(d['dt']); t = np.arange(int(d['n_steps'])) * dt
    def band(ts_key, color, label, ls='-'):
        ts = d[ts_key] * 100.0  # cm
        m = ts.mean(axis=0)
        rs = np.random.RandomState(0)
        idx = rs.randint(0, ts.shape[0], size=(2000, ts.shape[0]))
        boot = ts[idx].mean(axis=1)
        lo = np.percentile(boot, 2.5, axis=0); hi = np.percentile(boot, 97.5, axis=0)
        axB.plot(t, m, color=color, lw=1.9, ls=ls, label=label)
        axB.fill_between(t, lo, hi, color=color, alpha=0.18, lw=0)
    # IMU grey and LC-OFF orange are near-isoluminant and interleave over t=0-25 s, so they merge
    # in a greyscale print exactly where they cross. Dash the IMU line to add a non-color channel.
    band('imu_err_ts', C_IMU, 'IMU dead-reckoning', ls=(0, (5, 2)))
    band('cl_off_err_ts', C_OFF, 'SLAM, LC OFF')
    band('cl_on_err_ts', C_ON, 'SLAM, LC ON')
    axB.set_xlabel('time (s)'); axB.set_ylabel('position error (cm)')
    axB.set_title('Error vs time (mean, 95% CI band)', loc='left')
    figstyle.panel(axB, 'B')
    axB.legend(loc='upper left', fontsize=8.5)

    # No suptitle: it restated the caption's opening sentence and repeated the mean-246 / precision
    # numbers the caption already carries (journals discourage a figure title baked into the raster).
    plt.tight_layout()
    figstyle.save(fig, __file__, 'snn_slam_loop_closure')
    plt.close(fig)


if __name__ == '__main__':
    main()
