#!/usr/bin/env python3
"""Fig 3 -- classical baselines on identical IMU streams (deployed PoseCANN, deployed dt = 20 ms,
calibrated velocity gain, hardware-rate IMU front-end).

Data: ring_diagnostics_corrected.json (produced by rerun_ring_diagnostics.py).

(A) Across the wingbeat-vibration envelope the gravity-anchored ring tracks the raw complementary
    filter essentially exactly -- the attractor is a faithful readout of the CF it is anchored to,
    neither improving nor degrading it. A 1D EKF is slightly more accurate.
(B) Under transient accelerometer outliers the picture inverts: the fixed-gain CF (and the ring
    that reads it out) stay bounded, while the EKF's heavier steady-state gain (K -> 0.040, twice
    the CF's 1 - alpha_fuse = 0.02) admits the outliers and its error grows. With fixed Q and R the
    Kalman gain converges to a constant, so this is a gain-MAGNITUDE effect, not run-time
    adaptation. The robustness belongs to the LOWER-GAIN complementary fusion, not the attractor.

Uses the shared figstyle house style (Okabe-Ito palette, panel labels, dual PDF+PNG save to the
local dir AND manuscript/figures).
"""
import os, json, sys
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
import figstyle

d = json.load(open(os.path.join(_HERE, 'ring_diagnostics_corrected.json')))
# Colors avoid figstyle's cross-figure semantic roles: green (=loop-closure ON / "ours") and
# vermillion (=IMU dead-reckoning) both carry meanings elsewhere, so tagging the criticized EKF
# green or the CF vermillion would mislead a reader cross-referencing figures. Ours=blue (C_ANCH),
# CF=slate-grey (matches the CF baseline color in Fig 2B), EKF=purple (unused, colorblind-safe).
C_RING, C_CF, C_EKF = figstyle.C_ANCH, figstyle.C_MUTED, figstyle.OKABE['purple']

figstyle.apply()
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.7))

# (A) vibration envelope ------------------------------------------------------
amps = [r['amp'] for r in d['envelope']]
axA.plot(amps, [r['ring_anch'] for r in d['envelope']], 'o-', color=C_RING, lw=2.2, ms=5,
         label='Gravity-anchored ring (ours)')
axA.plot(amps, [r['cf'] for r in d['envelope']], 's--', color=C_CF, lw=1.5, ms=4,
         label='Complementary filter')
axA.plot(amps, [r['ekf'] for r in d['envelope']], '^:', color=C_EKF, lw=1.5, ms=4, label='1D EKF')
axA.set_xscale('log')
axA.set_xlabel('Vibration amplitude (rad/s)')
axA.set_ylabel('Attitude RMSE (deg)')
axA.set_title('Clean vibration: identical inputs')
axA.set_ylim(0, None)
axA.annotate('ring tracks the CF with a\nsmall substrate overhead',
             xy=(amps[-2], d['envelope'][-2]['ring_anch']), xytext=(0.06, 0.70),
             textcoords='axes fraction', fontsize=8, color=C_RING,
             arrowprops=dict(arrowstyle='->', color=figstyle.C_MUTED, lw=0.9,
                             connectionstyle='arc3,rad=0.15'))
axA.legend(loc='upper left')
figstyle.panel(axA, 'A')

# (B) outlier sweep -----------------------------------------------------------
rates = [r['rate'] * 100 for r in d['outliers']]
axB.plot(rates, [r['ring'] for r in d['outliers']], 'o-', color=C_RING, lw=2.2, ms=5,
         label='Gravity-anchored ring (ours)')
axB.plot(rates, [r['cf'] for r in d['outliers']], 's--', color=C_CF, lw=1.5, ms=4,
         label='Complementary filter')
axB.plot(rates, [r['ekf'] for r in d['outliers']], '^:', color=C_EKF, lw=1.5, ms=4, label='1D EKF')
axB.set_xlabel('Transient accel-outlier rate (%/step)')
axB.set_ylabel('Attitude RMSE (deg)')
axB.set_title('Transient accelerometer outliers')
axB.set_ylim(0, None)
# EKF label near its own steep rise; CF/ring label moved into the empty lower-right band (rate
# 18-36, RMSE 0.4-1.1) so it no longer sits across the ring/CF markers at rate 2/5/8. Pointer
# arrows desaturated to slate so they read as leaders, not as extra data series.
axB.annotate("EKF's heavier fixed gain\nadmits the outliers",
             xy=(rates[-2], d['outliers'][-2]['ekf']), xytext=(0.04, 0.88),
             textcoords='axes fraction', fontsize=8, color=C_EKF, ha='left', va='top',
             arrowprops=dict(arrowstyle='->', color=figstyle.C_MUTED, lw=0.9,
                             connectionstyle='arc3,rad=0.2'))
axB.annotate('fixed-gain CF (and the ring\nreading it out) stay bounded',
             xy=(rates[-1], d['outliers'][-1]['cf']), xytext=(0.44, 0.06),
             textcoords='axes fraction', fontsize=8, color=figstyle.C_INK, ha='left', va='bottom',
             arrowprops=dict(arrowstyle='->', color=figstyle.C_MUTED, lw=0.9,
                             connectionstyle='arc3,rad=-0.2'))
figstyle.panel(axB, 'B')

fig.tight_layout()
figstyle.save(fig, __file__, 'snn_slam_classical_baseline')
plt.close(fig)
print('wrote snn_slam_classical_baseline.{pdf,png}')
