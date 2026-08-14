#!/usr/bin/env python3
"""SI Fig S2 -- bump drift on the deployed PoseCANN.

Running (cumulative) RMSE of the pitch-attitude error over 30 s under a persistent gyro bias and
a hardware-rate 115 Hz wingbeat, with a calibrated velocity gain (travel/truth ratio ~1.0). The
unanchored attractor path-integrates the biased gyro and drifts; the gravity anchor bounds it.

Reads the SAME canonical ring-diagnostics file as Fig 3 / Fig 2B
(figure8_classical_baseline/ring_diagnostics_corrected.json, key `bump_drift`). Uses the shared
figstyle house style.
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

SRC = os.path.normpath(os.path.join(_HERE, '..', 'figure8_classical_baseline',
                                    'ring_diagnostics_corrected.json'))
d = json.load(open(SRC))['bump_drift']
t = np.array(d['t']); ra = np.array(d['running_anch']); ru = np.array(d['running_unanch'])

figstyle.apply()
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(t, ru, color=figstyle.C_IMU, lw=2.0,
        label=f'Unanchored ring (drifts) — {ru[-1]:.1f}° at 30 s')
ax.plot(t, ra, color=figstyle.C_ANCH, lw=2.0,
        label=f'Gravity-anchored ring — {ra[-1]:.1f}° at 30 s')
ax.axhline(d['gt_rms_deg'], color=figstyle.C_MUTED, ls=':', lw=1.2)
# Plain text just BELOW the reference line (the old arrow landed on the dotted line in the same
# slate colour and read as a stray marker). Sentence case, not all-caps; Unicode degree.
ax.text(0.6, d['gt_rms_deg'] - 1.6,
        f"RMS of the ground-truth motion ({d['gt_rms_deg']:.1f}°):\n"
        "the value a stationary estimate would report",
        va='top', ha='left', fontsize=7.4, color=figstyle.C_MUTED)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Running pitch-attitude RMSE (deg)')
ax.set_title('Bump drift under a persistent gyro bias', loc='left')
ax.set_xlim(t[0], t[-1])
ax.set_ylim(0, max(66.0, ru.max() * 1.12))
# Legend into the empty band (upper-left, below the reference-line note and above the unanchored
# curve), off the drift ramp it previously sat on top of.
leg = ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.74), fontsize=8, framealpha=1.0)
figstyle.style_legend(leg)

# Zoom inset: the anchored trace's 0.9->1.8 deg excursion is ~1% of the 0-66 deg axis and invisible
# at full scale, so show it on its own 0-3 deg scale. Rendering only -- same data.
axins = ax.inset_axes([0.55, 0.12, 0.40, 0.27])
axins.plot(t, ra, color=figstyle.C_ANCH, lw=1.6)
axins.set_xlim(0, 30); axins.set_ylim(0, 3)
axins.set_xticks([0, 15, 30]); axins.set_yticks([0, 1, 2, 3])
axins.tick_params(labelsize=7)
axins.set_title('gravity-anchored (zoom)', fontsize=7.4, color=figstyle.C_ANCH, pad=2)
axins.grid(True, axis='y')
fig.tight_layout()
figstyle.save(fig, __file__, 'snn_slam_bump_drift')
plt.close(fig)
print(f"wrote snn_slam_bump_drift.{{pdf,png}}  (anchored {ra[-1]:.2f} deg, unanchored {ru[-1]:.2f} deg,"
      f" reduction {100*(1-ra[-1]/ru[-1]):.1f}%)")
