#!/usr/bin/env python3
"""Merged main-text attitude figure (Fig. 2): the gravity anchor stabilizes body-pitch
attitude across three regimes.

  (A) With reliable vision present -- clean single-variable gravity-anchor ablation
      (same spiking network, seed, and forward pass; only the gravity injection toggled):
      mean heading RMSE, anchor OFF vs ON, on the open random course and the longer
      revisit-rich circuit.
  (B) Across the full wingbeat-vibration envelope (IMU-only diagnostic ablation):
      heading RMSE vs vibration amplitude (mean over 10-260 Hz), anchored vs unanchored
      vs a raw complementary-filter (CF) baseline.
  (C) Under visual occlusion (event cues blocked 5-10 s): the anchored arm keeps fusing
      the accelerometer's gravity reference and stays bounded; the unanchored arm
      dead-reckons on the biased gyro and drifts.

Panels B and C read the frozen data produced by the S1 vibration-phase and occlusion
runs; panel A uses the clean-ablation heading RMSE reported in SI Appendix.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import figstyle
figstyle.apply()
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SCR  = os.path.normpath(os.path.join(HERE, '..'))

# ── data ────────────────────────────────────────────────────────────────────
# (A) clean single-variable gravity-anchor ablation, vision present.
# Loaded from the committed JSON emitted by clean_anchor_ablation.py (NOT hard-coded), so the
# flagship attitude numbers are regenerated from the experiment (with 95% CI error bars).
import json
_abl = json.load(open(os.path.join(HERE, 'anchor_ablation_results.json')))
_n_abl = _abl['n_seeds']
courses    = ['Open random\ncourse', 'Revisit-rich\ncircuit']
head_off   = [_abl['random']['mean']['h_off'], _abl['circuit']['mean']['h_off']]   # anchor OFF (K_g=0), deg
head_on    = [_abl['random']['mean']['h_on'],  _abl['circuit']['mean']['h_on']]    # anchor ON, deg
head_off_ci = [_abl['random']['ci95']['h_off'], _abl['circuit']['ci95']['h_off']]
head_on_ci  = [_abl['random']['ci95']['h_on'],  _abl['circuit']['ci95']['h_on']]

# (B) vibration envelope -- read the SAME canonical ring-diagnostics file that Fig 3 and Fig S2
# use (figure8_classical_baseline/ring_diagnostics_corrected.json, produced by
# rerun_ring_diagnostics.py on the deployed PoseCANN), so Panel B, Fig 3A and Fig S2 can never
# diverge. Each envelope row carries the per-frequency RMSE and its mean over frequency.
_env = json.load(open(os.path.join(SCR, 'figure8_classical_baseline',
                                   'ring_diagnostics_corrected.json')))['envelope']
amp = np.array([r['amp'] for r in _env])                       # rad/s
an  = np.array([r['ring_anch'] for r in _env])                 # mean over frequency
un  = np.array([r['ring_unanch'] for r in _env])
cf  = np.array([r['cf'] for r in _env])
_anch_pf = [[p['ring_anch'] for p in r['per_freq']] for r in _env]
an_lo = np.array([min(pf) for pf in _anch_pf])                 # range over frequency
an_hi = np.array([max(pf) for pf in _anch_pf])

# (C) occlusion. Plot the ABSOLUTE pitch error vs time, averaged over all n seeds with a 95% CI band.
# This "zooms in" past the large common-mode pitch sinusoid: the raw +/-45 deg attitude traces differ
# by only ~1-3 deg during occlusion and sat on top of each other, hiding the very effect the panel
# exists to show. Plotting the error (i) makes the divergence visible, (ii) matches Panels A and B
# (both already show error/RMSE), and (iii) uses ALL seeds -- the earlier version drew one seed's raw
# attitude that happened to be below-median (unanchored 1.75 deg vs the 2.86 deg mean it annotated).
occ = np.load(os.path.join(SCR, 'figure6_sensory_deprivation', 'sensory_deprivation_data.npz'))
t     = occ['time']
eu    = np.abs(occ['err_unanchored_seeds'])          # (n_seeds, steps), |pitch error| in deg
ea    = np.abs(occ['err_anchored_seeds'])
eu_m, ea_m = eu.mean(0), ea.mean(0)
_ci_t = lambda x: 1.96 * x.std(0, ddof=1) / np.sqrt(x.shape[0])
eu_ci, ea_ci = _ci_t(eu), _ci_t(ea)
OCC0, OCC1 = 5.0, 10.0
rmse_u  = float(occ['rmse_unanchored_mean']); rmse_a  = float(occ['rmse_anchored_mean'])
ci_u    = float(occ['rmse_unanchored_ci']);   ci_a    = float(occ['rmse_anchored_ci'])
n_occ   = int(occ['n_seeds'])

# ── figure ──────────────────────────────────────────────────────────────────
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(11.2, 3.35))
fig.subplots_adjust(wspace=0.34, bottom=0.20, top=0.86, left=0.06, right=0.985)

# (A) grouped bars ------------------------------------------------------------
x = np.arange(len(courses)); w = 0.36
ebar = dict(ecolor=figstyle.C_INK, elinewidth=0.9, capsize=2.5, capthick=0.9)
bO = axA.bar(x - w/2, head_off, w, yerr=head_off_ci, error_kw=ebar, color=figstyle.C_IMU,
             label=f'Anchor off ($K_g{{=}}0$)', edgecolor='white', linewidth=0.6, zorder=3)
bN = axA.bar(x + w/2, head_on,  w, yerr=head_on_ci, error_kw=ebar, color=figstyle.C_ANCH,
             label='Anchor on', edgecolor='white', linewidth=0.6, zorder=3)
axA.set_xticks(x); axA.set_xticklabels(courses)
axA.set_ylabel('Pitch attitude error (deg)')   # clean_anchor_ablation reports mean absolute error
axA.set_title(f'Attitude with vision present ($n={_n_abl}$)', fontsize=10.5)
axA.set_ylim(0, float(max(np.array(head_off) + np.array(head_off_ci))) * 1.28)
for xi, off, on, oci, nci in zip(x, head_off, head_on, head_off_ci, head_on_ci):
    axA.text(xi - w/2, off + oci + 1.0, f'{off:.1f}', ha='center', va='bottom',
             fontsize=8.5, color=figstyle.C_IMU)
    axA.text(xi + w/2, on + nci + 1.0, f'{on:.1f}', ha='center', va='bottom',
             fontsize=8.5, color=figstyle.C_ANCH)
    red = 100 * (off - on) / off
    # Reduction label sits well ABOVE both value labels (which top out at cap+1.0) and centered over
    # the group, so it never abuts the '8.7'/'47.2' value labels on either the short or tall bars.
    axA.annotate(f'−{red:.0f}%', xy=(xi, max(off + oci, on + nci) + 6.0), ha='center',
                 va='bottom', fontsize=8.5, fontweight='bold', color=figstyle.C_INK)
axA.legend(loc='upper left', fontsize=8.2, handlelength=1.2)
figstyle.panel(axA, 'A')

# (B) vibration sweep ---------------------------------------------------------
axB.fill_between(amp, an_lo, an_hi, color=figstyle.C_ANCH, alpha=0.14, lw=0,
                 label='anchored range (10--260 Hz)')
axB.semilogx(amp, un, 'o-', color=figstyle.C_IMU,  ms=4, label='Unanchored')
axB.semilogx(amp, an, 'o-', color=figstyle.C_ANCH, ms=4, label='Gravity-anchored (mean)')
axB.semilogx(amp, cf, 's--', color=figstyle.C_MUTED, ms=3.5, lw=1.4, label='Raw CF baseline')
axB.set_xlabel('Vibration amplitude (rad/s)')
axB.set_ylabel('Pitch attitude RMSE (deg)')
axB.set_title('Across vibration envelope', fontsize=10.5)
axB.set_ylim(0, max(8.0, float(un.max()) * 1.2))
# Annotation labels the steep orange rise from the clear top-centre band (arrow points down to the
# curve) instead of floating over it; legend moves high-left, clear of the (low) left end of the
# orange curve so it no longer buries the drift-onset knee. (\rightarrow via mathtext: the fallback
# font has no U+2192 glyph, unlike the degree/approx signs.)
axB.annotate('path-integrates\nvibration $\\rightarrow$ drifts',
             xy=(amp[-4], un[-4]), xytext=(0.66, 0.97), textcoords='axes fraction',
             fontsize=7.2, color=figstyle.C_IMU, ha='left', va='top',
             arrowprops=dict(arrowstyle='->', color=figstyle.C_IMU, lw=0.8,
                             connectionstyle='arc3,rad=0.25'))
axB.annotate('anchored ring tracks the raw CF',
             xy=(amp[3], an[3]), xytext=(0.06, 0.10), textcoords='axes fraction',
             fontsize=7.2, color=figstyle.C_ANCH, ha='left',
             arrowprops=dict(arrowstyle='->', color=figstyle.C_ANCH, lw=0.8))
axB.legend(loc='upper left', bbox_to_anchor=(0.02, 1.0), fontsize=7.4, handlelength=1.4,
           framealpha=1.0)
figstyle.panel(axB, 'B')

# (C) occlusion: |pitch error| vs time, mean +/- 95% CI over n seeds ----------
axC.axvspan(OCC0, OCC1, color=figstyle.OKABE['grey'], alpha=0.16, lw=0, zorder=0)
axC.fill_between(t, eu_m - eu_ci, eu_m + eu_ci, color=figstyle.C_IMU,  alpha=0.16, lw=0, zorder=2)
axC.fill_between(t, ea_m - ea_ci, ea_m + ea_ci, color=figstyle.C_ANCH, alpha=0.16, lw=0, zorder=3)
axC.plot(t, eu_m, color=figstyle.C_IMU,  lw=1.8, label='Unanchored', zorder=4)
axC.plot(t, ea_m, color=figstyle.C_ANCH, lw=1.8, label='Gravity-anchored', zorder=5)
axC.set_xlabel('Time (s)')
axC.set_ylabel('Absolute pitch error (deg)')
axC.set_title('Under visual occlusion', fontsize=10.5)
axC.set_xlim(t.min(), t.max())
axC.set_ylim(0, max(eu_m + eu_ci) * 1.32)          # headroom for the RMSE box (top-right)
# "reference blocked" sits in the clear mid-band region: above the unanchored ramp (~1.5 deg at
# mid-window) and below the top-right RMSE box, so nothing overlaps a curve or the box.
axC.text(7.5, axC.get_ylim()[1] * 0.60, 'reference\nblocked', ha='center', va='center',
         fontsize=7.8, color=figstyle.C_MUTED, zorder=6)
axC.annotate(f'window RMSE ($n={n_occ}$)\n'
             f'{rmse_u:.1f}±{ci_u:.1f}° vs {rmse_a:.1f}±{ci_a:.1f}°',
             xy=(0.975, 0.97), xycoords='axes fraction', ha='right', va='top',
             fontsize=7.8, color=figstyle.C_INK,
             bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#d3d8dd', lw=0.8), zorder=7)
axC.legend(loc='upper left', fontsize=7.8, handlelength=1.6, ncol=1)
figstyle.panel(axC, 'C')

out = figstyle.save(fig, __file__, 'snn_slam_attitude_anchor')
print('MERGED ATTITUDE FIGURE:', out)
print(f'  (A) off/on random {head_off[0]}/{head_on[0]}  circuit {head_off[1]}/{head_on[1]}')
print(f'  (B) anchored {an.min():.2f}-{an.max():.2f}  unanchored {un.mean():.1f}  cf {cf.min():.2f}-{cf.max():.2f}')
print(f'  (C) occlusion-window RMSE  unanchored {rmse_u:.1f}  anchored {rmse_a:.1f}')
