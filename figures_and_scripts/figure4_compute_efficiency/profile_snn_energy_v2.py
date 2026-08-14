#!/usr/bin/env python3
"""
profile_snn_energy_v2.py -- dt-INDEPENDENT synaptic-operation / energy estimate.

WHY
---
The original estimate derived each layer's "firing rate" as

    f_layer = mean(fraction of units with activity > 0) x (1 / DT)

i.e. "one spike per active unit per simulation step", and additionally multiplied the ring's
recurrent SOPs by RING_SUBSTEPS = 10 (a numerical-integration choice). Both are properties of
the SIMULATOR, not of the network:

  * the active fraction is a shape property of the activity bump and is dt-invariant
    (measured 0.1313 / 0.1300 / 0.1305 at dt = 20 / 10 / 5 ms),
  * so f_layer, and hence SOPs and power, scale LINEARLY with 1/dt.

Consequently the published figure (9.7e6 SOPs/s, 0.12-0.25 mW at dt = 20 ms) becomes
0.49-1.01 mW if the IMU is merely sampled fast enough not to alias the 115 Hz wingbeat, and
2.5-5.0 mW at 1 kHz -- for no physical reason. A neuron's firing rate is set by its input, not
by the integrator's timestep.

WHAT THIS DOES INSTEAD
----------------------
An explicit, disclosed rate-to-spike mapping. Each layer's activity r is mapped linearly to a
firing rate with a per-layer gain chosen so that the layer's PEAK activity corresponds to a
stated maximum firing rate R_MAX:

    g_layer = R_MAX / peak(r_layer)          f_layer = g_layer * mean(r_layer)
            => f_layer = R_MAX * mean(r_layer) / peak(r_layer)

which is a pure SHAPE ratio: dt-independent by construction. Synaptic operations are then

    SOPs = sum over projections of  ( N_pre * f_pre ) * fanout

with no substep multiplier (a spike drives its targets once, however finely the field equation
is integrated). The estimate now depends on a disclosed neuroscience/hardware parameter (R_MAX)
rather than on an arbitrary simulation timestep; we report its sensitivity.
"""
import os, sys, json
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', 'neuro-symbolic-slam'))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, os.path.join(_HERE, '..'))
import figstyle
import src.snn_slam_system as S
from snn_place_cells import N_PLACE, N_CSNN, N_STDP, N_DEPTH
from snn_pose_cann import RING_N, CANN_SIZES

R_MAX_DEFAULT = 100.0      # Hz, peak firing rate of the rate-to-spike mapping (disclosed)
E_LO, E_MID, E_HI = 12.7e-12, 23.6e-12, 26.0e-12   # ODIN / Loihi / TrueNorth  J per SOP


WARMUP_STEPS = 100    # 2 s: let the online-learning place code settle before measuring
WINDOW_STEPS = 400    # 8 s operating window over which the activity statistics are taken


def collect(steps=WINDOW_STEPS, warmup=WARMUP_STEPS, seed=42):
    """Run the deployed network; record mean and peak activity per layer (shape statistics).

    Two corrections vs the first version of this profile:
      * The ToF layer is tapped at the ACTUAL signal that drives the place cells --
        `system.tof_coder(tof)`, the N_DEPTH Gaussian-RBF depth code fed to `forward_step`
        (snn_slam_system.phase_perception). The previous version read
        `place_state.trace_tof`, a state field that is only ever initialised to zeros and never
        written, so the entire ToF->place projection silently contributed 0 SOPs.
      * The statistics are taken over a disclosed operating WINDOW after a WARMUP, not over a
        50-step cold start. Place-cell activity is non-stationary (it sparsifies as the code is
        learned), so the SOP rate is time-varying; we report the window mean and its range.
    """
    env = S.LiveEnvironment(random.PRNGKey(seed), chunk_size=steps + warmup + 100, course_type='random')
    system = S.SNNSLAMSystem(random.PRNGKey(43), n_depth=S.N_DEPTH); system.reset(1)
    ev, kin, tof, p0, th0, _ = env.step()
    system.initialize_pose(jnp.array([p0]), jnp.array([th0]))
    acc = {k: {'mean': [], 'peak': []} for k in ('csnn', 'stdp', 'ring', 'grid', 'place', 'tof')}

    def note(key, arr):
        a = np.asarray(arr, dtype=np.float64).ravel()
        a = np.maximum(a, 0.0)
        acc[key]['mean'].append(a.mean()); acc[key]['peak'].append(a.max())

    for i in range(steps + warmup):
        ev, kin, tof, *_ = env.step()
        _, r_place, r_ring, *_ = system.forward_step(jnp.array([ev]), jnp.array([kin]), jnp.array([tof]),
                                                     autopilot_on=True)
        if i < warmup:
            continue
        note('csnn', system.vision_state.csnn_trace)
        note('stdp', system.vision_state.stdp_trace)
        note('ring', system.pose._r_ring)
        note('grid', np.concatenate([np.asarray(g).ravel() for g in system.pose._r_canns]))
        note('place', r_place)
        note('tof', system.tof_coder(jnp.array([tof])))    # the REAL ToF drive (see docstring)

    out = {}
    for k, v in acc.items():
        m = float(np.mean(v['mean']))
        # Normalize by a robust high percentile of the per-step peaks, NOT the global max: the
        # global max is an order statistic that keeps growing with window length (heavy-tailed
        # place activity), which made the "dt-independent" shape ratio window-length-dependent.
        pk = float(np.percentile(np.asarray(v['peak']), 99.0))
        out[k] = dict(mean=m, peak=pk, shape=(m / pk if pk > 0 else 0.0))
    return out


def sops_from(shape, r_max):
    """f = R_MAX * mean/peak (dt-free).  SOPs = sum_proj (N_pre * f_pre) * fanout. No substeps.

    Scope: the RATE-CODED state estimator (ring + grid attractors) and the place-cell integration
    layer. Counts the synapse arrays that actually run each step, verified against the deployed
    weights and driven by the CORRECTLY measured presynaptic activity shapes:
      * ring: symmetric DoG (W_ring) + asymmetric velocity (W_ring_asym) = 2 * RING_N fan-out.
      * grid: EACH of the 3 co-prime modules applies THREE dense N x N matrices per step -- the
        symmetric DoG recurrence W_cann and the two velocity-shift matrices W_cann_asym_x/_y
        (snn_pose_cann.py:466/467/472/473/498), driven by the grid activity. Fan-out = 3 * N.
        (The earlier version counted only the DoG matrix, a ~3x undercount.)
      * place: the 5 place-cell INPUT projections into the N_PLACE cells (CSNN->place, STDP->place,
        ToF->place, grid->place, and the place recurrence), each driven by its measured presynaptic
        OUTPUT rate. The vision OUTPUTS (CSNN/STDP barcodes) therefore DO enter this count via their
        ->place projections.
    NOT counted, and explicitly disclosed as a separate cost class: the spiking event-vision
    front-end's INTERNAL synapses -- the frozen convolutional CSNN edge extractor and the STDP
    feature-layer dense (512->256). Their SOP rate is set by the sparse DVS EVENT rate driving the
    time-surface, not by these rate-coded activity shapes, so counting them at a rate-coded proxy
    would be unfounded (an event-driven front-end is exactly where a spiking implementation is
    sparse). Also excluded: the sparse, zero-initialised online-learned depth x heading conditional
    ring tensors (W_*_to_ring). Reported as a rate-coded-attractor + place-layer estimate.
    """
    f = {k: r_max * v['shape'] for k, v in shape.items()}
    n_grid = sum(s * s for s in CANN_SIZES)
    N_grid_syn = 3 * sum((s * s) ** 2 for s in CANN_SIZES)   # 3 dense NxN matrices per module
    sop = {
        'ring':  RING_N * f['ring'] * (2 * RING_N),
        'grid':  f['grid'] * N_grid_syn,
        # Appearance-key projection W_vis_hash: a FROZEN (N_CSNN x N_PLACE) random matrix applied to
        # the frozen-CSNN feature vector every step (snn_place_cells.py:140 definition, :371 read),
        # followed by top-k. It is counted separately from 'place' because it is the ONLY place-module
        # tensor that is load-bearing for a reported result: every reported loop-closure candidate is
        # generated from its output, whereas the five 'place' projections below feed outputs the
        # analysis harness discards. Together with 'ring' it constitutes the minimal sufficient stack.
        'hash':  N_CSNN * f['csnn'] * N_PLACE,
        # T2.2 note (2026-07-28): the STDP layer is retained as part of the as-built architecture and
        # counted here, although the component audit (Results \S "component dependence") shows the
        # reported results do not depend on it in this environment.
        'place': (N_CSNN * f['csnn'] * N_PLACE + N_STDP * f['stdp'] * N_PLACE
                  + N_DEPTH * f['tof'] * N_PLACE + n_grid * f['grid'] * N_PLACE
                  + N_PLACE * f['place'] * N_PLACE),
    }
    return f, sop, sum(sop.values())


def plot_figure(sop, total, power_mW):
    """Per-layer synaptic-operation breakdown (the Fig 6 figure). Data-driven: title and the
    power band read the computed values, so the figure can never drift from energy_v2_results.json.

    Rendered as a Cleveland DOT plot on a log axis, not bars: bar LENGTH on a log axis encodes
    log(v)-log(x_min) (x_min arbitrary), which mis-states the shares the layer % labels give. Here
    the value is the dot POSITION; the thin leader lines are explicit guides, not the magnitude
    channel. A single-hue grey ramp (not the blue/green/orange figstyle reserves for cross-figure
    data roles) keeps it greyscale-safe and free of semantic-color collisions."""
    figstyle.apply()
    names = {'place': 'Place cells', 'grid': 'Grid modules', 'ring': 'Ring attractor'}
    keys = sorted(('ring', 'grid', 'place'), key=lambda k: sop[k])   # ascending -> place on top
    vals = [sop[k] for k in keys]
    shades = ['#9fb0bf', '#5c6b7a', '#26313d']                        # light->dark = small->large
    # one decimal on the share so the figure matches SI Table S8 (ring is 0.7%, not '1%')
    labels = [f"{names[k]}\n{100 * sop[k] / total:.1f}%" for k in keys]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.set_xscale('log')
    xmin = 10 ** np.floor(np.log10(min(vals)) - 0.2)
    for i, (v, c) in enumerate(zip(vals, shades)):
        ax.hlines(i, xmin, v, color=figstyle.C_GRID, lw=1.3, zorder=2)
        ax.plot(v, i, 'o', ms=11, color=c, zorder=4)
        # compact, estimate-appropriate precision (5.10M / 1.75M / 48k), not 7 significant figures
        ax.text(v * 1.5, i, figstyle.fmt_si(v, 2), va='center', fontsize=9, color=figstyle.C_INK)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlim(xmin, total * 3.5)
    ax.grid(True, axis='x'); ax.grid(False, axis='y')
    ax.set_xlabel('Synaptic operations per second')
    ax.set_title(f'Neuromorphic compute (neural layers only): {total / 1e6:.2f}$\\times$10$^6$ SOPs/s',
                 loc='left')
    ax.annotate(f"≈ {power_mW['lo']:.2f}–{power_mW['hi']:.2f} mW (ODIN–TrueNorth per-op energy)",
                xy=(0.975, 0.12), xycoords='axes fraction', ha='right', va='center', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#d3d8dd', lw=0.8))
    figstyle.save(fig, __file__, 'snn_slam_compute_efficiency')
    plt.close(fig)


def main():
    shape = collect()
    print("Per-layer activity SHAPE statistics (dt-independent):\n")
    print(f"{'layer':>7} | {'mean r':>10} {'peak r':>10} {'mean/peak':>10}")
    for k, v in shape.items():
        print(f"{k:>7} | {v['mean']:>10.4f} {v['peak']:>10.4f} {v['shape']:>10.4f}")

    f, sop, total = sops_from(shape, R_MAX_DEFAULT)
    print(f"\nRate-to-spike mapping: peak activity -> R_MAX = {R_MAX_DEFAULT:.0f} Hz\n")
    print(f"{'layer':>7} | {'firing rate (Hz)':>17} | {'SOPs/s':>14}")
    for k in ('ring', 'grid', 'place'):
        print(f"{k:>7} | {f[k]:>17.2f} | {sop[k]:>14,.0f}")
    print(f"{'TOTAL':>7} | {'':>17} | {total:>14,.0f}")
    print(f"\n  measured over a {WINDOW_STEPS}-step ({WINDOW_STEPS*0.02:.0f} s) operating window "
          f"after a {WARMUP_STEPS}-step warm-up")

    print(f"\nPower (neural layers only):")
    for lbl, e in (('ODIN 12.7 pJ', E_LO), ('Loihi 23.6 pJ', E_MID), ('TrueNorth 26 pJ', E_HI)):
        print(f"   {lbl:>18}: {total*e*1e3:6.3f} mW")

    print(f"\nSensitivity to the DISCLOSED parameter R_MAX (replaces dt-dependence):")
    print(f"{'R_MAX (Hz)':>11} | {'SOPs/s':>13} | {'power band (mW)':>18}")
    sens = {}
    for rm in (50.0, 100.0, 200.0):
        _, _, t = sops_from(shape, rm)
        sens[rm] = t
        print(f"{rm:>11.0f} | {t:>13.3e} | {t*E_LO*1e3:6.3f} - {t*E_HI*1e3:6.3f}")

    power_mW = {'lo': total*E_LO*1e3, 'mid': total*E_MID*1e3, 'hi': total*E_HI*1e3}
    minimal = {'ring': sop['ring'], 'hash': sop['hash'],
               'total_sops': sop['ring'] + sop['hash'],
               'power_mW': {k: (sop['ring'] + sop['hash']) * e * 1e3
                            for k, e in (('lo', 12.7e-12), ('mid', 23.6e-12), ('hi', 26.0e-12))}}
    json.dump({'shape': shape, 'R_MAX': R_MAX_DEFAULT, 'firing_rates': f,
               'minimal_sufficient': minimal,
               'sops': sop, 'total_sops': total, 'power_mW': power_mW,
               'window': {'warmup_steps': WARMUP_STEPS, 'window_steps': WINDOW_STEPS,
                          'note': 'place-cell activity is non-stationary (sparsifies with learning); '
                                  'the SOP rate is a window mean, not a fixed constant'},
               'sensitivity_R_MAX': {str(k): v for k, v in sens.items()}},
              open(os.path.join(_HERE, 'energy_v2_results.json'), 'w'), indent=2)
    print("\nwrote energy_v2_results.json")

    plot_figure(sop, total, power_mW)
    print("wrote snn_slam_compute_efficiency.pdf/.png")


if __name__ == '__main__':
    main()
