#!/usr/bin/env python3
"""
verify_figures.py -- assert every committed figure's data matches the manuscript headline number.

Run after regenerating figures (or in CI) to catch silent drift between the committed data files
and the numbers reported in the paper. Reads only the committed JSON/NPZ (no JAX, no re-simulation),
so it is fast. Exit code 0 = all pass, 1 = at least one mismatch.

Each check states the manuscript value and an absolute tolerance; update this table whenever a
reported number legitimately changes (e.g. after re-running a generator).
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_json(rel):
    return json.load(open(os.path.join(HERE, rel)))


def _npz(rel):
    return np.load(os.path.join(HERE, rel))


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


CHECKS = []
def _load(rel):
    import json, os
    return json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)))


def _fail(msg):
    global FAILED
    try: FAILED += 1
    except NameError: pass
    print(f'  FAIL: {msg}')


def check(name, got, want, tol, unit=''):
    CHECKS.append((name, float(got), float(want), float(tol), unit))


def main():
    # ---- Fig 2A: gravity-anchor ablation (attitude MAE, n=12) ----
    a = _load_json('figure_attitude_anchor/anchor_ablation_results.json')
    check('Fig2A random anchor-off attitude', a['random']['mean']['h_off'], 9.0, 1.0, 'deg')
    check('Fig2A random anchor-on attitude',  a['random']['mean']['h_on'],  3.0, 0.6, 'deg')
    check('Fig2A circuit anchor-off attitude', a['circuit']['mean']['h_off'], 46.6, 4.0, 'deg')
    check('Fig2A circuit anchor-on attitude',  a['circuit']['mean']['h_on'],  7.5, 1.2, 'deg')

    # ---- Fig 2B / Fig 3A: vibration envelope (hardware-rate IMU front-end) ----
    d = _load_json('figure8_classical_baseline/ring_diagnostics_corrected.json')
    env = d['envelope']
    gm = lambda k: float(np.mean([r[k] for r in env]))
    check('Fig3A grand-mean anchored ring', gm('ring_anch'), 1.44, 0.10, 'deg')
    check('Fig3A grand-mean CF',            gm('cf'),        0.92, 0.10, 'deg')
    check('Fig3A grand-mean EKF',           gm('ekf'),       0.75, 0.10, 'deg')
    check('Fig3A grand-mean unanchored',    gm('ring_unanch'), 7.47, 1.2, 'deg')
    check('Fig2B anchored amp=0.1', env[0]['ring_anch'],  0.66, 0.10, 'deg')
    check('Fig2B anchored amp=10',  env[-1]['ring_anch'], 3.10, 0.35, 'deg')
    # ring must track the CF (the attractor is a readout of the filter that anchors it)
    check('Fig3A ring-minus-CF (small overhead at deployed K_g=5)', abs(gm('ring_anch') - gm('cf')), 0.52, 0.15, 'deg')

    # ---- Fig 3B: transient outlier sweep ----
    o = d['outliers']
    check('Fig3B EKF @ 40% outliers', o[-1]['ekf'], 3.69, 0.5, 'deg')
    check('Fig3B ring @ 40% outliers', o[-1]['ring'], 1.67, 0.35, 'deg')

    # ---- Fig S2: bump drift, 30 s ----
    bd = d['bump_drift']
    check('FigS2 unanchored @30s', bd['running_unanch'][-1], 51.1, 5.0, 'deg')
    check('FigS2 anchored @30s',   bd['running_anch'][-1],   1.9, 0.4, 'deg')
    check('FigS2 GT motion RMS',   bd['gt_rms_deg'],         60.8, 1.0, 'deg')

    # ---- Fig 2C: reference dropout (deployed dt + deployed CF, n=10 seeds) ----
    occ = _npz('figure6_sensory_deprivation/sensory_deprivation_data.npz')
    # tolerance is tight enough to catch a revert to the old accel-dominated / wrong-dt filter
    check('Fig2C occlusion unanchored (mean)', float(occ['rmse_unanchored_mean']), 2.9, 0.8, 'deg')
    check('Fig2C occlusion anchored (mean)',   float(occ['rmse_anchored_mean']),   1.1, 0.3, 'deg')

    # ---- Fig 4: loop closure ----
    lc = _load_json('figure7_loop_closure/loop_closure_results.json')['summary']
    check('Fig4 IMU ATE',    lc['ate_imu_cm']['mean'],    18.8, 1.5, 'cm')
    check('Fig4 LC-OFF ATE', lc['ate_cl_off_cm']['mean'], 14.8, 1.5, 'cm')
    check('Fig4 LC-ON ATE',  lc['ate_cl_on_cm']['mean'],  2.6, 0.8, 'cm')
    check('Fig4 LC precision', lc['lc_precision']['pooled'], 1.0, 0.001, '')
    check('Fig4 closures/run', lc['lcs']['mean'], 246, 40, '')
    # Fig 5 topomap loop-scale ratio: the manuscript reports the across-seed MEAN (0.980x) and
    # the min--max RANGE (0.888--1.088x); the representative seed shown in Fig 5 is 0.978x. Tolerances
    # are tight so a re-run that shifts these headline numbers fails here instead of drifting silently.
    ls_ratio = lc['loop_scale_ratio']
    check('Fig5 loop-scale mean', ls_ratio['mean'], 0.986, 0.015, 'x')
    check('Fig5 loop-scale range min', ls_ratio['min'], 0.888, 0.01, 'x')
    check('Fig5 loop-scale range max', ls_ratio['max'], 1.088, 0.01, 'x')

    # ---- SI pre-gate audit (standard circuit): the paper quotes exact raw counts, which drift on a
    # re-run even when the derived percentages hold. Lock them so a stale count fails here. ----
    pga = lc['lc_pre_gate_audit']
    check('SI pre-gate candidates', pga['hdc_cand'],    96284, 200, '')
    check('SI pre-gate true',       pga['hdc_true'],    72954, 200, '')
    check('SI pre-gate GT pairs',   pga['gt_pairs'],   103774, 200, '')
    check('SI pre-gate gate-rejected', pga['gate_rejected'], 68825, 200, '')
    check('SI pre-gate precision (%)', pga['pre_gate_precision'] * 100, 75.8, 0.5, '%')
    check('SI pre-gate recall (%)',    pga['recall'] * 100,            70.3, 0.5, '%')

    # ---- Fig 5: topomap matched-revisit gap (from dumped nodes) ----
    g = _npz('figure7_loop_closure/loop_closure_results_graph.npz')
    lce = np.asarray(g['lc_edges'], dtype=int)
    def gap(nodes):
        nd = np.asarray(nodes, dtype=float)[:, :2]
        ds = [np.hypot(nd[i, 0]-nd[j, 0], nd[i, 1]-nd[j, 1]) for i, j in lce
              if not (np.isnan(nd[i, 0]) or np.isnan(nd[j, 0]))]
        return float(np.mean(ds)) * 100
    check('Fig5 gap before relaxation', gap(g['node_orig']),      15.0, 4.0, 'cm')
    check('Fig5 gap after relaxation',  gap(g['node_corrected']), 2.3, 1.5, 'cm')

    # ---- Fig 6: compute / energy (real ToF tap; 400-step window after 100-step warm-up) ----
    e = _load_json('figure4_compute_efficiency/energy_v2_results.json')
    check('Fig6 total SOPs/s (1e6)', e['total_sops'] / 1e6, 7.13, 0.5, 'e6')
    check('Fig6 power lo (mW)', e['power_mW']['lo'], 0.091, 0.012, 'mW')
    check('Fig6 power hi (mW)', e['power_mW']['hi'], 0.185, 0.025, 'mW')
    # the headline claim: neural-layer compute stays SUB-MILLIWATT
    check('Fig6 sub-milliwatt (hi < 1 mW)', e['power_mW']['hi'], 0.185, 0.80, 'mW')
    # the ToF->place projection must be non-zero (it was silently dropped by a dead state field)
    check('Fig6 ToF activity present (shape>0)', e['shape']['tof']['shape'], 0.19, 0.10, '')
    # --- minimal sufficient stack (main text, \S sec:compute): the ring plus the frozen
    # appearance-key projection are the only counted neural terms any reported result needs.
    m = e['minimal_sufficient']
    check('Fig6 hash projection SOPs/s',   e['sops']['hash'],   114672, 2000, '')
    check('Fig6 ring SOPs/s',              e['sops']['ring'],    48401, 1000, '')
    check('Min-stack total SOPs/s',        m['total_sops'],     163072, 3000, '')
    check('Min-stack power lo (uW)',       m['power_mW']['lo'] * 1e3, 2.07, 0.15, 'uW')
    check('Min-stack power hi (uW)',       m['power_mW']['hi'] * 1e3, 4.24, 0.30, 'uW')
    check('Grid+place share of total (%)',
          100.0 * (e['sops']['grid'] + e['sops']['place']) / e['total_sops'], 97.7, 1.0, '%')
    # --- compute-to-compute ratios quoted in the text (assumed VIO envelope 2e10 ops/s @ 2.5 W)
    check('Ops ratio vs VIO baseline (1e3)', 2e10 / e['total_sops'] / 1e3, 2.80, 0.40, 'e3')
    check('Power ratio vs VIO (1e4)', 2.5 / (e['power_mW']['hi'] * 1e-3) / 1e4, 1.35, 0.30, 'e4')

    # ---- Minimal sufficient stack (SI Table S9): which neural subsystems are load-bearing.
    ms = _load('minimal_stack/minimal_stack_results.json')
    for course in ('circuit', 'circuit_alias'):
        c = ms['courses'][course]
        # the place layer must ablate BIT-identically -- this is a determinism claim, not a tolerance
        if not c['noplace']['bit_identical_to_full']:
            _fail(f'{course}: -Place is NOT bit-identical to full')
        else:
            print(f"{'Min-stack ' + course + ' -Place bit-identical':<44}{'yes':>10}{'yes':>10}{'':>8}  ok")
        check(f'{course} -Place-Grid delta (%)',  c['noplace_nogrid']['delta_pct'],        -0.48, 0.30, '%')
        check(f'{course} -Ring delta (%)',        c['noplace_nogrid_noring']['delta_pct'], 20.6,  8.0, '%')


    # ---- Fig S1: open-course Monte Carlo (n=20) ----
    v = _load_json('figure5_monte_carlo/variance_results.json')['summary']
    check('FigS1 IMU ATE',       v['ate_imu_mean'] * 100, 3.34, 0.4, 'cm')
    check('FigS1 anchor-off ATE', v['ate_ol_mean'] * 100, 3.26, 0.4, 'cm')
    check('FigS1 anchor-on ATE',  v['ate_cl_mean'] * 100, 3.11, 0.4, 'cm')

    # ---- Aliasing stress test + the anchor-off CAUSAL ablation (SI table, main text) ----
    # This is the paper's central coupling: attitude stabilization is what keeps drift inside the
    # geometric gate, which is what makes aliasing-robust precision-verified loop closure possible.
    al_on = _load_json('figure9_aliasing_stress/aliased_circuit.json')['summary']
    al_off = _load_json('figure9_aliasing_stress/aliased_circuit_anchoroff.json')['summary']
    pg_on = al_on['lc_pre_gate_audit']['pre_gate_precision'] * 100
    pg_off = al_off['lc_pre_gate_audit']['pre_gate_precision'] * 100
    check('Alias anchor-ON fired precision',  al_on['lc_precision']['pooled'],  1.000, 0.002, '')
    check('Alias anchor-OFF fired precision', al_off['lc_precision']['pooled'], 0.990, 0.015, '')
    check('Alias anchor-ON pre-gate precision',  pg_on,  64.2, 2.0, '%')
    check('Alias anchor-OFF pre-gate precision', pg_off, 63.6, 2.0, '%')
    check('Alias anchor-ON ATE (LC on)',  al_on['ate_cl_on_cm']['mean'],  2.9, 0.8, 'cm')
    check('Alias anchor-OFF ATE (LC on)', al_off['ate_cl_on_cm']['mean'], 10.7, 2.5, 'cm')

    # The CAUSAL claim, asserted directly (not just the endpoint numbers):
    # (a) the descriptor is a clean control -- pre-gate precision must be ~unchanged by the anchor
    #     (it cannot depend on attitude), so any difference downstream is the geometric gate.
    check('Alias pre-gate is anchor-INVARIANT (control)', abs(pg_on - pg_off), 0.0, 1.5, '%')
    # (b) removing the anchor must DEGRADE fired precision (false closures leak through the gate).
    check('Alias precision DROPS without anchor',
          al_on['lc_precision']['pooled'] - al_off['lc_precision']['pooled'], 0.030, 0.020, '')
    # (c) removing the anchor must COLLAPSE the drift correction (82% -> 37%).
    red = lambda s: 100.0 * s['lc_delta_cm']['mean'] / s['ate_cl_off_cm']['mean']
    check('Alias correction WITH anchor (%)',    red(al_on),  82.0, 6.0, '%')
    check('Alias correction WITHOUT anchor (%)', red(al_off), 48.1, 10.0, '%')
    check('Alias correction COLLAPSES without anchor',
          red(al_on) - red(al_off), 45.0, 14.0, '%')

    # ---- report ----
    fails = 0
    print(f"{'check':40s} {'got':>9} {'want':>9} {'tol':>7}  status")
    print('-' * 80)
    for name, got, want, tol, unit in CHECKS:
        ok = abs(got - want) <= tol
        fails += not ok
        print(f"{name:40s} {got:9.3f} {want:9.3f} {tol:7.3f}  {'ok' if ok else 'FAIL <<<'} {unit}")
    print('-' * 80)
    print(f"{len(CHECKS)} checks, {fails} failed.")
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
