#!/usr/bin/env python3
"""Clean single-variable gravity-anchor ablation (reproduces SI Appendix Table:
"Clean Gravity-Anchor Ablation --- Attitude vs. Position").

Both arms use an identically seeded spiking network (PRNG seed 43), the same full forward
pass (forward_step), and the same environment stream, toggling ONLY the gravity-injection
gain K_GRAVITY (deployed default -> anchor on, 0 -> anchor off). It reports the aligned ATE, the mean
absolute attitude (heading) error, and the ATE decomposed into along-track (parallel to
travel) and cross-track (perpendicular) components.

CORRECTION (2026-07-23). An earlier version of this docstring asserted that this script's
arms differ MATERIALLY from the production open-/closed-loop comparison in slam_variance.py
-- specifically in random seed and in whether the place-cell inference path runs. That claim
was wrong, and is retracted: across all 12 circuit seeds the outputs here are BIT-IDENTICAL
to loop_closure_results.json. The nominal differences are therefore immaterial to every
reported number -- which is itself a direct demonstration that the place-cell inference path
does not affect any result reported in the paper.

Usage:  python neuro-symbolic-slam/scripts/clean_anchor_ablation.py [--seeds N]
"""
import os, sys, argparse, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # neuro-symbolic-slam/
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import jax, jax.numpy as jnp
from jax import random
import numpy as np
from scipy import stats
import src.snn_slam_system as S

# Committed JSON artifact so the attitude figure (Fig. 2A) and SI Table are regenerated
# from data, not hard-coded literals (repo-relative).
_OUT = os.path.normpath(os.path.join(ROOT, '..', 'figures_and_scripts',
                                     'figure_attitude_anchor', 'anchor_ablation_results.json'))


def mean_ci(vals):
    """Mean and 95% CI half-width (Student-t, small n)."""
    a = np.asarray(vals, float); m = float(a.mean())
    if a.size < 2:
        return m, 0.0
    sem = float(a.std(ddof=1) / np.sqrt(a.size))
    return m, float(stats.t.ppf(0.975, a.size - 1) * sem)


def align(a, gt):
    R, t = S.get_optimal_alignment_2d(a, gt); return (R @ a.T).T + t


def decomp(a, gt):
    """ATE plus its along-track (parallel to GT travel) and cross-track (perpendicular) RMS, cm."""
    aa = align(a, gt); e = aa - gt
    d = np.gradient(gt, axis=0); nrm = np.hypot(d[:, 0], d[:, 1]) + 1e-9
    tx, ty = d[:, 0]/nrm, d[:, 1]/nrm
    along = e[:, 0]*tx + e[:, 1]*ty
    cross = e[:, 0]*(-ty) + e[:, 1]*tx
    return (np.mean(np.hypot(e[:, 0], e[:, 1]))*100,
            np.sqrt((along**2).mean())*100, np.sqrt((cross**2).mean())*100)


def herr(th, gth):
    e = np.abs((np.asarray(th) - np.asarray(gth) + np.pi) % (2*np.pi) - np.pi)
    return float(np.rad2deg(e).mean())


def trial(seed, n_steps, course):
    env = S.LiveEnvironment(random.PRNGKey(seed), chunk_size=n_steps+100, course_type=course)
    s_on  = S.SNNSLAMSystem(random.PRNGKey(43), n_depth=S.N_DEPTH)
    s_off = S.SNNSLAMSystem(random.PRNGKey(43), n_depth=S.N_DEPTH)
    s_on.reset(1); s_off.reset(1); s_off.pose.K_GRAVITY = 0.0        # <-- the ONLY difference
    _, _, _, pos0, th0, _ = env.step()
    s_on.initialize_pose(jnp.array([pos0]), jnp.array([th0]))
    s_off.initialize_pose(jnp.array([pos0]), jnp.array([th0]))
    gt, on, off, gth, on_th, off_th = [], [], [], [], [], []
    for st in range(n_steps):
        ev, kin, tof, gp, gth_t, _ = env.step()
        ej, kj, tj = jnp.array([ev]), jnp.array([kin]), jnp.array([tof])
        p_on = s_on.forward_step(ej, kj, tj)[0]; p_off = s_off.forward_step(ej, kj, tj)[0]
        gt.append(gp); gth.append(gth_t)
        on.append([float(p_on[0, 0]), float(p_on[0, 1])]); on_th.append(float(p_on[0, 2]))
        off.append([float(p_off[0, 0]), float(p_off[0, 1])]); off_th.append(float(p_off[0, 2]))
    gt = np.array(gt)
    aon, aoff = decomp(np.array(on), gt), decomp(np.array(off), gt)
    return dict(ate_on=aon[0], along_on=aon[1], cross_on=aon[2], h_on=herr(on_th, gth),
               ate_off=aoff[0], along_off=aoff[1], cross_off=aoff[2], h_off=herr(off_th, gth))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=6); a = ap.parse_args()
    seeds = [42 + i*111 for i in range(a.seeds)]
    out = {'n_seeds': a.seeds, 'seeds': seeds, 'ci': 'Student-t 95%'}
    for course, steps in [('random', 600), ('circuit', 2000)]:
        print(f"\n### CLEAN ANCHOR ABLATION -- {course}, {steps} steps ({a.seeds} seeds) ###")
        R = [trial(s, steps, course) for s in seeds]
        keys = list(R[0].keys())
        per_seed = {k: [float(r[k]) for r in R] for k in keys}
        mean = {k: mean_ci(per_seed[k])[0] for k in keys}
        ci95 = {k: mean_ci(per_seed[k])[1] for k in keys}
        out[course] = {'steps': steps, 'per_seed': per_seed, 'mean': mean, 'ci95': ci95}
        print(f"  anchor ON : heading {mean['h_on']:.1f}+/-{ci95['h_on']:.1f} deg | "
              f"along {mean['along_on']:.2f}  cross {mean['cross_on']:.2f}  ATE {mean['ate_on']:.2f} cm")
        print(f"  anchor OFF: heading {mean['h_off']:.1f}+/-{ci95['h_off']:.1f} deg | "
              f"along {mean['along_off']:.2f}  cross {mean['cross_off']:.2f}  ATE {mean['ate_off']:.2f} cm")
    with open(_OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n--> wrote {_OUT}")


if __name__ == '__main__':
    main()
