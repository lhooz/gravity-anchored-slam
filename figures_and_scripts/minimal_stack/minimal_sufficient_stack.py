#!/usr/bin/env python3
"""Minimal-sufficient-stack ablation (reviewer item 55).

Four arms, n=12 seeds, on `circuit` and `circuit_alias`, testing which NEURAL subsystems are
load-bearing for the reported SLAM numbers:

  1. full          shipped configuration (control)
  2. noplace       place layer lobotomised: forward_mapping is a no-op, every plastic tensor stays
                   at its zero init, and every learned output is zeroed. ONLY the frozen random
                   projection W_vis_hash + top-k WTA survives (that is what generates candidates).
  3. noplace_nogrid  additionally bypass the 579-unit grid CANN decode with a direct read of the
                   float integrator `pose.imu_integrated_xy`.
  4. noplace_nogrid_noring  additionally K_GRAVITY=0 on the closed-loop arm (the arm that must break).

Reports ate_cl / ate_ol / ate_imu, fired closures, fired precision and keyframe count per arm.
Run sequentially -- never two JAX sims at once.
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
SLAM = os.path.normpath(os.path.join(HERE, '..', '..', 'neuro-symbolic-slam'))
sys.path.insert(0, SLAM); sys.path.insert(0, os.path.join(SLAM, 'src'))
sys.path.insert(0, os.path.join(SLAM, 'scripts'))
import jax.numpy as jnp
import src.snn_place_cells as PCL
import src.snn_slam_system as S


def lobotomise_place():
    """forward_mapping -> no-op; all learned readouts zeroed. W_vis_hash/top-k untouched."""
    def dead_mapping(self, state, vis_csnn, vis_stdp, tof_features, pose_bump,
                     ring_bump=None, heading=None, angular_vel=None, learn=True, confidence=None):
        B = vis_csnn.shape[0]
        return state, (jnp.zeros((B, self.n_place)), jnp.zeros((B, self.n_ring if hasattr(self,'n_ring') else 64)))
    PCL.PlaceCellNetwork.forward_mapping = dead_mapping

    orig_gates = PCL.PlaceCellNetwork.compute_confidence_with_gates
    def gates_hash_only(self, state, vis_csnn, vis_stdp, tof_features, pose_bump, heading, ring_bump):
        st, is_conf, peak, dbg = orig_gates(self, state, vis_csnn, vis_stdp,
                                            tof_features, pose_bump, heading, ring_bump)
        B = vis_csnn.shape[0]
        # keep ONLY the frozen-projection descriptor; zero every learned-weight-dependent output
        keep = dbg['Visual_Barcode']
        for k in list(dbg):
            if k != 'Visual_Barcode':
                try: dbg[k] = jnp.zeros_like(jnp.asarray(dbg[k]))
                except Exception: pass
        dbg['Visual_Barcode'] = keep
        return st, jnp.zeros((B,), dtype=bool), jnp.zeros((B,), dtype=jnp.int32), dbg
    PCL.PlaceCellNetwork.compute_confidence_with_gates = gates_hash_only


def bypass_grid():
    """Bypass the 579-unit grid CANN: read the float Euler integrator the grid is PINNED to.

    NOTE: decode_grid_to_xy(bump, prior) unwraps the bump phase RELATIVE TO `prior`, so a naive
    `return prior` freezes the estimate rather than bypassing the grid (it yields a degenerate
    fixed point, not a measurement). The correct bypass returns pose.imu_integrated_xy -- the
    conventionally integrated position that K_IMU_POS=200 pins the bump to every step.
    Both call sites (phase_odometry, forward_step) are wrapped so the substitution is complete.
    """
    for meth in ('phase_odometry', 'forward_step'):
        orig = getattr(S.SNNSLAMSystem, meth)
        def make(orig=orig):
            def wrapped(self, *a, **kw):
                real = S.decode_grid_to_xy
                S.decode_grid_to_xy = lambda bump, prior: self.pose.imu_integrated_xy
                try:
                    return orig(self, *a, **kw)
                finally:
                    S.decode_grid_to_xy = real
            return wrapped
        setattr(S.SNNSLAMSystem, meth, make())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True,
                    choices=['full', 'noplace', 'noplace_nogrid', 'noplace_nogrid_noring'])
    ap.add_argument('--course', default='circuit')
    ap.add_argument('--seeds', type=int, default=12)
    ap.add_argument('--steps', type=int, default=600)
    a = ap.parse_args()

    if a.arm != 'full':
        lobotomise_place()
    if a.arm in ('noplace_nogrid', 'noplace_nogrid_noring'):
        bypass_grid()
    if a.arm == 'noplace_nogrid_noring':
        os.environ['LC_ANCHOR_OFF'] = '1'

    import slam_variance as SV
    rows = []
    for i in range(a.seeds):
        seed = 42 + 111 * i
        r = SV.run_trial(seed=seed, n_steps=a.steps, course_type=a.course, enable_lc=True)
        KEEP = ('ate_imu', 'ate_ol', 'ate_cl', 'final_ol', 'final_cl',
                'n_loop_closures', 'n_lc_true_pos', 'lc_precision', 'n_nodes')
        row = {'seed': seed, **{k: r[k] for k in KEEP}}
        rows.append(row)
        print(f"  [{a.arm}/{a.course}] seed {seed}: "
              f"ate_cl={row['ate_cl']:.6f} ate_ol={row['ate_ol']:.6f} "
              f"lc={row['n_loop_closures']} prec={row['lc_precision']} kf={row['n_nodes']}",
              flush=True)

    out = os.path.join(HERE, f'{a.arm}_{a.course}.json')
    json.dump({'arm': a.arm, 'course': a.course, 'steps': a.steps, 'rows': rows},
              open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
