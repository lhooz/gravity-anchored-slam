"""
Regenerate every paper figure (Fig 1-6 + SI Fig S1-S3) from the committed scripts.
Run with the project venv:  python figures_and_scripts/make_all_figures.py

Pipeline (data-generator -> plotter). Every figure that appears in the manuscript is listed;
each is reproducible from committed code. Superseded/exploratory scripts live under archive/.

  Fig 1  architecture     figure1_architecture/make_architecture.py            (hand-authored SVG -> PDF; needs cairosvg)
  Fig 2  attitude anchor  A: scripts/clean_anchor_ablation.py -> anchor_ablation_results.json
                          B: figure8_classical_baseline/rerun_ring_diagnostics.py -> ring_diagnostics_corrected.json
                          C: figure6_sensory_deprivation/simulate_sensory_deprivation.py -> sensory_deprivation_data.npz
                          plot: figure_attitude_anchor/generate_attitude_anchor.py
  Fig 3  classical base.  figure8_classical_baseline/rerun_ring_diagnostics.py -> ring_diagnostics_corrected.json
                          plot: figure8_classical_baseline/plot_classical_baseline.py
  Fig 4  loop closure     scripts/slam_variance.py --compare-lc --course circuit --seeds 12 --steps 2000
                          plot: figure7_loop_closure/plot_loop_closure.py
  Fig 5  topomap          same compare-lc run (graph dump) ; plot: figure7_loop_closure/plot_topomap.py
  Fig 6  compute          figure4_compute_efficiency/profile_snn_energy_v2.py (computes + plots)
  Fig S1 Monte Carlo      scripts/slam_variance.py --seeds 20 --steps 600 ; plot: figure5_monte_carlo/plot_monte_carlo.py
  Fig S2 bump drift       ring_diagnostics_corrected.json (key bump_drift) ; plot: figure3_bump_drift/plot_bump_drift_corrected.py
  Fig S3 aliasing setup   figure9_aliasing_stress/plot_aliasing_setup.py (live sparse_forest generators)

Notes:
  * The sensors do not share the estimator's clock: the MEMS IMU samples at its own ~1 kHz hardware
    rate and the estimator consumes the mean rate over each step (snn_slam_system.imu_rate_average,
    imported by the figure harness -- one front-end, so deployed and diagnostics cannot diverge).
    A wingbeat tone is therefore attenuated by |sinc(f*DT)| (8.9x at 115 Hz), not aliased.
  * The gravity anchor is a complementary filter (ALPHA_FUSE, deployed) injected into the ring CANN.
  * Position/ATE uses the modeled event+ToF VO velocity (SLAM_VO_NOISE, default on), identical
    across all baselines. See PROVENANCE.md.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SLAM = os.path.normpath(os.path.join(HERE, '..', 'neuro-symbolic-slam'))


def run(cmd, cwd=None, env=None):
    print(f"\n=== {' '.join(c if not os.path.sep in c else os.path.basename(c) for c in cmd)} ===", flush=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    subprocess.run(cmd, cwd=cwd, check=True, env=e)


def script(rel):
    return os.path.join(HERE, rel)


def main():
    F5 = os.path.join(HERE, 'figure5_monte_carlo')
    # ---- data generators (heavy; JAX) ------------------------------------------------
    run([PY, 'scripts/clean_anchor_ablation.py', '--seeds', '12'], cwd=SLAM)   # writes anchor_ablation_results.json
    # LC_AUDIT=1 is REQUIRED: without it the run writes lc_pre_gate_audit: null and destroys the
    # pre-gate appearance precision/recall (72.1% / 66.8%) that the manuscript quotes.
    run([PY, 'scripts/slam_variance.py', '--compare-lc', '--course', 'circuit', '--seeds', '12', '--steps', '2000',
         '--output', os.path.join(HERE, 'figure7_loop_closure', 'loop_closure_results.json')],
        cwd=SLAM, env={'LC_AUDIT': '1'})
    run([PY, 'scripts/slam_variance.py', '--seeds', '20', '--steps', '600',
         '--output', os.path.join(F5, 'variance_results.json')], cwd=SLAM)
    # perceptual-aliasing stress test -> figure9_aliasing_stress/aliased_circuit.json (SI table)
    run([PY, 'scripts/slam_variance.py', '--compare-lc', '--course', 'circuit_alias', '--seeds', '12', '--steps', '2000',
         '--output', os.path.join(HERE, 'figure9_aliasing_stress', 'aliased_circuit.json')],
        cwd=SLAM, env={'LC_AUDIT': '1', 'SLAM_ALIAS_KFOLD': '2'})
    # anchor-OFF arm of the aliasing stress test -> aliased_circuit_anchoroff.json (SI Table S7).
    # This is the CONTROL for the causal claim: pre-gate precision must be ~unchanged by the anchor, so
    # BOTH arms must be regenerated under the same configuration. Omitting it silently mixes configs and
    # produces a spurious "anchor-invariance" failure (observed 2026-07-28).
    run([PY, 'scripts/slam_variance.py', '--compare-lc', '--course', 'circuit_alias', '--seeds', '12', '--steps', '2000',
         '--output', os.path.join(HERE, 'figure9_aliasing_stress', 'aliased_circuit_anchoroff.json')],
        cwd=SLAM, env={'LC_AUDIT': '1', 'SLAM_ALIAS_KFOLD': '2', 'LC_ANCHOR_OFF': '1'})
    # minimal-sufficient-stack ablation -> minimal_stack/*.json (Results \S "Which Components Are
    # Load-Bearing?" and SI Table S9). Four arms x two courses, n=12.
    run(['bash', script('minimal_stack/run_all.sh')])
    run([PY, script('figure8_classical_baseline/rerun_ring_diagnostics.py')])
    run([PY, script('figure6_sensory_deprivation/simulate_sensory_deprivation.py')])

    # ---- figures ---------------------------------------------------------------------
    run([PY, script('figure1_architecture/make_architecture.py')])
    run([PY, script('figure_attitude_anchor/generate_attitude_anchor.py')])
    run([PY, script('figure8_classical_baseline/plot_classical_baseline.py')])
    run([PY, script('figure7_loop_closure/plot_loop_closure.py')])
    run([PY, script('figure7_loop_closure/plot_topomap.py')])
    run([PY, script('figure4_compute_efficiency/profile_snn_energy_v2.py')])
    run([PY, script('figure5_monte_carlo/plot_monte_carlo.py')])
    run([PY, script('figure5_monte_carlo/equivalence_test.py'), '--json', os.path.join(F5, 'variance_results.json')])
    run([PY, script('figure3_bump_drift/plot_bump_drift_corrected.py')])
    run([PY, script('figure9_aliasing_stress/plot_aliasing_setup.py')])
    print("\nAll paper figures regenerated (Fig 1-6, SI S1-S3).")


if __name__ == '__main__':
    main()
