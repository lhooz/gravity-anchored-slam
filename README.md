# Gravity-Anchored Neuromorphic Navigation: Sub-Milliwatt Attitude Stabilization and Loop Closure for Insect-Scale Flyers

Code and figure-reproduction scripts for the paper:

> **Gravity-Anchored Neuromorphic Navigation: Sub-Milliwatt Attitude Stabilization and Loop Closure for Insect-Scale Flyers.** Hao Li, Yanlai Zhang, Runqi Chai, and Jianghao Wu. Manuscript under review, 2026.

## Summary

Biological flyers maintain precise spatial orientation despite severe high-frequency wingbeat
vibration, using the energy-efficient ring-attractor circuits of the insect central complex. We
present a **neuromorphic SLAM system with a neuro-symbolic loop-closure back-end** for small,
low-mass micro air vehicles (MAVs). Operating in a vertical (sagittal) plane—where gravity
(invariant under yaw) supplies an absolute *tilt* reference—we track body-pitch attitude with a
1D rate-coded continuous attractor network (CANN) and position with a 2D multi-module grid-cell
CANN, fused with a modeled event+ToF visual-odometry velocity (realistic error model; a 3-beam
time-of-flight rangefinder also drives the place code) and a complementary gravity filter injected into the ring attractor.

The gravity anchor's demonstrated benefit is **attitude stabilization**—sharply reducing
body-pitch error even with vision present—while translation-limited position (ATE) is
essentially unchanged on open courses. On revisit-rich courses, appearance-keyed topological
loop closure with tight geometric verification (**precision 1.000, no false closures**) removes
accumulated *real global drift*, reducing closed-loop error from **14.8 to 2.6 cm (~83 %)**.
Profiling the running network yields **~7.1×10⁶ synaptic operations/s** and an estimated
**~0.09–0.19 mW** (sub-milliwatt) on-chip budget (an estimate for the neural layers only, not a hardware
measurement).

The system is a **hybrid — neuromorphic and neuro-symbolic**: event-driven perception (CSNN/STDP)
and the visual-odometry feature front-end are **spiking**; the attitude, grid, and place state
estimators are **rate-coded** continuous attractors; loop closure (sparse-binary appearance-hash matching →
geometric pose-gate → pose-graph relaxation) is **symbolic**.

## Repository layout

```
neuro-symbolic-slam/src/      core neuromorphic SLAM system (perception, CANN, place/grid cells, VO, relaxation)
neuro-symbolic-slam/scripts/  experiment drivers (slam_variance.py, clean_anchor_ablation.py produce the numbers)
figures_and_scripts/          one folder per figure; each script regenerates a figure (PDF/PNG) + its data
figures_and_scripts/figstyle.py   shared house style (Okabe-Ito palette, panel labels, PDF+PNG save)
requirements.txt              pinned environment (Python 3.14)
```

## Environment

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # CPU JAX; see jax.dev for GPU/TPU wheels
```

## Reproducing the figures

Run from the repository root with the environment active. `python figures_and_scripts/make_all_figures.py`
regenerates every figure end-to-end (data generators + plotters). Each script writes a vector
`*.pdf` (the canonical paper figure) plus a `*.png` preview to its `figureN/` dir (and to a
sibling `manuscript/figures/` directory when one exists, as in the full paper tree). Data
generators run **sequentially** on CPU JAX; the full end-to-end regeneration takes roughly an
hour or two on a modern laptop, dominated by the n=12, 2000-step loop-closure sweeps.

| Paper figure | Output | Command |
|---|---|---|
| Fig 1 — architecture *(schematic)* | `snn_slam_architecture` | `python figures_and_scripts/figure1_architecture/make_architecture.py` |
| Fig 2 — gravity attitude anchor (ablation + vibration + occlusion) | `snn_slam_attitude_anchor` | `python figures_and_scripts/figure_attitude_anchor/generate_attitude_anchor.py` |
| Fig 3 — classical baselines (ring vs CF vs EKF) | `snn_slam_classical_baseline` | `python figures_and_scripts/figure8_classical_baseline/plot_classical_baseline.py` |
| Fig 4 — loop closure (ATE bars + error-vs-time) | `snn_slam_loop_closure` | `python figures_and_scripts/figure7_loop_closure/plot_loop_closure.py` |
| Fig 5 — topological map (before/after relaxation) | `snn_slam_topomap` | `python figures_and_scripts/figure7_loop_closure/plot_topomap.py` |
| Fig 6 — compute / energy | `snn_slam_compute_efficiency` | `python figures_and_scripts/figure4_compute_efficiency/profile_snn_energy_v2.py` |
| SI Fig S1 — Monte Carlo drift | `snn_slam_monte_carlo_drift` | `python figures_and_scripts/figure5_monte_carlo/plot_monte_carlo.py` |
| SI Fig S2 — bump drift | `snn_slam_bump_drift` | `python figures_and_scripts/figure3_bump_drift/plot_bump_drift_corrected.py` |
| SI Fig S3 — perceptual-aliasing setup | `snn_slam_aliasing_setup` | `python figures_and_scripts/figure9_aliasing_stress/plot_aliasing_setup.py` |

**Data vs schematic.** Figs 2–6 and SI Figs S1–S3 are **data figures** generated from real
simulation runs (they read/compute the committed `*.json` / `*.npz`). **Fig 1 is a hand-authored
vector schematic** (`figure1_architecture/snn_slam_architecture.svg`, rendered to PDF by
`make_architecture.py` via `cairosvg`)—an illustration with no underlying data; every block
corresponds to a real component in `neuro-symbolic-slam/src/`.

### Regenerating the underlying data

```bash
# Attitude anchor ablation (Fig 2A): random + circuit, n=12
python neuro-symbolic-slam/scripts/clean_anchor_ablation.py --seeds 12
# -> figures_and_scripts/figure_attitude_anchor/anchor_ablation_results.json

# Ring diagnostics on the deployed PoseCANN (Fig 3 / Fig 2B / Fig S2), hardware-rate IMU front-end
python figures_and_scripts/figure8_classical_baseline/rerun_ring_diagnostics.py
# -> figures_and_scripts/figure8_classical_baseline/ring_diagnostics_corrected.json

# Occlusion diagnostic (Fig 2C), deployed complementary filter
python figures_and_scripts/figure6_sensory_deprivation/simulate_sensory_deprivation.py
# -> figures_and_scripts/figure6_sensory_deprivation/sensory_deprivation_data.npz

# Loop closure on revisit-rich circuits (Fig 4 / Fig 5), n=12, 2000 steps
python neuro-symbolic-slam/scripts/slam_variance.py --compare-lc --course circuit --seeds 12 --steps 2000
# -> figures_and_scripts/figure7_loop_closure/loop_closure_results.json (+ _timeseries.npz, _graph.npz)

# Open-course Monte Carlo (SI Fig S1), n=20, 600 steps
python neuro-symbolic-slam/scripts/slam_variance.py --seeds 20 --steps 600
# -> figures_and_scripts/figure5_monte_carlo/variance_results.json (+ _timeseries.npz)

# Compute / energy estimate (Fig 6)
python figures_and_scripts/figure4_compute_efficiency/profile_snn_energy_v2.py
# -> figures_and_scripts/figure4_compute_efficiency/energy_v2_results.json
```

### Verifying against the committed artifacts

Every quantitative claim in the paper is encoded as a numeric check with an explicit tolerance:

```bash
python figures_and_scripts/verify_figures.py   # 65 checks, prints PASS/FAIL per value
```

This validates the committed `*.json` / `*.npz` artifacts (or any you have regenerated) against
the values reported in the paper, without re-running any simulation.

### Headline numbers (current)

| Result | Value | Figure |
|---|---|---|
| Gravity-anchor attitude ablation (mean abs. error, n=12) | random **8.7°→2.9°**, circuit **47.2°→7.5°** | Fig 2A |
| Vibration envelope (attitude RMSE, hardware-rate IMU front-end, 0.1–10 rad/s, deployed K_g=5) | anchored ring **0.66°→3.1°** tracks CF with small overhead (grand means ring 1.44°, CF 0.92°, EKF 0.75°); unanchored drifts 2.2°→12.7° (grand mean 7.47°) | Fig 2B / 3A |
| Transient accel outliers (0→40 %/step) | EKF **0.21°→3.69°**; lower-gain CF **0.33°→1.61°** and ring **0.60°→1.67°** stay bounded | Fig 3B |
| Reference-dropout occlusion (window RMSE) | unanchored **2.9±0.9°** vs gravity-anchored **1.1±0.2°** (~62 %, n=10) | Fig 2C |
| Loop closure (revisit-rich circuit, n=12, 2000 steps) | IMU 18.8, LC-OFF 14.8, LC-ON **2.6 cm**; paired Δ **12.2 cm** [11.0, 13.5] (~83 %); **precision 1.000** (2950/2950), mean **246** closures/run | Fig 4 / 5 |
| Neuromorphic compute (neural layers only) | **7.1×10⁶ SOPs/s**, **~0.09–0.19 mW** as built; minimal task budget **0.16×10⁶ SOPs/s** (2–4 µW) (ODIN/Loihi/TrueNorth per-op band) | Fig 6 |
| Open-course Monte Carlo ATE (n=20) | IMU / anchor-off / anchor-on **3.34 / 3.26 / 3.05 cm** (statistically indistinguishable) | Fig S1 |
| Bump drift, 30 s, persistent gyro bias | unanchored **51.1°** vs anchored **1.9°** (~96 %) | Fig S2 |

> **Notes.** (1) **The sensors do not share the estimator's clock.** The MEMS IMU samples at its own ~1 kHz
> hardware rate and the estimator consumes the mean rate over each step (`snn_slam_system.imu_rate_average`,
> imported by the figure harness), so a wingbeat tone is attenuated by |sinc(f·Δt)| (8.7× at 115 Hz)
> rather than aliased. One front-end, shared by the deployed system and every diagnostic.
> (2) The translational velocity is a **modeled event+ToF visual-odometry estimate** with a
> realistic error model, **ON by default** (`SLAM_VO_NOISE=1`) and applied **identically to all
> baselines**—so the position/ATE numbers reflect real odometry drift. See `PROVENANCE.md`.

The committed `*.json` / `*.npz` files hold the exact values reported in the paper.

## Cite this

```bibtex
@article{li2026gravityanchored,
  title   = {Gravity-Anchored Neuromorphic Navigation: Sub-Milliwatt Attitude
             Stabilization and Loop Closure for Insect-Scale Flyers},
  author  = {Li, Hao and Zhang, Yanlai and Chai, Runqi and Wu, Jianghao},
  year    = {2026},
  note    = {Manuscript under review},
  url     = {https://github.com/lhooz/gravity-anchored-slam}
}
```

A citable **Zenodo DOI** for this code will be minted from a tagged release and added here.

## License

MIT — see [LICENSE](LICENSE).
