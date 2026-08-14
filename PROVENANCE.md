# Number Provenance — Sub-Milliwatt Neuromorphic Topological SLAM (2D)

Maps every quantitative claim in the manuscript to the code/data that produce it. Regenerate
figures with the per-figure scripts under `code/figures_and_scripts/` and the experiment drivers
under `code/neuro-symbolic-slam/scripts/`, or run `figures_and_scripts/make_all_figures.py`.

All position/trajectory-error results use a **modeled event+ToF visual-odometry velocity** (see
"Velocity-error model" below), applied identically to every baseline; env `SLAM_VO_NOISE`
(default on). The accuracy-relevant claims are **attitude stabilization** (real corrupted gyro,
hardware-rate IMU front-end) and **precision-verified loop closure**; the position/ATE
results are a mechanism demonstration on realistic odometry drift.

## Headline results (5 main figures beyond the architecture schematic + 3 SI)

| Manuscript claim | Value | Script / data |
|---|---|---|
| **Fig 1** — system architecture | hand-authored vector schematic (no underlying data) | `figure1_architecture/make_architecture.py` (renders `snn_slam_architecture.svg`) |
| **Fig 2A** — gravity-anchor clean ablation, pitch-attitude error (anchor off→on), n=12 | random **8.7±4.1°→2.9±0.5°**; circuit **47.2±19.2°→7.5±2.1°** (mean abs. error ± 95 % CI) — attitude cut several-fold *and variance sharply reduced*, even with reliable vision; paired ATE 3.52→3.16 random; 20.45→14.83 circuit (CIs contain zero) | `scripts/clean_anchor_ablation.py --seeds 12` → `figure_attitude_anchor/anchor_ablation_results.json` |
| **Fig 2B / Fig 3A** — vibration envelope (IMU-only), mean over 10–260 Hz per amplitude, deployed anchor gain K_g=5 | anchored ring **0.66°→3.1°** tracks the raw-CF baseline with a small substrate overhead (grand means ring **1.44°**, CF **0.92°**); unanchored ring drifts (**2.2°→12.7°**, grand mean 7.47°); 1D EKF slightly better (**0.75°**). The wingbeat is attenuated ~8.7× (the 20-sample rate-average, ≈|sinc(f·Δt)|) in the IMU front-end. | `figure8_classical_baseline/rerun_ring_diagnostics.py` → `ring_diagnostics_corrected.json` |
| **Fig 2C** — occlusion window (t∈[5,10] s), deployed complementary filter | unanchored **2.9±0.9°** vs gravity-anchored **1.1±0.2°** (~62 %, n=10 seeds; deployed dt + deployed CF) | `figure6_sensory_deprivation/simulate_sensory_deprivation.py` → `sensory_deprivation_data.npz` |
| **Fig 3B** — transient accel-outlier sweep, 0→40 %/step | the EKF's steady-state gain (K→0.040, ~2×α_fuse) admits the outliers (**0.21°→3.69°**); the lower-gain CF (**0.33°→1.61°**) and the ring reading it out (**0.60°→1.67°**) stay bounded | `figure8_classical_baseline/rerun_ring_diagnostics.py` (outliers block) |
| **Fig 4** — loop closure, revisit-rich circuit, n=12, 2000 steps | IMU **18.8**, LC-OFF **14.8**, LC-ON **2.6** cm; paired Δ **12.2** [11.0, 13.5] cm (CI-separated, **~83 %**); **precision 1.000** (GT-verified < 0.30 m, 2950/2950); **246** closures/run (212–266). IMU ≈ LC-OFF ⇒ event VO does not beat aligned dead-reckoning; loop closure is the alignment-proof win. | `scripts/slam_variance.py --compare-lc --course circuit --seeds 12 --steps 2000` (set `LC_AUDIT=1` for the pre-gate audit) → `figure7_loop_closure/loop_closure_results.json`; `plot_loop_closure.py` |
| **Fig 5** — topological map, matched-revisit residual | before relaxation ≈ **17 cm**, after relaxation ≈ **2 cm** (same detected revisits in both; only the pose-graph relaxation differs) — a mechanism view; accuracy is quantified in Fig 4. Chord lengths read from the dumped pose-graph nodes `node_orig`/`node_corrected`. | pose graph dumped by `slam_variance.py --compare-lc` (`loop_closure_results_graph.npz`); `plot_topomap.py` |
| **Fig 6** — neuromorphic compute | **7.1×10⁶** SOPs/s; **≈0.09–0.19 mW** (sub-milliwatt) band ({ODIN 12.7, Loihi 23.6, TrueNorth 26} pJ/SOP). Neural-layers-only estimate; layer sizes read from the deployed network (**256** place cells) scoped to the rate-coded attractors + place layer (grid counts all 3 dense matrices/module; the event-driven spiking vision front-end is disclosed separately); **dt-independent** rate-to-spike mapping (peak activity → R_MAX = 100 Hz). | `figure4_compute_efficiency/profile_snn_energy_v2.py` → `energy_v2_results.json` |
| **Equivalence (TOST)** — open-course ATE null, n=20 | paired TOST vs ±0.5 cm margin on the current variance data; paired-diff 95 % bootstrap CIs contain zero | `figure5_monte_carlo/equivalence_test.py --json variance_results.json` → `equivalence_test_results.json` |
| **SI Fig S1** — open-course Monte Carlo, n=20 | IMU / anchor-off / anchor-on **3.34 / 3.26 / 3.05** cm — statistically indistinguishable: position is set by the modeled VO velocity, not by attitude | `scripts/slam_variance.py --seeds 20 --steps 600` → `figure5_monte_carlo/variance_results.json`; `plot_monte_carlo.py` |
| **SI Fig S2** — bump drift, 30 s, persistent gyro bias | unanchored **51.1°** vs anchored **1.9°** (~96 %); dotted line = RMS of the ground-truth sinusoid (**60.8°**) | `figure8_classical_baseline/rerun_ring_diagnostics.py` (bump_drift block) → `ring_diagnostics_corrected.json`; `figure3_bump_drift/plot_bump_drift_corrected.py` |
| **SI Fig S3** — perceptual-aliasing setup | schematic drawn from the actual obstacle generators (14 landmarks; k=2 symmetric replica; A/A′ ≈ 0.8 m apart) | `figure9_aliasing_stress/plot_aliasing_setup.py` (imports `sparse_forest` generators) |
| **Aliasing stress (SI table)** — perceptual-aliasing loop closure | k-fold symmetric landmark ring; pre-gate vs fired precision vs control circuit | `scripts/slam_variance.py --compare-lc --course=circuit_alias` (`LC_AUDIT=1`, `SLAM_ALIAS_KFOLD`) → `figure9_aliasing_stress/` |

## IMU front-end — hardware sampling rates (single source of truth)

**The sensors do not share the estimator's clock.** The MEMS IMU samples at its own hardware rate
(`IMU_OVERSAMPLE = 20` × the 50 Hz estimator = **1 kHz**), where the 115 Hz wingbeat is properly
represented and is *not* aliased. What the estimator consumes each step is the **mean** angular
rate / specific force over that step — exactly what integrating the IMU at its native rate over
the interval yields. Implemented once, in the deployed system:
`snn_slam_system.imu_rate_average()`, and **imported** by the figure-side ring diagnostics
(`figure8_classical_baseline/imu_filters.py`), so the deployed system and the diagnostics cannot
diverge.

Consequences (all physical, no synthetic filter involved):
* A zero-mean tone is attenuated by the 20-sample rate-average (Dirichlet kernel, ≈|sinc(f·Δt)|) — **0.114 (8.7×)** at 115 Hz.
* A tone at an exact multiple of 1/Δt = 50 Hz integrates to **zero** over a step: a real physical
  null (a whole number of periods), not a point-sampling artifact. The envelope frequency grid
  therefore avoids those points (`freqs = [10, 30, 70, 115, 170, 260]`).

This replaces two earlier, wrong models: (i) point-sampling the tone *at* the 50 Hz estimator rate
(which fabricated a persistent 15 Hz disturbance and zeroed tones on the sample-rate comb), and
(ii) a synthetic anti-alias Butterworth filter applied only in the figure harness (which decoupled
the figures from the deployed system and made the wingbeat numerically inert).

## Velocity-error model (the modeled VO front-end)

The translational velocity delivered to the position path-integrator is a **modeled** event+ToF
VO estimate, not ground truth. On the true velocity `[vx, vy]` we apply: a per-trial
multiplicative metric-scale bias (~4 %), per-step magnitude (~8 %) and direction (~3°) noise, and
a slowly-varying velocity-bias random walk. Injected at the single `kin` source
(`snn_slam_system.py`, `generate_new_chunk`, after the gyro-error split) so the IMU /
anchor-off / anchor-on baselines all share it and the synthesized accelerometer / gravity
**attitude** channel stays clean. Env `SLAM_VO_NOISE` (default on). Consequence: position/ATE is a
mechanism demonstration on realistic odometry drift — not an idealized best case and not
hardware-realistic accuracy from a fielded estimator. A real 2D-event-camera VO front-end is
future work.

## Method / sensor provenance (unchanged by the velocity model)

| Item | Value | Source |
|---|---|---|
| Sensor suite | 1D event camera (256 px, 90° FOV) + 3-beam ToF (0/±45°, 0.1–2.83 m = √8 room diagonal) + planar IMU (2-axis accel + pitch-rate gyro); 2×2 m walled arena | `sparse_forest.py` (ROOM_W/H=2.0, N_PIXELS=256, FOV_DEG=90); `snn_slam_system.py` (ToF coder, angles [−π/4, 0, +π/4]) |
| ToF → metric VO scale | `v_x_vis = v_x_scale·(F_left·d_left + F_right·d_right)`; the dimensionless event rate `F` is made metric by the ToF depth | `snn_slam_system.py` (`phase_odometry`) |
| ToF → place code | `primary_senses = I_csnn_place · I_tof_place` (multiplicative gate): a place cell responds only when the visual and depth cues agree | `snn_place_cells.py` |
| Place representation | grid-cell CANN (co-prime periods 11/13/17) → sparse 1D place-cell code + a sparse binary appearance hash (fixed random projection + k-WTA, 32-of-256; FlyHash motif, Dasgupta et al. 2017 -- NOT hyperdimensional/VSA: no binding, bundling, or permutation) | `snn_pose_cann.py`, `snn_place_cells.py` |
| Gravity anchor | complementary filter (`ALPHA_FUSE = 0.02`, gyro-dominated) fusing accelerometer gravity + gyro, injected as a Gaussian current (`K_GRAVITY = 5`, env-overridable, used by the full system and the IMU-only ring diagnostics alike; `SIGMA_GRAVITY = 0.15`) into the ring CANN; anchor-off = `K_GRAVITY = 0` | `snn_slam_system.py` (`forward_step`), `snn_pose_cann.py` |
| Appearance-keyed loop closure | candidates by appearance-hash cosine overlap ≥ 0.60 (true revisits ~0.72 vs ~0.03 unrelated) at > 10 keyframes separation; geometric gate (< 0.25 m, < 0.35 rad). **Fired precision comes from the GEOMETRIC gate, not the descriptor**: the hash is a lossy compression of the CSNN features (AUC 0.909 vs 0.933 for dense-feature cosine) and its precision cost is absorbed downstream. The place-cell confidence gates (`is_conf`) govern MAP LEARNING and do NOT gate closure firing | `scripts/slam_variance.py` (`run_trial`) |
| Pose-graph relaxation | 3-DOF SE(2) force-directed spring relaxation; damped-momentum gradient descent with simulated-annealing damping; DCS robust kernel (Agarwal/Olson/Stachniss 2013); loop-spring weight `LC_WEIGHT_SCALE=0.12`; first node frozen | `snn_slam_system.py` (`relax_graph`) |
| ATE alignment | rigid SE(2) Umeyama (rotation + translation, **no scale**), applied identically to IMU / anchor-off / anchor-on before scoring | `snn_slam_system.py` (`get_optimal_alignment_2d`); `scripts/slam_variance.py` |
| Energy profiling config | dt-independent rate-to-spike mapping: firing rate = R_MAX·(mean/peak) activity shape (R_MAX=100 Hz); SOP = presynaptic rate × presynaptic count × fan-out, summed; grid = 121+169+289 = 579 neurons; place = 256 | `figure4_compute_efficiency/profile_snn_energy_v2.py` |
| Wingbeat frequency (115 Hz) | representative demanding test point: ~120 Hz RoboBee (Ma2013), ~100–130 Hz hornets; 10–260 Hz is a robustness envelope (attenuated by |sinc(f·Δt)| in the rate-averaging IMU front-end) | literature; `snn_slam_system.imu_rate_average`, `figure8_classical_baseline/rerun_ring_diagnostics.py` |
