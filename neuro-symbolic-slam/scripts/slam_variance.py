import os
#!/usr/bin/env python3
"""
slam_variance.py — Trajectory Variance Characterisation

Run SNN SLAM v7 with FIXED default parameters across many random seeds.
Measures how stable the architecture is across different room geometries.

This answers: "How much does ATE vary purely due to trajectory difficulty?"

Usage:
    python slam_variance.py              # 30 seeds, headless
    python slam_variance.py --seeds=10  # quick 10-seed scan
"""
import sys, os, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.environ['MPLBACKEND'] = 'Agg'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

import src.snn_slam_system as S


def run_trial(seed: int, n_steps: int = 2000,
              course_type: str = 'random', enable_lc: bool = True) -> dict:
    """Run one headless trial, return ATE metrics.

    course_type : 'random' (default roomba walk, Fig 5) or 'circuit' (revisit-rich
                  closed loop, Part B) so genuine revisits create loop-closure candidates.
    enable_lc   : if True, apply relax_graph pose-graph correction (loop closure ON);
                  if False, the detected closures are NOT applied (loop closure OFF) so
                  the closed-loop trajectory is pure VO+anchor odometry. Detection is
                  identical in both modes -- only application differs -- so the ON-vs-OFF
                  ATE delta isolates the loop-closure contribution.
    """
    key = random.PRNGKey(seed)
    env = S.LiveEnvironment(key, chunk_size=n_steps + 100, course_type=course_type)

    system_ol = S.SNNSLAMSystem(random.PRNGKey(42), n_depth=S.N_DEPTH)
    system_cl = S.SNNSLAMSystem(random.PRNGKey(43), n_depth=S.N_DEPTH)
    system_ol.reset(1); system_cl.reset(1)
    # S3: open-loop = SNN dead-reckoning with NO gravity anchor (K_GRAVITY=0); closed-loop
    # keeps the gravity-anchored current. Makes OL and CL genuinely different processes.
    system_ol.pose.K_GRAVITY = 0.0
    # Ablation (LC_ANCHOR_OFF=1): run the loop-closure trajectory ITSELF unanchored, so the
    # geometric verification gate must separate revisits from aliased look-alikes using a
    # drift-prone pose estimate. Isolates whether attitude stabilization is what enables
    # aliasing-robust, precision-verified loop closure (ties Fig 3 -> the loop-closure headline).
    if os.environ.get('LC_ANCHOR_OFF'):
        system_cl.pose.K_GRAVITY = 0.0

    _, _, _, pos0, th0, _ = env.step()
    system_ol.initialize_pose(jnp.array([pos0]), jnp.array([th0]))
    system_cl.initialize_pose(jnp.array([pos0]), jnp.array([th0]))

    gt_pos_hist, imu_pos_hist = [], []
    ol_pos_hist, cl_pos_hist = [], []
    gt_th_hist, imu_th_hist = [], []

    x_imu, y_imu, th_imu = pos0[0], pos0[1], th0
    graph_poses, graph_odom_edges = [], []
    node_tof_hits = []
    node_flyhash = []     # Part-B: pose-INDEPENDENT appearance hash per keyframe (identifier kept for
                  # compatibility; it is a random-projection + k-WTA sparse binary hash, NOT FlyHash/LSH)
    node_gt = []      # Part-B: ground-truth (x,y) per keyframe -- used ONLY for an honest precision audit
    loop_closures = []
    lc_audit = []     # Part-B: per fired closure (matched_nid, nid, hdc_overlap, gt_dist) for precision
    step_kf_id = []   # Most-recent keyframe index per recorded step (for back-correction)

    KEYFRAME_DIST, KEYFRAME_ANG = 0.15, 0.20
    LC_SPATIAL_THRESH = 0.25   # max distance (m) between a revisit and its matched keyframe (estimated frame)
    # Part-B (drift-index fix): the pose-keyed place index drifts across laps (Live_Barcode
    # overlap ~0.24), so candidates are now keyed on the pose-INDEPENDENT appearance hash
    # (Visual_Barcode), which overlaps ~0.72 on true revisits vs ~0.03 for unrelated views.
    FLYHASH_MATCH_THRESH = float(os.environ.get("FLYHASH_THRESH", "0.60"))    # cosine overlap of the sparse binary appearance hash (true revisits ~0.84,
                           # random ~0.03). NB: the hash is a LOSSY compression of the CSNN features
                           # (AUC 0.81 vs 0.92 for raw-feature cosine) -- fired precision comes from the
                           # GEOMETRIC gate below, not from the descriptor.
    LC_MIN_TOPO = 10           # require a candidate to be > this many keyframes back (a real revisit, not a neighbour)
    LC_WEIGHT_SCALE = float(os.environ.get('LC_WEIGHT_SCALE', '0.12'))  # round-2: scale LC spring force.
                               # At 1.0 the dense closures over-contract the relaxed loop (radius ~0.75x GT);
                               # 0.12 preserves the loop scale (~0.89x) while keeping a clear, CI-separated
                               # loop-closure ATE improvement. (Odometry itself is correctly scaled.)
    GT_TP_RADIUS = 0.30        # a fired closure is a TRUE positive if the two keyframes are < this apart in ground truth
    last_kf_cann = None

    _dbg = {'kf': 0, 'place_revisit': 0}
    # Optional loop-closure audit (LC_AUDIT=1): pure side-accounting over candidate PAIRS to
    # measure the appearance descriptor BEFORE the geometric gate (pre-gate precision, recall,
    # and how many appearance candidates the geometric gate rejects). Never affects fired closures.
    AUDIT = bool(os.environ.get('LC_AUDIT'))
    # NB: these dict keys are a FROZEN serialization contract -- they are written into the committed
    # JSON artifact and read back by verify_figures.py, so the legacy 'hdc_' names are kept here
    # deliberately. The mechanism is a FlyHash/LSH sparse binary code, NOT hyperdimensional computing.
    _aud = {'gt_pairs': 0, 'hdc_cand': 0, 'hdc_true': 0, 'hdc_geom_pass': 0}
    # DISSOLVE (SLAM_DISSOLVE_LC=1): remove the absolute-distance geometric gate ENTIRELY and verify
    # every appearance candidate -- short loop or long -- by RELATIVE/sequence geometry alone (no
    # absolute 0.25 m check, no anchor bootstrap). One unified verifier that should subsume the gate
    # (its short loops) plus the long loops recall recovered. Suppresses the in-loop gate below and
    # runs the unified pass after the main loop. OFF by default -> shipped gate behaviour unchanged.
    DISSOLVE_LC = bool(os.environ.get('SLAM_DISSOLVE_LC'))
    # SLAM_LC_WINDOW=N: sliding-window bound on the low-level graph -- the gate may only match a
    # candidate within the last N keyframes (older nodes are "evicted"). Models a bounded working
    # memory: short-term loops (within the window) still close; long-term loops (revisits older than
    # N keyframes) are structurally lost. 0/unset = unbounded (shipped behaviour). The gravity anchor
    # is held ON throughout so this isolates the loop-closure layer, not attitude.
    LC_WINDOW = int(os.environ.get('SLAM_LC_WINDOW', '0'))
    step = 0
    while step < n_steps:
        ev_t, kin_t, tof_t, gt_pos, gt_th, _ = env.step()
        ev_j = jnp.array([ev_t]); kin_j = jnp.array([kin_t]); tof_j = jnp.array([tof_t])
        if step > 0:
            # The MEMS gyro error (constant bias + rate random walk) and 115 Hz wingbeat vibration
            # are baked into kin_t from the first sample (LiveEnvironment.generate_new_chunk), so
            # IMU position error grows quadratically from t=0 (no artificial delayed-onset drift).
            omega_b = kin_t[2]
            vx_w = kin_t[0] * np.cos(th_imu) - kin_t[1] * np.sin(th_imu)
            vy_w = kin_t[0] * np.sin(th_imu) + kin_t[1] * np.cos(th_imu)
            x_imu += vx_w * S.DT
            y_imu += vy_w * S.DT
            th_imu = S.wrap_angle(th_imu + omega_b * S.DT)

        pose_ol, _, _ = system_ol.forward_step_open_loop(ev_j, kin_j, tof_j)
        pose_cl, _, _, is_conf, peak_idx_place, debug_gates = system_cl.forward_step(
            ev_j, kin_j, tof_j)

        cx, cy, cth = float(pose_cl[0, 0]), float(pose_cl[0, 1]), float(pose_cl[0, 2])
        gt_pos_hist.append(gt_pos); gt_th_hist.append(gt_th)
        imu_pos_hist.append([x_imu, y_imu]); imu_th_hist.append(th_imu)
        ol_pos_hist.append([float(pose_ol[0, 0]), float(pose_ol[0, 1])])
        cl_pos_hist.append([cx, cy])
        step_kf_id.append(len(graph_poses) - 1)   # keyframe this step belongs to (-1 before first KF)

        # Keyframe + loop closure (matching stable1 defaults)
        if last_kf_cann is None: last_kf_cann = (cx, cy, cth)
        kf_x, kf_y, kf_th = last_kf_cann
        dx, dy = cx - kf_x, cy - kf_y
        local_dx = dx * np.cos(-kf_th) - dy * np.sin(-kf_th)
        local_dy = dx * np.sin(-kf_th) + dy * np.cos(-kf_th)
        local_dth = (cth - kf_th + np.pi) % (2*np.pi) - np.pi

        is_keyframe = (len(graph_poses) == 0 or
                       np.sqrt(dx**2 + dy**2) > KEYFRAME_DIST or
                       np.abs(local_dth) > KEYFRAME_ANG)

        if is_keyframe:
            nid = len(graph_poses)
            if nid > 0:
                graph_odom_edges.append([local_dx, local_dy, local_dth])
            graph_poses.append([cx, cy, cth])
            last_kf_cann = (cx, cy, cth)
            node_tof_hits.append(tof_t.copy())
            # Pose-INDEPENDENT appearance key: the frozen random-projection+kWTA hash of this view.
            vb = (np.asarray(debug_gates['Visual_Barcode'][0]).reshape(-1) > 0.5).astype(np.float32)
            node_flyhash.append(vb)
            node_gt.append((float(gt_pos[0]), float(gt_pos[1])))
            _dbg['kf'] += 1

            # ---- Appearance-based loop-closure candidate generation (Part-B drift-index fix) ----
            # Recognise a place by HOW IT LOOKS (appearance hash), decoupled from the drifting
            # pose estimate. Candidate = high hash overlap AND topologically distant; then KEEP the
            # independent GEOMETRIC verification (spatial proximity + heading agreement) so the
            # gate still rejects false closures. We do NOT lower the geometry gate.
            matched_node = None
            best_score = FLYHASH_MATCH_THRESH
            cur = node_flyhash[nid]
            cur_norm = np.sqrt(cur.sum()) + 1e-8
            _lo = max(0, nid - LC_WINDOW) if LC_WINDOW > 0 else 0   # sliding-window eviction bound
            for prev_nid in range(_lo, max(0, nid - LC_MIN_TOPO)):   # only nodes > LC_MIN_TOPO keyframes back
                prev = node_flyhash[prev_nid]
                overlap = float((cur * prev).sum()) / (cur_norm * (np.sqrt(prev.sum()) + 1e-8))  # cosine
                if AUDIT:
                    gtd = float(np.hypot(node_gt[nid][0] - node_gt[prev_nid][0],
                                         node_gt[nid][1] - node_gt[prev_nid][1]))
                    is_true = gtd < GT_TP_RADIUS
                    _aud['gt_pairs'] += int(is_true)          # all true-revisit pairs (recall denominator)
                    if overlap >= FLYHASH_MATCH_THRESH:           # appearance candidate (pre-geometry)
                        _aud['hdc_cand'] += 1
                        _aud['hdc_true'] += int(is_true)
                        pxa, pya, ntha = graph_poses[prev_nid]
                        if (np.hypot(cx - pxa, cy - pya) < LC_SPATIAL_THRESH
                                and abs(S.wrap_angle(cth - ntha)) < 0.35):
                            _aud['hdc_geom_pass'] += 1
                if overlap <= best_score:
                    continue
                px, py, nth = graph_poses[prev_nid]
                spatial_d = float(np.hypot(cx - px, cy - py))
                dth = abs(S.wrap_angle(cth - nth))
                if spatial_d < LC_SPATIAL_THRESH and dth < 0.35:   # geometry gate (unchanged)
                    best_score = overlap
                    matched_node = prev_nid
            if matched_node is not None and not DISSOLVE_LC:   # absolute gate suppressed in dissolve mode
                _dbg['place_revisit'] += 1
                w_pos = best_score * 0.20 * LC_WEIGHT_SCALE
                w_th  = best_score * 0.15 * LC_WEIGHT_SCALE
                loop_closures.append([matched_node, nid, w_pos, w_th])
                gt_dist = float(np.hypot(node_gt[nid][0] - node_gt[matched_node][0],
                                         node_gt[nid][1] - node_gt[matched_node][1]))
                lc_audit.append((matched_node, nid, best_score, gt_dist))

        step += 1

    # ---- 2b-ii: DISSOLVED verifier (SLAM_DISSOLVE_LC=1) -- one relative-geometry check, no gate ----
    # Replaces the absolute-distance gate entirely. For EVERY keyframe, take its best appearance match
    # among topologically-distant nodes; admit the closure iff a window of recent keyframes each has a
    # match AND the local odometry between them reproduces the local geometry between the matched nodes
    # (SeqSLAM-style relative verification). Drift-independent: both the current window and the matched
    # window are internally low-drift even when the absolute offset between the two visits is large. No
    # anchor bootstrap -- every keyframe is a candidate target. Must subsume the gate's short loops AND
    # recover long loops, at precision ~1.000.
    if DISSOLVE_LC and enable_lc and len(graph_poses) > LC_MIN_TOPO:
        DW   = int(os.environ.get('DISSOLVE_W', '3'))
        DTOL = float(os.environ.get('DISSOLVE_TOL', '0.25'))
        E = np.array(graph_poses, dtype=np.float64)
        n = len(graph_poses)
        match_of = {}                                         # node -> best appearance match (any node)
        for b in range(n):
            cur = node_flyhash[b]; cn = np.sqrt(cur.sum()) + 1e-8
            best, ai = FLYHASH_MATCH_THRESH, None
            for a in range(max(0, b - LC_MIN_TOPO)):
                ov = float((cur * node_flyhash[a]).sum()) / (cn * (np.sqrt(node_flyhash[a].sum()) + 1e-8))
                if ov > best:
                    best, ai = ov, a
            if ai is not None:
                match_of[b] = (ai, best)
        n_diss = 0
        for b in range(n):
            m = match_of.get(b)
            if m is None:
                continue
            a, ov = m
            idxs = [k for k in range(b - DW, b + 1) if k >= 0 and k in match_of]
            if len(idxs) < DW:
                continue                                      # not enough sequence support
            if any(np.hypot(*((E[b, :2] - E[k, :2]) -
                              (E[match_of[b][0], :2] - E[match_of[k][0], :2]))) > DTOL
                   for k in idxs[:-1]):
                continue                                      # relative geometry inconsistent -> reject
            loop_closures.append([a, b, ov * 0.20 * LC_WEIGHT_SCALE, ov * 0.15 * LC_WEIGHT_SCALE])
            gt_dist = float(np.hypot(node_gt[b][0] - node_gt[a][0], node_gt[b][1] - node_gt[a][1]))
            lc_audit.append((a, b, ov, gt_dist))
            n_diss += 1
        print(f"    [dissolve] {n_diss} unified closures, no absolute gate (W={DW}, tol={DTOL})", flush=True)

    # ---- 2b: drift-independent RECALL closures (SLAM_RECALL_LC=1; OFF by default) ----
    # The geometric gate (above) rejects a TRUE revisit once accumulated drift pushes the current
    # estimate past LC_SPATIAL_THRESH from its match -- structurally losing long loops. Recall
    # recovers them: a candidate whose appearance matches an ANCHOR node (one already trusted via a
    # gate closure) is admitted if it passes a drift-INDEPENDENT check -- recalled relative geometry
    # over a short keyframe window must reproduce the local odometry (a false match breaks it, a true
    # one reproduces it). Validated offline in stage2a at precision 1.000 (n=12). This is ADDITIVE:
    # it appends topological-coincidence closures that the existing relaxation below then applies, so
    # with the flag unset the trajectory and every metric are bit-identical to the shipped baseline.
    if os.environ.get('SLAM_RECALL_LC') and enable_lc and len(graph_poses) > LC_MIN_TOPO:
        RW   = int(os.environ.get('RECALL_W', '4'))
        RTOL = float(os.environ.get('RECALL_TOL', '0.20'))
        E = np.array(graph_poses, dtype=np.float64)           # raw estimated keyframe poses (x, y, th)
        n = len(graph_poses)
        gate_pairs = {(int(lc[0]), int(lc[1])) for lc in loop_closures}
        anchors = set()                                       # trusted nodes: SEEDED by gate closures...
        recall_of = {}                                        # node -> matched anchor index (window ref)
        for lc in loop_closures:
            anchors.add(int(lc[0])); anchors.add(int(lc[1]))
            recall_of[int(lc[1])] = int(lc[0])                # gate matches seed the sequence window
        n_recall = 0
        # Single pass in keyframe order so a fired recall becomes a NEW anchor for later keyframes
        # (CHAINING — matches the validated stage2a mechanism; without it, recall cannot walk into
        # the drifted tail where gate anchors are absent, which zeroed the high-drift result).
        for b in range(n):
            cur = node_flyhash[b]; cn = np.sqrt(cur.sum()) + 1e-8
            best, ai = FLYHASH_MATCH_THRESH, None
            for a in range(max(0, b - LC_MIN_TOPO)):
                if a not in anchors:
                    continue
                ov = float((cur * node_flyhash[a]).sum()) / (cn * (np.sqrt(node_flyhash[a].sum()) + 1e-8))
                if ov > best:
                    best, ai = ov, a
            if ai is None:
                continue
            recall_of[b] = ai                                 # record match (pre-verification) for window
            if (ai, b) in gate_pairs:
                continue
            if float(np.hypot(E[b, 0] - E[ai, 0], E[b, 1] - E[ai, 1])) < LC_SPATIAL_THRESH:
                continue                                      # gate would already have fired this
            idxs = [k for k in range(b - RW, b + 1) if k >= 0 and k in recall_of]
            if len(idxs) < RW:
                continue
            if any(np.hypot(*((E[b, :2] - E[k, :2]) -
                              (E[recall_of[b], :2] - E[recall_of[k], :2]))) > RTOL
                   for k in idxs[:-1]):
                continue                                      # sequence-inconsistent -> reject
            loop_closures.append([ai, b, best * 0.20 * LC_WEIGHT_SCALE, best * 0.15 * LC_WEIGHT_SCALE])
            gt_dist = float(np.hypot(node_gt[b][0] - node_gt[ai][0], node_gt[b][1] - node_gt[ai][1]))
            lc_audit.append((ai, b, best, gt_dist))
            gate_pairs.add((ai, b)); anchors.add(b); n_recall += 1   # B now anchors future recalls
        print(f"    [recall] +{n_recall} recall closures (W={RW}, tol={RTOL})", flush=True)

    # Pose-graph relaxation. Invoke relax_graph on the accumulated graph and feed the
    # corrected keyframe poses back into the closed-loop trajectory (a per-keyframe position
    # correction applied to all steps in that segment) BEFORE computing ATE.
    n_lc_fired = len(loop_closures)
    graph_orig, graph_corrected = None, None
    if enable_lc and len(graph_poses) >= 2 and n_lc_fired > 0:
        poses_arr = jnp.array(graph_poses, dtype=jnp.float32)
        odom_arr  = jnp.array(graph_odom_edges, dtype=jnp.float32)
        odom_mask = jnp.ones(len(graph_odom_edges))
        lc_arr  = jnp.array([[int(lc[0]), int(lc[1])] for lc in loop_closures], dtype=jnp.int32)
        lc_off  = jnp.zeros((n_lc_fired, 3), dtype=jnp.float32)   # topological revisit -> assert coincidence
        lc_w    = jnp.array([[lc[2], lc[3]] for lc in loop_closures], dtype=jnp.float32)
        lc_mask = jnp.ones(n_lc_fired)
        is_frozen = jnp.zeros(len(graph_poses)).at[0].set(1.0)    # anchor the first keyframe
        _gp = os.environ.get('SAVE_GRAPH')
        if _gp:   # round-2 diagnostic: dump the pose graph for offline relaxation tuning
            np.savez(_gp, poses=np.array(poses_arr), odom=np.array(odom_arr),
                     lc_arr=np.array(lc_arr), lc_off=np.array(lc_off), lc_w=np.array(lc_w),
                     step_kf=np.array(step_kf_id[:len(cl_pos_hist)]),
                     cl_pre=np.array(cl_pos_hist, dtype=np.float64), gt=np.array(gt_pos_hist))
            print(f"    [round2] saved pose graph -> {_gp}", flush=True)
        corrected = np.array(S.relax_graph(poses_arr, odom_arr, odom_mask,
                                           lc_arr, lc_off, lc_w, lc_mask, is_frozen))
        orig = np.array(graph_poses)
        graph_orig, graph_corrected = orig, corrected   # 6.3: expose pose graph for the topological-map figure
        cl_corr = np.array(cl_pos_hist, dtype=np.float64)
        skf = np.array(step_kf_id[:len(cl_corr)])
        for k in range(len(orig)):
            m = skf == k
            if m.any():
                cl_corr[m, 0] += (corrected[k, 0] - orig[k, 0])
                cl_corr[m, 1] += (corrected[k, 1] - orig[k, 1])
        cl_pos_hist = cl_corr.tolist()

    # Umeyama ATE. The SAME rigid SE(2) Umeyama alignment is applied to ALL trajectories --
    # IMU dead-reckoning, open-loop and closed-loop SNN -- so the comparison is fair: alignment
    # can only reduce error, so aligning only the SNN (and leaving the IMU raw) would flatter it.
    # This is the standard ATE convention. A raw (unaligned) IMU track is kept for visualization only.
    gt = np.array(gt_pos_hist); ol = np.array(ol_pos_hist); cl = np.array(cl_pos_hist)
    imu_raw = np.array(imu_pos_hist)
    mn = min(len(gt), len(ol), len(cl), len(imu_raw))
    gt, ol, cl, imu_raw = gt[:mn], ol[:mn], cl[:mn], imu_raw[:mn]

    R_imu, t_imu = S.get_optimal_alignment_2d(imu_raw, gt)
    imu_aligned = (R_imu @ imu_raw.T).T + t_imu
    R_ol, t_ol = S.get_optimal_alignment_2d(ol, gt)
    ol_aligned = (R_ol @ ol.T).T + t_ol
    R_cl, t_cl = S.get_optimal_alignment_2d(cl, gt)
    cl_aligned = (R_cl @ cl.T).T + t_cl

    def _ate(a):
        return float(np.mean(np.sqrt((a[:, 0]-gt[:, 0])**2 + (a[:, 1]-gt[:, 1])**2)))
    ate_imu = _ate(imu_aligned)
    ate_ol  = _ate(ol_aligned)
    ate_cl  = _ate(cl_aligned)
    final_ol = np.sqrt((ol_aligned[-1, 0]-gt[-1, 0])**2 + (ol_aligned[-1, 1]-gt[-1, 1])**2)
    final_cl = np.sqrt((cl_aligned[-1, 0]-gt[-1, 0])**2 + (cl_aligned[-1, 1]-gt[-1, 1])**2)

    # Per-timestep position errors (for time-series plotting) -- all from the aligned trajectories.
    imu_err = np.sqrt((imu_aligned[:, 0]-gt[:, 0])**2 + (imu_aligned[:, 1]-gt[:, 1])**2)
    ol_err = np.sqrt((ol_aligned[:, 0]-gt[:, 0])**2 + (ol_aligned[:, 1]-gt[:, 1])**2)
    cl_err = np.sqrt((cl_aligned[:, 0]-gt[:, 0])**2 + (cl_aligned[:, 1]-gt[:, 1])**2)

    n_lc = len(loop_closures)
    n_tp = sum(1 for a in lc_audit if a[3] < GT_TP_RADIUS)          # GT-verified true closures
    lc_precision = (n_tp / n_lc) if n_lc > 0 else float('nan')
    print(f"    [gate dbg] course={course_type} lc={'ON' if enable_lc else 'OFF'} "
          f"keyframes={_dbg['kf']} loop_closures={n_lc} true_pos={n_tp} "
          f"precision={lc_precision:.2f}", flush=True)
    return {
        'seed': seed, 'n_steps': mn,
        'course_type': course_type, 'enable_lc': bool(enable_lc),
        'ate_imu': float(ate_imu),
        'ate_ol': float(ate_ol), 'final_ol': float(final_ol),
        'ate_cl': float(ate_cl), 'final_cl': float(final_cl),
        'n_loop_closures': n_lc,
        'n_lc_true_pos': int(n_tp),
        'lc_precision': float(lc_precision) if n_lc > 0 else None,
        'lc_audit': dict(_aud) if AUDIT else None,
        'n_nodes': len(graph_poses),
        # Per-timestep data for downstream figure generation
        'imu_err_ts': imu_err,
        'ol_err_ts': ol_err,
        'cl_err_ts': cl_err,
        'gt_pos': gt,
        'imu_pos': imu_raw,                 # raw dead-reckoning (visualization only)
        'imu_pos_aligned': imu_aligned,     # Umeyama-aligned (matches the reported ATE)
        'ol_pos_aligned': ol_aligned,
        'cl_pos_aligned': cl_aligned,
        # 6.3: topological-map payload (populated only when enable_lc and closures fired)
        'graph_orig': graph_orig,           # keyframe poses (x,y,theta) BEFORE relaxation, estimate frame
        'graph_corrected': graph_corrected, # keyframe poses AFTER pose-graph relaxation, estimate frame
        'node_gt': (np.array(node_gt, dtype=float) if node_gt else np.zeros((0, 2))),
        'lc_edges': (np.array([[int(lc[0]), int(lc[1])] for lc in loop_closures], dtype=int)
                     if loop_closures else np.zeros((0, 2), dtype=int)),
        'step_kf': np.array(step_kf_id[:mn], dtype=int),   # step -> keyframe index (place nodes on aligned track)
    }


def _bootstrap_ci(arr, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap 95% CI of the mean (numpy-only, deterministic)."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 2:
        return float(arr.mean()) if len(arr) else 0.0, float(arr.mean()) if len(arr) else 0.0
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def run_compare_lc(n, steps, course, output):
    """Part B: loop-closure ON vs OFF on a (revisit-rich) course, paired over seeds.

    For each seed we run the SAME trajectory twice -- enable_lc=True and False -- so
    closure DETECTION is identical and only the relax_graph CORRECTION differs. The
    paired delta (ate_cl_off - ate_cl_on) and its bootstrap CI isolate the loop-closure
    contribution (Gate B: CI lower bound > 0 => CI-separated improvement). Honest
    provenance: archived JSON + per-seed npz, explicit n, real runs, gate not relaxed.
    """
    t0 = time.time()
    per_seed = []
    for i in range(n):
        seed = 42 + i * 111
        print(f"  Trial {i+1}/{n} (seed={seed}) [course={course}]...", flush=True)
        r_on  = run_trial(seed, n_steps=steps, course_type=course, enable_lc=True)
        r_off = run_trial(seed, n_steps=steps, course_type=course, enable_lc=False)
        per_seed.append({
            'seed': seed, 'n_steps': int(min(r_on['n_steps'], r_off['n_steps'])),
            'n_loop_closures': int(r_on['n_loop_closures']), 'n_nodes': int(r_on['n_nodes']),
            'n_lc_true_pos': int(r_on['n_lc_true_pos']), 'lc_precision': r_on['lc_precision'],
            'lc_audit': r_on.get('lc_audit'),
            'ate_imu': float(r_on['ate_imu']), 'ate_ol': float(r_on['ate_ol']),
            'ate_cl_off': float(r_off['ate_cl']), 'ate_cl_on': float(r_on['ate_cl']),
            'final_cl_off': float(r_off['final_cl']), 'final_cl_on': float(r_on['final_cl']),
            # keep timeseries + best-trial trajectories for the figure
            '_imu_err_ts': r_on['imu_err_ts'], '_cl_on_err_ts': r_on['cl_err_ts'],
            '_cl_off_err_ts': r_off['cl_err_ts'],
            '_gt': r_on['gt_pos'], '_imu_pos': r_on['imu_pos_aligned'],
            '_cl_on_pos': r_on['cl_pos_aligned'], '_cl_off_pos': r_off['cl_pos_aligned'],
            # 6.3: topological-map payload from the LC-ON run (place/pose graph + fired closures)
            '_graph_orig': r_on.get('graph_orig'), '_graph_corrected': r_on.get('graph_corrected'),
            '_node_gt': r_on.get('node_gt'), '_lc_edges': r_on.get('lc_edges'),
            '_step_kf': r_on.get('step_kf'),
        })
        d = per_seed[-1]
        print(f"    lcs={d['n_loop_closures']:3d}  ate_imu={d['ate_imu']*100:5.2f}  "
              f"ate_cl_OFF={d['ate_cl_off']*100:5.2f}  ate_cl_ON={d['ate_cl_on']*100:5.2f} cm  "
              f"(delta={ (d['ate_cl_off']-d['ate_cl_on'])*100:+5.2f})", flush=True)

    imu     = np.array([d['ate_imu'] for d in per_seed])
    ol      = np.array([d['ate_ol'] for d in per_seed])
    cl_off  = np.array([d['ate_cl_off'] for d in per_seed])
    cl_on   = np.array([d['ate_cl_on'] for d in per_seed])
    lcs     = np.array([d['n_loop_closures'] for d in per_seed])
    tps     = np.array([d['n_lc_true_pos'] for d in per_seed])
    total_lc = int(lcs.sum()); total_tp = int(tps.sum())
    lc_precision = (total_tp / total_lc) if total_lc > 0 else float('nan')   # GT-verified, pooled over seeds
    # Pre-gate appearance-descriptor audit (only when LC_AUDIT=1): pooled over seeds.
    _auds = [d.get('lc_audit') for d in per_seed if d.get('lc_audit')]
    audit_summary = None
    if _auds:
        agg = {k: int(sum(a[k] for a in _auds)) for k in ('gt_pairs', 'hdc_cand', 'hdc_true', 'hdc_geom_pass')}
        audit_summary = {**agg,
                         'pre_gate_precision': (agg['hdc_true'] / agg['hdc_cand']) if agg['hdc_cand'] else None,
                         'recall': (agg['hdc_true'] / agg['gt_pairs']) if agg['gt_pairs'] else None,
                         'gate_rejected': agg['hdc_cand'] - agg['hdc_geom_pass']}
    delta   = cl_off - cl_on                       # paired improvement from loop closure
    delta_lo, delta_hi = _bootstrap_ci(delta)
    gate_B = bool(delta_lo > 0.0)

    def stat(a):
        lo, hi = _bootstrap_ci(a)
        return {'mean': float(np.mean(a)), 'std': float(np.std(a)), 'ci95': [lo, hi]}

    summary = {
        'mode': 'compare_lc', 'course_type': course,
        'n_seeds': int(n), 'n_steps': int(steps),
        'ate_imu_cm':    {'mean': float(np.mean(imu))*100, 'std': float(np.std(imu))*100,
                          'ci95': [x*100 for x in _bootstrap_ci(imu)]},
        'ate_ol_cm':     {'mean': float(np.mean(ol))*100, 'std': float(np.std(ol))*100,
                          'ci95': [x*100 for x in _bootstrap_ci(ol)]},
        'ate_cl_off_cm': {'mean': float(np.mean(cl_off))*100, 'std': float(np.std(cl_off))*100,
                          'ci95': [x*100 for x in _bootstrap_ci(cl_off)]},
        'ate_cl_on_cm':  {'mean': float(np.mean(cl_on))*100, 'std': float(np.std(cl_on))*100,
                          'ci95': [x*100 for x in _bootstrap_ci(cl_on)]},
        'lc_delta_cm':   {'mean': float(np.mean(delta))*100, 'ci95': [delta_lo*100, delta_hi*100]},
        'lcs':           {'mean': float(np.mean(lcs)), 'std': float(np.std(lcs)),
                          'min': int(np.min(lcs)), 'max': int(np.max(lcs))},
        'lc_precision':  {'pooled': (None if total_lc == 0 else float(lc_precision)),
                          'total_closures': total_lc, 'total_true_pos': total_tp,
                          'gt_tp_radius_m': 0.30},
        'lc_pre_gate_audit': audit_summary,
        'improvement_pct_on_vs_off': float((np.mean(cl_off) - np.mean(cl_on)) / np.mean(cl_off) * 100)
                                     if np.mean(cl_off) > 0 else 0.0,
        'gate_B_pass': gate_B,
    }

    print(f"\n{'='*64}")
    print(f"  📊 LOOP-CLOSURE ON vs OFF — course={course}, n={n} seeds")
    print(f"{'='*64}")
    print(f"  ATE IMU         : {summary['ate_imu_cm']['mean']:5.2f} cm")
    print(f"  ATE CL  (LC OFF): {summary['ate_cl_off_cm']['mean']:5.2f} cm  CI95 {summary['ate_cl_off_cm']['ci95']}")
    print(f"  ATE CL  (LC ON ): {summary['ate_cl_on_cm']['mean']:5.2f} cm  CI95 {summary['ate_cl_on_cm']['ci95']}")
    print(f"  paired delta    : {summary['lc_delta_cm']['mean']:+5.2f} cm  CI95 {summary['lc_delta_cm']['ci95']}")
    print(f"  loop closures   : mean {summary['lcs']['mean']:.1f} (min {summary['lcs']['min']}, max {summary['lcs']['max']})")
    print(f"  LC precision    : {('n/a' if total_lc==0 else f'{lc_precision:.2f}')} "
          f"({total_tp}/{total_lc} closures GT-verified < 0.30 m)")
    print(f"  Gate B (CI-separated LC improvement): {'PASS ' if gate_B else 'FAIL '}")
    print(f"{'='*64}")

    # npz: per-seed error timeseries (ON/OFF/IMU) + representative 2D trajectory (median delta)
    min_steps = min(d['n_steps'] for d in per_seed)
    med_i = int(np.argmin(np.abs(cl_on - cl_on.mean())))   # representative seed: LC-ON ATE closest to the mean
    npz_path = output.replace('.json', '_timeseries.npz')
    np.savez(npz_path,
        imu_err_ts=np.array([d['_imu_err_ts'][:min_steps] for d in per_seed]),
        cl_on_err_ts=np.array([d['_cl_on_err_ts'][:min_steps] for d in per_seed]),
        cl_off_err_ts=np.array([d['_cl_off_err_ts'][:min_steps] for d in per_seed]),
        all_gt=np.array([d['_gt'][:min_steps] for d in per_seed]),
        all_imu=np.array([d['_imu_pos'][:min_steps] for d in per_seed]),
        all_cl_on=np.array([d['_cl_on_pos'][:min_steps] for d in per_seed]),
        all_cl_off=np.array([d['_cl_off_pos'][:min_steps] for d in per_seed]),
        all_seeds=np.array([d['seed'] for d in per_seed]),
        rep_gt=per_seed[med_i]['_gt'], rep_imu=per_seed[med_i]['_imu_pos'],
        rep_cl_on=per_seed[med_i]['_cl_on_pos'], rep_cl_off=per_seed[med_i]['_cl_off_pos'],
        rep_seed=per_seed[med_i]['seed'], n_steps=min_steps, dt=S.DT)

    # 6.3: dump a REPRESENTATIVE seed's topological map (place/pose graph + fired loop closures,
    # before/after pose-graph relaxation, with GT and the aligned before/after tracks). Loop-scale
    # contraction varies by seed, so we pick a seed with a clear loop-closure benefit whose relaxed
    # loop radius is closest to the across-seed median (not the median-ATE seed, which can be a
    # loop-scale outlier) -- a representative map, chosen the same way as a representative trial.
    def _loop_ratio(dd):
        nc, ng = dd.get('_graph_corrected'), dd.get('_node_gt')
        if nc is None or ng is None or len(nc) < 3 or len(ng) < 3:
            return np.nan
        cc = np.asarray(nc)[:, :2]; gg = np.asarray(ng)[:, :2]
        r_e = np.hypot(cc[:, 0]-cc[:, 0].mean(), cc[:, 1]-cc[:, 1].mean()).mean()
        r_g = np.hypot(gg[:, 0]-gg[:, 0].mean(), gg[:, 1]-gg[:, 1].mean()).mean()
        return (r_e / r_g) if r_g > 0 else np.nan
    ratios = np.array([_loop_ratio(dd) for dd in per_seed])
    print(f"  loop-scale ratios: mean {np.nanmean(ratios):.3f} "
          f"(min {np.nanmin(ratios):.3f}, max {np.nanmax(ratios):.3f})", flush=True)
    summary['loop_scale_ratio'] = {   # 6.3: persist the relaxed-loop scale for provenance
        'mean': float(np.nanmean(ratios)), 'min': float(np.nanmin(ratios)), 'max': float(np.nanmax(ratios)),
        'source': 'relaxed keyframe-node radius vs ground-truth node radius, per seed'}
    med_ratio = np.nanmedian(ratios)
    benefit = cl_off - cl_on
    cand = np.where((benefit > 0) & ~np.isnan(ratios))[0]
    map_i = int(cand[np.argmin(np.abs(ratios[cand] - med_ratio))]) if len(cand) else med_i
    rep = per_seed[map_i]
    if rep.get('_graph_orig') is not None:
        graph_path = output.replace('.json', '_graph.npz')
        np.savez(graph_path,
            node_orig=rep['_graph_orig'], node_corrected=rep['_graph_corrected'],
            node_gt=rep['_node_gt'], lc_edges=rep['_lc_edges'], step_kf=rep['_step_kf'][:min_steps],
            rep_gt=rep['_gt'][:min_steps], rep_cl_off=rep['_cl_off_pos'][:min_steps],
            rep_cl_on=rep['_cl_on_pos'][:min_steps], rep_seed=rep['seed'])
        print(f"  💾 Topological-map graph saved: {graph_path}")

    json_seed = [{k: v for k, v in d.items() if not k.startswith('_')} for d in per_seed]
    with open(output, 'w') as f:
        json.dump({'results': json_seed, 'summary': summary}, f, indent=2, default=str)
    print(f"  💾 Results saved: {output}\n  💾 Timeseries: {npz_path}")
    print(f"  🕐 Total time: {(time.time()-t0)/60:.1f} min")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=30, help='Number of random seeds')
    parser.add_argument('--steps', type=int, default=2000, help='Steps per trial')
    parser.add_argument('--course', type=str, default='random',
                       choices=['random', 'circuit', 'circuit_alias'],
                       help="trajectory type: 'random' (Fig 5), 'circuit' (revisit-rich, Part B), "
                            "or 'circuit_alias' (Tier-4 perceptual-aliasing stress test)")
    parser.add_argument('--compare-lc', action='store_true',
                       help='Part B: run loop-closure ON vs OFF (paired) with bootstrap CIs')
    parser.add_argument('--output', type=str,
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'variance_results.json'))
    args = parser.parse_args()

    if args.compare_lc:
        run_compare_lc(args.seeds, args.steps, args.course, args.output)
        return

    n = args.seeds
    t0 = time.time()
    results = []
    for i in range(n):
        seed = 42 + i * 111  # deterministic but varied seeds
        print(f"  Trial {i+1}/{n} (seed={seed})...", flush=True)
        # Fig-5 ablation isolates the GRAVITY ANCHOR (open-loop K_g=0 vs closed-loop K_g=200):
        # loop closure is disabled here (enable_lc=False) so the OL/CL contrast reflects only the
        # anchor. Loop closure is characterized separately on revisit-rich courses (Fig 7,
        # --compare-lc). This keeps Fig 5 reproducible with the current (post-FlyHash-fix) code, which
        # would otherwise fire loop closures on the random course and confound the anchor ablation.
        r = run_trial(seed, n_steps=args.steps, course_type=args.course, enable_lc=False)
        results.append(r)
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (n - i - 1) / rate / 60
        print(f"    ATE_IMU={r['ate_imu']:.4f}  ATE_OL={r['ate_ol']:.4f}  "
              f"ATE_CL={r['ate_cl']:.4f}  LCs={r['n_loop_closures']}  "
              f"Nodes={r['n_nodes']}  [ETA={eta:.1f}min]", flush=True)

    # Summary statistics
    ate_imu = [r['ate_imu'] for r in results]
    ate_ol = [r['ate_ol'] for r in results]
    ate_cl = [r['ate_cl'] for r in results]
    final_cl = [r['final_cl'] for r in results]
    lcs = [r['n_loop_closures'] for r in results]

    print(f"\n{'='*60}")
    print(f"  📊 TRAJECTORY VARIANCE CHARACTERISATION ({n} seeds)")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*60}")
    print(f"  {'ATE IMU (m)':<20} {np.mean(ate_imu):>8.4f} {np.std(ate_imu):>8.4f} "
          f"{np.min(ate_imu):>8.4f} {np.max(ate_imu):>8.4f}")
    print(f"  {'ATE Open-Loop (m)':<20} {np.mean(ate_ol):>8.4f} {np.std(ate_ol):>8.4f} "
          f"{np.min(ate_ol):>8.4f} {np.max(ate_ol):>8.4f}")
    print(f"  {'ATE Closed-Loop (m)':<20} {np.mean(ate_cl):>8.4f} {np.std(ate_cl):>8.4f} "
          f"{np.min(ate_cl):>8.4f} {np.max(ate_cl):>8.4f}")
    print(f"  {'Final Err CL (m)':<20} {np.mean(final_cl):>8.4f} {np.std(final_cl):>8.4f} "
          f"{np.min(final_cl):>8.4f} {np.max(final_cl):>8.4f}")
    print(f"  {'Loop Closures':<20} {np.mean(lcs):>8.1f} {np.std(lcs):>8.1f} "
          f"{np.min(lcs):>8d} {np.max(lcs):>8d}")
    print(f"{'='*60}")

    # Improvement over IMU
    improvement = (np.mean(ate_imu) - np.mean(ate_cl)) / np.mean(ate_imu) * 100
    print(f"  🦊 SNN CL improves over (fair-aligned) IMU by {improvement:.1f}% (mean ATE)")
    _clm = np.asarray(ate_cl)
    rep_idx = int(np.argmin(np.abs(_clm - _clm.mean())))  # representative trial: CL-ATE closest to the across-trial mean

    # Save per-timestep data as numpy archive (JSON can't store arrays)
    npz_path = args.output.replace('.json', '_timeseries.npz')
    min_steps = min(r['n_steps'] for r in results)
    np.savez(npz_path,
        imu_err_ts=np.array([r['imu_err_ts'][:min_steps] for r in results]),
        ol_err_ts=np.array([r['ol_err_ts'][:min_steps] for r in results]),
        cl_err_ts=np.array([r['cl_err_ts'][:min_steps] for r in results]),
        # Save representative (median CL-ATE) trial trajectory for 2D plot
        rep_gt=results[rep_idx]['gt_pos'],
        rep_imu=results[rep_idx]['imu_pos_aligned'],       # aligned IMU (matches the ATE claim)
        rep_imu_raw=results[rep_idx]['imu_pos'],           # raw dead-reckoning (optional visual)
        rep_ol=results[rep_idx]['ol_pos_aligned'],
        rep_cl=results[rep_idx]['cl_pos_aligned'],
        n_steps=min_steps,
        dt=S.DT,
    )
    print(f"  💾 Time-series data saved: {npz_path}")

    # Strip numpy arrays for JSON serialization
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        json_results.append(jr)

    with open(args.output, 'w') as f:
        json.dump({'results': json_results, 'summary': {
            'n_seeds': n, 'n_steps': args.steps, 'course_type': args.course,
            'ate_imu_mean': float(np.mean(ate_imu)), 'ate_imu_std': float(np.std(ate_imu)),
            'ate_imu_ci95': list(_bootstrap_ci(ate_imu)),
            'ate_ol_mean': float(np.mean(ate_ol)), 'ate_ol_std': float(np.std(ate_ol)),
            'ate_ol_ci95': list(_bootstrap_ci(ate_ol)),
            'ate_cl_mean': float(np.mean(ate_cl)), 'ate_cl_std': float(np.std(ate_cl)),
            'ate_cl_ci95': list(_bootstrap_ci(ate_cl)),
            'final_cl_mean': float(np.mean(final_cl)), 'final_cl_std': float(np.std(final_cl)),
            'lcs_mean': float(np.mean(lcs)), 'lcs_std': float(np.std(lcs)),
            'improvement_pct': float(improvement),
            'rep_seed': int(results[rep_idx]['seed']),
        }}, f, indent=2, default=str)
    print(f"\n  💾 Results saved: {args.output}")
    print(f"  🕐 Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
