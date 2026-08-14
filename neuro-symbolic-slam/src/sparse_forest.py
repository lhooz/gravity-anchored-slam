#!/usr/bin/env python3
"""
Sparse Forest — 2D Navigation with 1D Event Camera
(Upgraded: Texture Indexing Fix & Holonomic Kinematics)
"""

import os
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOM_W = 2.0
ROOM_H = 2.0

N_PIXELS = 256
FOV_DEG = 90.0
DT = 0.02
TIME_STEPS = 2000        
BATCH_SIZE = 8

BARCODE_RESOLUTION = 512
THRESHOLD = 0.015

N_OBSTACLES = 15
OBS_SIZE_MIN = 0.02        # 2cm twig (×0.2 from 10m-room scale)
OBS_SIZE_MAX = 0.14        # 14cm stem (×0.2 from 10m-room scale)
OBS_MARGIN = 0.4           # 40cm wall buffer (×0.2)

VX_RANGE = (-0.5, 0.5)     # physical hornet forward speed cap
VY_RANGE = (-0.15, 0.15)   # physical lateral speed cap
OMEGA_RANGE = (-1.0, 1.0)

SAFE_MARGIN = 0.1          # 10cm spawn clearance (×0.2)
MAX_ROOM_ATTEMPTS = 100  
MAX_TRAJ_ATTEMPTS = 50   

TEX_FREQS = [0.5, 1.0, 2.0]
TEX_AMPS  = [0.8, 0.4, 0.2]

SEED = 42

# =============================================================================
# Revisit-rich "circuit" course (Part B: loop closure)
# A deterministic closed-loop (circle) traced repeatedly so the robot GENUINELY
# returns to prior (x,y) locations (geometric ground truth) -> place cells re-fire
# and the confidence+geometry loop-closure gate has real revisits to trigger on.
# Obstacles are confined to the central disk (inside the loop) so the circular path
# is collision-free while the arena keeps distinctive texture/ToF landmarks.
# =============================================================================
CIRCUIT_RADIUS    = 0.40                 # m; small loop so the surrounding landmark ring is in view
CIRCUIT_SPEED     = 0.6                  # m/s forward speed along the loop
# Landmark RING placed OUTSIDE the (small) loop. The camera faces the direction of
# travel (tangent); the tangent line bends outward, so landmarks in a ring just
# beyond the loop fall HEAD-ON in the forward FOV -- the same condition that makes the
# random course's place recognition fire (head-on features -> sharp, distinctive place
# codes). Each loop position faces a different arc of the ring, so revisits are
# recognised confidently WITHOUT relaxing the closure gate. The ring is clear of the
# loop band (collision-free) and inside the wall margin.
N_CIRCUIT_OBS     = 14                   # rich landmark ring (room normally has 15)
CIRCUIT_OBS_RMIN  = 0.60                 # ring inner radius (loop is at 0.40, margin >0.1)
CIRCUIT_OBS_RMAX  = 0.80                 # ring outer radius (outer edge <0.9 wall margin)
CIRCUIT_OBS_SMIN  = 0.06
CIRCUIT_OBS_SMAX  = 0.14

# =============================================================================
# Obstacle Generation
# =============================================================================
def generate_obstacles(key):
    keys = jax.random.split(key, N_OBSTACLES)
    def _one(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        cx = jax.random.uniform(k1, (), minval=OBS_MARGIN, maxval=ROOM_W - OBS_MARGIN)
        cy = jax.random.uniform(k2, (), minval=OBS_MARGIN, maxval=ROOM_H - OBS_MARGIN)
        w = jax.random.uniform(k3, (), minval=OBS_SIZE_MIN, maxval=OBS_SIZE_MAX)
        h = jax.random.uniform(k4, (), minval=OBS_SIZE_MIN, maxval=OBS_SIZE_MAX)
        return jnp.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return jax.vmap(_one)(keys)

def obstacles_to_segments(obstacles):
    room_segs = jnp.array([
        [[0, 0], [ROOM_W, 0]], [[ROOM_W, 0], [ROOM_W, ROOM_H]],
        [[ROOM_W, ROOM_H], [0, ROOM_H]], [[0, ROOM_H], [0, 0]],
    ], dtype=jnp.float32)
    def _rect(r):
        x0, y0, x1, y1 = r
        return jnp.array([
            [[x0, y0], [x1, y0]], [[x1, y0], [x1, y1]],
            [[x1, y1], [x0, y1]], [[x0, y1], [x0, y0]],
        ], dtype=jnp.float32)
    obs_segs = jax.vmap(_rect)(obstacles).reshape(-1, 2, 2)
    return jnp.concatenate([room_segs, obs_segs], axis=0)

# =============================================================================
# Collision Detection
# =============================================================================
def _point_rect_dist(px, py, rect):
    cx = jnp.clip(px, rect[0], rect[2])
    cy = jnp.clip(py, rect[1], rect[3])
    outside = jnp.sqrt((px - cx)**2 + (py - cy)**2)
    inside = -jnp.minimum(jnp.minimum(px - rect[0], rect[2] - px),
                           jnp.minimum(py - rect[1], rect[3] - py))
    inside_rect = (px >= rect[0]) & (px <= rect[2]) & (py >= rect[1]) & (py <= rect[3])
    return jnp.where(inside_rect, inside, outside)

def _min_clearance_to_obstacles(px, py, obstacles):
    dists = jax.vmap(lambda r: _point_rect_dist(px, py, r))(obstacles)
    # Use initial=inf to prevent zero-size array crashes
    return jnp.min(jnp.where(dists > 0, dists, jnp.inf), initial=jnp.inf)

def _wall_clearance(px, py):
    return jnp.minimum(jnp.minimum(px, py),
                       jnp.minimum(ROOM_W - px, ROOM_H - py))

def _is_clear(px, py, obstacles, margin=SAFE_MARGIN):
    obs_ok = _min_clearance_to_obstacles(px, py, obstacles) >= margin
    wall_ok = (px >= margin) & (px <= ROOM_W - margin) & \
              (py >= margin) & (py <= ROOM_H - margin)
    return obs_ok & wall_ok

def _trajectory_clear(positions, obstacles, margin=SAFE_MARGIN):
    checks = jax.vmap(lambda p: _is_clear(p[0], p[1], obstacles, margin))(positions)
    return jnp.all(checks)

# =============================================================================
# KINEMATIC "ROOMBA" EXPLORER
# =============================================================================
def _make_trajectory(key, time_steps, dt, obstacles=None):
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)
    
    pos = np.zeros((time_steps, 2), dtype=np.float32)
    hdg = np.zeros(time_steps, dtype=np.float32)
    vx = np.zeros(time_steps, dtype=np.float32)
    vy = np.zeros(time_steps, dtype=np.float32)
    omega = np.zeros(time_steps, dtype=np.float32)
    
    margin = SAFE_MARGIN + 0.05
    obs_np = np.array(obstacles) if obstacles is not None else np.zeros((0, 4))
    
    spawn_attempts = 0
    while spawn_attempts < 1000:
        sx = rng.uniform(margin, ROOM_W - margin)
        sy = rng.uniform(margin, ROOM_H - margin)
        hit = False
        for o in obs_np:
            if sx > o[0]-margin and sx < o[2]+margin and sy > o[1]-margin and sy < o[3]+margin:
                hit = True
                break
        if not hit:
            pos[0] = [sx, sy]
            break
        spawn_attempts += 1
        
    if spawn_attempts >= 1000:
        return None 

    hdg[0] = rng.uniform(0, 2 * np.pi)
    v_forward = 0.3   # reduced for 2m room (was 0.6 in 10m room)
    current_omega = rng.uniform(-0.5, 0.5)
    
    for t in range(1, time_steps):
        # 1. Update intended commands
        if rng.uniform() < 0.02: 
            current_omega = rng.uniform(OMEGA_RANGE[0], OMEGA_RANGE[1])
            
        v_slip = rng.uniform(VY_RANGE[0], VY_RANGE[1]) if rng.uniform() > 0.6 else 0.0
        v_forward = 0.6
            
        # 2. Calculate intended next pose using RK2 (Midpoint Integration)
        h_mid = hdg[t-1] + (current_omega * dt) / 2.0
        h_next = hdg[t-1] + current_omega * dt
        
        px_next = pos[t-1, 0] + (v_forward * np.cos(h_mid) - v_slip * np.sin(h_mid)) * dt
        py_next = pos[t-1, 1] + (v_forward * np.sin(h_mid) + v_slip * np.cos(h_mid)) * dt
        
        # 3. Collision check on the intended pose
        hit = False
        if px_next < margin or px_next > ROOM_W - margin or py_next < margin or py_next > ROOM_H - margin:
            hit = True
        else:
            for o in obs_np:
                if (px_next > o[0] - margin and px_next < o[2] + margin and 
                    py_next > o[1] - margin and py_next < o[3] + margin):
                    hit = True
                    break
                
        # 4. Resolve Kinematics & Sync Logging
        if hit:
            # Re-roll a rotation to bounce away from the wall
            if abs(current_omega) < 0.1:
                current_omega = OMEGA_RANGE[1] if rng.uniform() > 0.5 else OMEGA_RANGE[0]
            else:
                current_omega = OMEGA_RANGE[1] * np.sign(current_omega)
            
            # Recalculate heading using the NEW bounce rotation
            h_next = hdg[t-1] + current_omega * dt
            
            # Freeze translation for this frame
            px_next, py_next = pos[t-1, 0], pos[t-1, 1] 
            v_actual_fwd = 0.0
            v_actual_slip = 0.0
        else:
            v_actual_fwd = v_forward
            v_actual_slip = v_slip
            
        # 5. Commit strictly synchronized states
        pos[t] = [px_next, py_next]
        hdg[t] = h_next % (2 * np.pi)
        
        vx[t] = v_actual_fwd
        vy[t] = v_actual_slip
        omega[t] = current_omega
        
    return jnp.array(pos), jnp.array(hdg), jnp.array(vx), jnp.array(vy), jnp.array(omega)

# =============================================================================
# REVISIT-RICH CIRCUIT COURSE (Part B)
# =============================================================================
def generate_circuit_obstacles(key):
    """Distinctive landmark RING placed just OUTSIDE the (small) loop, where the
    forward-facing camera sees them head-on. The loop band stays collision-free while
    every loop position faces a unique arc of the ring -> sharp, confident place
    recognition on revisit (matching the head-on feature condition of the random course)."""
    keys = jax.random.split(key, N_CIRCUIT_OBS)
    cx0, cy0 = ROOM_W / 2.0, ROOM_H / 2.0
    def _one(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        r = jax.random.uniform(k1, (), minval=CIRCUIT_OBS_RMIN, maxval=CIRCUIT_OBS_RMAX)
        ang = jax.random.uniform(k2, (), minval=0.0, maxval=2.0 * np.pi)
        cx = cx0 + r * jnp.cos(ang)
        cy = cy0 + r * jnp.sin(ang)
        w = jax.random.uniform(k3, (), minval=CIRCUIT_OBS_SMIN, maxval=CIRCUIT_OBS_SMAX)
        h = jax.random.uniform(k4, (), minval=CIRCUIT_OBS_SMIN, maxval=CIRCUIT_OBS_SMAX)
        return jnp.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return jax.vmap(_one)(keys)

def generate_circuit_obstacles_aliased(key, kfold=2):
    """Perceptual-aliasing landmark ring (Tier-4 stress test): a k-fold rotationally
    symmetric ring. We draw base = N_CIRCUIT_OBS // kfold landmarks in one angular sector
    [0, 2*pi/kfold) and REPLICATE them kfold times around the ring, so two loop positions
    separated by 2*pi/kfold face near-identical landmark arcs. On the 0.4 m loop those
    positions are spatially distant (> the 0.30 m ground-truth-revisit radius) yet look
    alike -- exactly the spatially-distinct-but-similar-looking regime the appearance
    descriptor is not otherwise stressed on. Obstacle index i has within-sector index
    i % base, so the caller shares textures across replicas via sym_period = base."""
    base = N_CIRCUIT_OBS // kfold
    keys = jax.random.split(key, base)
    cx0, cy0 = ROOM_W / 2.0, ROOM_H / 2.0

    def _sector_landmark(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        r = jax.random.uniform(k1, (), minval=CIRCUIT_OBS_RMIN, maxval=CIRCUIT_OBS_RMAX)
        ang = jax.random.uniform(k2, (), minval=0.0, maxval=2.0 * np.pi / kfold)  # base sector
        w = jax.random.uniform(k3, (), minval=CIRCUIT_OBS_SMIN, maxval=CIRCUIT_OBS_SMAX)
        h = jax.random.uniform(k4, (), minval=CIRCUIT_OBS_SMIN, maxval=CIRCUIT_OBS_SMAX)
        return r, ang, w, h

    rs, angs, ws, hs = jax.vmap(_sector_landmark)(keys)
    rects = []
    for s in range(kfold):                       # sector (replica) index
        for j in range(base):                    # within-sector landmark index
            a = float(angs[j]) + s * (2.0 * np.pi / kfold)
            cx = cx0 + float(rs[j]) * np.cos(a)
            cy = cy0 + float(rs[j]) * np.sin(a)
            w, h = float(ws[j]), float(hs[j])
            rects.append(jnp.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    return jnp.stack(rects)                       # index = s*base + j, so i % base == j


def _make_circuit_trajectory(key, time_steps, dt):
    """Trace a circle repeatedly: the robot genuinely returns to every prior (x,y)
    once per lap (period ~ 2*pi*R/v). Per-seed variety in direction, start phase,
    radius and centre so Monte-Carlo seeds differ while all are loop-closing.
    Heading is tangent to the circle (direction of motion), so a revisit at the
    same (x,y) is also at the same heading -> consistent appearance for place cells."""
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)

    direction = 1.0 if rng.uniform() > 0.5 else -1.0      # CW / CCW
    phase0    = rng.uniform(0.0, 2.0 * np.pi)             # where on the loop we start
    radius    = CIRCUIT_RADIUS + rng.uniform(-0.05, 0.05) # mild radius jitter
    speed     = CIRCUIT_SPEED * (1.0 + rng.uniform(-0.08, 0.08))
    ccx       = ROOM_W / 2.0 + rng.uniform(-0.05, 0.05)   # mild centre jitter
    ccy       = ROOM_H / 2.0 + rng.uniform(-0.05, 0.05)

    omega_c = direction * speed / radius                  # constant angular rate (rad/s)
    t_arr   = np.arange(time_steps, dtype=np.float32) * dt
    ang     = phase0 + omega_c * t_arr                    # angular position on the circle
    pos     = np.stack([ccx + radius * np.cos(ang),
                        ccy + radius * np.sin(ang)], axis=1).astype(np.float32)
    hdg     = (ang + direction * (np.pi / 2.0)) % (2.0 * np.pi)  # tangent heading
    vx      = np.full(time_steps, speed, dtype=np.float32)        # forward speed along heading
    vy      = np.zeros(time_steps, dtype=np.float32)
    omega   = np.full(time_steps, omega_c, dtype=np.float32)
    vx[0] = 0.0; omega[0] = 0.0
    return (jnp.array(pos), jnp.array(hdg.astype(np.float32)),
            jnp.array(vx), jnp.array(vy), jnp.array(omega))

def _build_observations(positions, headings, segments, surface_textures, jax_obstacles, tex_t):
    """Event stream + ToF + intensities for a given path (factored for the circuit branch).

    SLAM_OBS_SLICE>0 processes the per-timestep vmap in memory-bounded slices (the
    (T x N_PIXELS x N_SEGMENTS) ray-cast intermediate scales with T and OOMs for long paths).
    Results are identical to the full vmap; each slice is materialized to numpy immediately so
    peak footprint is one slice, not the whole trajectory. Off (=0) -> original code path.
    """
    _slice = int(os.environ.get('SLAM_OBS_SLICE', '0'))
    T = int(positions.shape[0])
    if _slice > 0 and T > _slice:
        inten_parts, dist_parts, tof_parts = [], [], []
        for t0 in range(0, T, _slice):
            t1 = min(t0 + _slice, T)
            r = jax.vmap(
                lambda p, h: compute_pixel_readings(p, h, segments, surface_textures, jax_obstacles, tex_t)
            )(positions[t0:t1], headings[t0:t1])
            inten_parts.append(np.asarray(r[0])); dist_parts.append(np.asarray(r[1]))
            tof_parts.append(np.asarray(
                jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(positions[t0:t1], headings[t0:t1], segments)))
        intensities = jnp.asarray(np.concatenate(inten_parts, axis=0))
        distances = jnp.asarray(np.concatenate(dist_parts, axis=0))
        tof_dists = jnp.asarray(np.concatenate(tof_parts, axis=0))
    else:
        readings = jax.vmap(
            lambda p, h: compute_pixel_readings(p, h, segments, surface_textures, jax_obstacles, tex_t)
        )(positions, headings)
        intensities = readings[0]
        distances = readings[1]
        tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(positions, headings, segments)
    prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
    delta = intensities - prev
    events = jnp.where(delta > THRESHOLD, 1.0, jnp.where(delta < -THRESHOLD, -1.0, 0.0))
    events = events.at[0].set(0.0)
    return events, intensities, distances, tof_dists

def _make_explore_trajectory(key, time_steps, dt):
    """Room-filling, self-crossing REPEATED Lissajous path (view-rich / place-rich course).

    Unlike the single 0.40 m circuit circle (~5 distinct places), a Lissajous with amplitude ~0.65 m
    fills most of the 2 m room, so declustering gives many more distinct places. It closes smoothly,
    so REPEATING it revisits every point once per lap AT THE SAME HEADING (heading is tangent to the
    path) -> genuine same-appearance loop closures, exactly what a windowed low-level graph will lose
    once a lap predates the window. Frequencies (default 3:2) and lap length are env-tunable to dial
    the distinct-place count. Velocities are derived from the path so vx/vy/omega stay self-consistent
    (holonomic: the robot faces its direction of motion)."""
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)
    A   = float(os.environ.get('EXPLORE_AMP', '0.80'))
    fa  = float(os.environ.get('EXPLORE_FA', '3.0'))
    fb  = float(os.environ.get('EXPLORE_FB', '2.0'))
    speed = float(os.environ.get('EXPLORE_SPEED', '0.45'))       # constant hornet-scale speed (m/s)
    phx = rng.uniform(0, 2*np.pi); direction = 1.0 if rng.uniform() > 0.5 else -1.0
    cx = ROOM_W/2.0 + rng.uniform(-0.03, 0.03); cy = ROOM_H/2.0 + rng.uniform(-0.03, 0.03)

    # ARC-LENGTH parametrization: sample the closed Lissajous densely in its own parameter u, then
    # resample at CONSTANT arc-length steps (speed*dt) so the traversal speed is realistic and gentle
    # (only curvature drives turning). Without this, |dp/dt| ~ A*f*w reaches several m/s -> the
    # estimator drifts catastrophically and loop closure false-fires. One Lissajous period (u:0->2pi)
    # is one closed lap; resampling wraps around it so laps repeat -> same-heading revisits.
    U = np.linspace(0.0, 2*np.pi, 20000, endpoint=False); dU = U[1] - U[0]
    xp  =  A*fa*np.cos(fa*U + phx);      yp  =  A*fb*np.cos(fb*U)          # dp/dU
    xpp = -A*fa*fa*np.sin(fa*U + phx);   ypp = -A*fb*fb*np.sin(fb*U)       # d2p/dU2
    spU = np.hypot(xp, yp)                                                 # |dp/dU|
    kappa = np.abs(xp*ypp - yp*xpp) / (spU**3 + 1e-9)                      # path curvature
    ds = spU * dU                                                         # arc-length increments
    # Slow in tight turns so omega = v*kappa <= omega_max (physical hornet
    # turn-rate cap); constant `speed` on straights. Constant-speed traversal of the Lissajous cusps
    # otherwise demands omega ~ 18 rad/s, far above OMEGA_RANGE (+/-1.0). Time to cross each segment
    # is ds / v_local, so the fixed-dt resample naturally lingers in curves -> bounded per-step turn.
    omega_max = float(os.environ.get('EXPLORE_OMEGA_MAX', '0.9'))
    v_loc = np.minimum(speed, omega_max / (kappa + 1e-6))
    dt_seg = ds / (v_loc + 1e-9)
    T = np.concatenate([[0.0], np.cumsum(dt_seg)]); lap_time = T[-1]
    want_t = (np.arange(time_steps) * dt * direction) % lap_time          # cumulative time, wrapped
    idx_u = np.interp(want_t, T[:-1], U, period=lap_time)
    x = cx + A*np.sin(fa*idx_u + phx); y = cy + A*np.sin(fb*idx_u)
    pos = np.stack([x, y], 1).astype(np.float32)
    dx = np.gradient(x); dy = np.gradient(y)
    hdg = np.arctan2(dy, dx) % (2*np.pi)
    omega = np.gradient(np.unwrap(hdg), dt)
    vx = (np.hypot(dx, dy) / dt).astype(np.float32); vx[0] = 0.0          # actual (variable) speed
    omega = omega.astype(np.float32); omega[0] = 0.0
    return (jnp.array(pos), jnp.array(hdg.astype(np.float32)),
            jnp.array(vx), jnp.array(omega))


def generate_explore_obstacles(key, path_xy, n=None, clear=None):
    """Scatter many landmark obstacles across the room for VIEW diversity, keeping each at least
    `clear` m from the Lissajous path so the prescribed trajectory stays sensible. More obstacles +
    varied positions => each place sees a distinct obstacle configuration => more distinct VIEWS in
    the same 2 m room (the user's point: view count, not arena size)."""
    n = int(os.environ.get('EXPLORE_NOBS', '44')) if n is None else n
    clear = float(os.environ.get('EXPLORE_CLEAR', '0.09')) if clear is None else clear
    rng = np.random.RandomState(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    P = np.asarray(path_xy); obs = []
    tries = 0
    while len(obs) < n and tries < 4000:
        tries += 1
        cxy = rng.uniform(OBS_MARGIN, ROOM_W - OBS_MARGIN, 2)
        if np.min(np.hypot(P[:, 0] - cxy[0], P[:, 1] - cxy[1])) < clear:
            continue
        if obs and np.min([np.hypot(cxy[0]-o[0], cxy[1]-o[1]) for o in obs]) < 0.10:
            continue
        sz = rng.uniform(OBS_SIZE_MIN, OBS_SIZE_MAX)
        # CORNER format [x0,y0,x1,y1] to match the codebase (generate_obstacles / _point_rect_dist /
        # _rect all read corners). Previously [cx,cy,sz,sz] -> parsed as corners -> giant malformed
        # rectangles from centre to near-origin, corrupting every DVS/ToF view.
        obs.append([cxy[0]-sz/2, cxy[1]-sz/2, cxy[0]+sz/2, cxy[1]+sz/2])
    return jnp.array(np.array(obs, dtype=np.float32))


def _make_l2probe_trajectory(key, time_steps, dt):
    """L2-PROBE course (two-scale revisit schedule) for the bounded-L1 / weight-L2 examination.

    Geometry: a FIGURE-EIGHT of two externally tangent circles -- a small anchor loop A (traversed
    CCW) and a larger excursion loop B (traversed CW) -- meeting at the tangency point T. Traversing
    A counter-clockwise and B clockwise makes the tangent direction CONTINUOUS at T, so the whole
    schedule is C1 with NO connector segments and curvature bounded by 1/rA and 1/rB (omega = v*kappa
    stays well inside OMEGA_RANGE). An earlier Hermite-connector design was abandoned: returning from
    B to A forced a U-turn at the join (kappa ~ 36/m, omega ~ 5 rad/s, 5x the physical cap).

    Schedule:  A x n_alaps  ->  B x 1  ->  A x 1
      * A x n_alaps : SHORT-gap appearance revisits (gap ~= |A| places) at the SAME heading, so a
                      bounded rolling L1 window still holds them -> it can close the loop, relax, and
                      RE-INSTAR the L2 weights while the place is still inside the window.
      * B x 1       : one pass of the larger loop, creating many NEW places (so |A| leaves the window)
                      and accumulating drift.
      * A (return)  : revisit to A at gap ~= |A|+|B| >> cap -> the bounded L1 has EVICTED those places
                      and structurally cannot close them; only the L2 weight memory can.
    Every lap of A is traversed in the same rotational sense, so each point of A is seen at the same
    heading on all laps (genuine same-appearance revisits) regardless of where the lap starts."""
    # This generator previously IGNORED `key` entirely, so every "seed" produced a
    # BIT-IDENTICAL trajectory (verified: PRNGKey(42) vs PRNGKey(999999) -> np.array_equal on
    # positions/headings/vx/omega). All multi-seed statistics were therefore three sensor
    # realizations of ONE trajectory, i.e. effective trajectory-level n = 1. Derive an rng from the
    # key and jitter the schedule geometry, conservatively enough to stay inside the wall margin.
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    _rng = np.random.RandomState(key_np)
    rA = float(os.environ.get('L2P_RA', '0.32')) * (1.0 + _rng.uniform(-0.08, 0.08))
    rB = float(os.environ.get('L2P_RB', '0.50')) * (1.0 + _rng.uniform(-0.08, 0.08))
    n_alaps = int(os.environ.get('L2P_ALAPS', '2'))       # A laps BEFORE the excursion
    n_blaps = int(os.environ.get('L2P_BLAPS', '1'))       # excursion passes
    # Jitter ONLY parameters that PRESERVE the tangency (cB and thA0/thB0 are derived from
    # cA and u, so the C1 join at T survives). A lap-phase offset was tried and REVERTED: it
    # moves both arcs off T, breaking continuity -> |omega| exploded to ~60 rad/s.
    # NB rA/rB are jittered too: jittering only cA and the axis is a RIGID TRANSFORM of the
    # whole figure-eight (identical L and curvature every seed) -- shape must vary for real
    # trajectory diversity. rA also sets the tightest turn: omega_max = v/rA, and at the
    # DEPLOYED STEPS=1500 (v~0.29 m/s) rA=0.26 gave omega 1.10, over the 1.0 cap.
    cAx = 0.58 + _rng.uniform(-0.02, 0.02)                 # per-seed geometry jitter (B2)
    cAy = 0.58 + _rng.uniform(-0.02, 0.02)
    _ang = np.pi / 4.0 + _rng.uniform(-0.08, 0.08)         # cA -> cB axis, jittered
    u = np.array([np.cos(_ang), np.sin(_ang)])
    cB = np.array([cAx, cAy]) + (rA + rB) * u             # externally tangent
    thA0 = np.arctan2(u[1], u[0])                         # angle of T on circle A
    thB0 = np.arctan2(-u[1], -u[0])                       # angle of T on circle B

    nA = int(os.environ.get('L2P_NSAMP_A', '4000'))
    nB = int(os.environ.get('L2P_NSAMP_B', '9000'))

    def arcA(laps):                                        # CCW from T
        t = thA0 + np.linspace(0.0, 2*np.pi*laps, nA*laps, endpoint=False)
        return np.stack([cAx + rA*np.cos(t), cAy + rA*np.sin(t)], 1)

    def arcB(laps):                                        # CW from T
        t = thB0 - np.linspace(0.0, 2*np.pi*laps, nB*laps, endpoint=False)
        return np.stack([cB[0] + rB*np.cos(t), cB[1] + rB*np.sin(t)], 1)

    # ALTERNATING schedule: A x n_alaps, then (B, A) x n_alts.
    # Every arc both starts AND ends at the tangency point T, so arbitrary A/B concatenations stay C1
    # continuous for free. Place indices stop growing once every place has been seen once, so
    # lo = n - cap FREEZES and loop A's places (the low indices) become PERMANENTLY evicted --
    # therefore EVERY return to A yields |A| evicted-revisit opportunities. This makes evicted
    # revisits STRUCTURALLY FREQUENT rather than incidental: a testbed deliberately built to exercise
    # the L2 mechanism, not a claim about how often this occurs in the wild.
    n_alts = int(os.environ.get('L2P_ALTS', '3'))
    segs = [arcA(n_alaps)]
    for _ in range(n_alts):
        segs.append(arcB(n_blaps)); segs.append(arcA(1))
    P = np.concatenate(segs, 0)

    # CONSTANT-ARC-LENGTH resample. Curvature-limited pacing is deliberately NOT used: it sets only
    # RELATIVE speed, so against a fixed step budget it produces an extreme fast/slow ratio that
    # under-samples the straights and manufactures spurious omega spikes. At constant speed
    # v = L/(steps*dt) the turn rate is omega = v*kappa, a pure geometry property, bounded here by
    # construction. Heading is taken from the SOURCE tangent, since differencing the resampled points
    # is dominated by interpolation noise whenever the step budget out-resolves the source polyline.
    d1 = np.gradient(P, axis=0)
    sp = np.hypot(d1[:, 0], d1[:, 1]) + 1e-12
    s = np.concatenate([[0.0], np.cumsum(sp)])
    want = np.linspace(0.0, s[-1]*(1 - 1e-9), time_steps)
    idx = np.interp(want, s[:-1], np.arange(len(P)))
    x = np.interp(idx, np.arange(len(P)), P[:, 0])
    y = np.interp(idx, np.arange(len(P)), P[:, 1])
    th_src = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))
    hd = np.interp(idx, np.arange(len(P)), th_src)
    positions = jnp.asarray(np.stack([x, y], 1))
    headings = jnp.asarray(hd)
    vx = jnp.asarray(np.hypot(np.gradient(x), np.gradient(y)) / dt)   # holonomic: faces travel dir
    omega = jnp.asarray(np.gradient(hd) / dt)
    return positions, headings, vx, omega


def _explore_room_dataset(rng, n_samples, obstacles, time_steps, dt, traj_fn=None):
    """Mirror of _circuit_room_dataset for the view-rich room-filling explore course.
    traj_fn selects the path generator (explore Lissajous by default; l2probe two-scale schedule)."""
    traj_fn = traj_fn or _make_explore_trajectory
    import hashlib
    ev_l, lab_l, tof_l, pos_l, hdg_l, int_l = [], [], [], [], [], []
    for _ in range(n_samples):
        k_traj = jax.random.PRNGKey(rng.randint(0, 2**31))
        positions, headings, vx, omega = traj_fn(k_traj, time_steps, dt)
        vy = jnp.zeros_like(vx)
        obs = obstacles
        if obs is None:
            obs = generate_explore_obstacles(jax.random.PRNGKey(rng.randint(0, 2**31)), np.array(positions))
        segments = obstacles_to_segments(obs)
        obs_np = np.array(obs)
        room_seed = int(hashlib.md5(obs_np.tobytes()).hexdigest(), 16) % (2**31 - 1)
        surface_textures = _generate_surface_textures(obs_np, room_seed)
        tex_t = _precompute_barcode_tensors(surface_textures, obs_np)
        events, intensities, distances, tof_dists = _build_observations(
            positions, headings, segments, surface_textures, jnp.asarray(obs), tex_t)
        min_clear_arr = jnp.min(distances, axis=1)
        clearance_norm = jnp.tanh(min_clear_arr / 2.0)
        labels = jnp.stack([vx / abs(VX_RANGE[1]), vy / abs(VY_RANGE[1]),
                            omega / abs(OMEGA_RANGE[1]), clearance_norm], axis=1)
        ev_l.append(events); lab_l.append(labels); tof_l.append(tof_dists)
        pos_l.append(positions); hdg_l.append(headings); int_l.append(intensities)
    return (jnp.stack(ev_l), jnp.stack(lab_l), jnp.stack(tof_l),
            jnp.stack(pos_l), jnp.stack(hdg_l), obs, segments, jnp.stack(int_l))


def _circuit_room_dataset(rng, n_samples, obstacles, time_steps, dt, sym_period=None):
    """Build a fixed-room dataset of revisit-rich circuit trajectories (no rejection
    loop: the circle is collision-free by construction since obstacles sit inside it).
    sym_period (perceptual-aliasing course): share textures across replicated landmarks."""
    import hashlib
    if obstacles is None:
        k_obs = jax.random.PRNGKey(rng.randint(0, 2**31))
        obstacles = generate_circuit_obstacles(k_obs)
    segments = obstacles_to_segments(obstacles)
    obstacles_np = np.array(obstacles) if hasattr(obstacles, 'dtype') else obstacles
    room_seed = int(hashlib.md5(obstacles_np.tobytes()).hexdigest(), 16) % (2**31 - 1)
    surface_textures = _generate_surface_textures(obstacles_np, room_seed, sym_period=sym_period)
    tex_t = _precompute_barcode_tensors(surface_textures, obstacles_np)
    jax_obstacles = jnp.asarray(obstacles)

    ev_l, lab_l, tof_l, pos_l, hdg_l, int_l = [], [], [], [], [], []
    for _ in range(n_samples):
        k_traj = jax.random.PRNGKey(rng.randint(0, 2**31))
        positions, headings, vx, vy, omega = _make_circuit_trajectory(k_traj, time_steps, dt)
        events, intensities, distances, tof_dists = _build_observations(
            positions, headings, segments, surface_textures, jax_obstacles, tex_t)
        min_clear_arr = jnp.min(distances, axis=1)
        clearance_norm = jnp.tanh(min_clear_arr / 2.0)
        labels = jnp.stack([
            vx.astype(jnp.float32)    / abs(VX_RANGE[1]),
            vy.astype(jnp.float32)    / abs(VY_RANGE[1]),
            omega.astype(jnp.float32) / abs(OMEGA_RANGE[1]),
            clearance_norm,
        ], axis=1)
        ev_l.append(events); lab_l.append(labels); tof_l.append(tof_dists)
        pos_l.append(positions); hdg_l.append(headings); int_l.append(intensities)

    return (jnp.stack(ev_l), jnp.stack(lab_l), jnp.stack(tof_l),
            jnp.stack(pos_l), jnp.stack(hdg_l), obstacles, segments, jnp.stack(int_l))

# =============================================================================
# Ray–Segment Intersection
# =============================================================================
def cast_rays(origins, directions, segments):
    A = segments[:, 0, :]
    B = segments[:, 1, :]
    E = B - A
    D = directions[:, None, :]
    diff = A[None, :, :] - origins[:, None, :]
    det = (D[:, :, 0] * E[None, :, 1] - D[:, :, 1] * E[None, :, 0])
    safe = jnp.where(jnp.abs(det) > 1e-10, det, 1.0)
    t = (diff[:, :, 0] * E[None, :, 1] - diff[:, :, 1] * E[None, :, 0]) / safe
    s = (diff[:, :, 0] * D[:, :, 1] - diff[:, :, 1] * D[:, :, 0]) / safe
    valid = (jnp.abs(det) > 1e-10) & (t > 0.01) & (s >= 0) & (s <= 1)
    dists = jnp.where(valid, t, 1e6)
    hit_pts = origins[:, None, :] + t[:, :, None] * directions[:, None, :]
    return dists, hit_pts

# =============================================================================
# Pixel Intensity (no dimming)
# =============================================================================
def _barcode_texture(barcode_key, local_coords):
    rng = np.random.RandomState(int(barcode_key) & 0xFFFFFFFF)
    n_stripes = rng.randint(3, 8) 
    boundaries = sorted([0.0] + list(rng.uniform(0.0, 1.0, n_stripes - 1)) + [1.0])
    boundaries = np.array(boundaries, dtype=np.float32)
    brightness = rng.uniform(0.15, 0.95, n_stripes).astype(np.float32)
    stripe_idx = np.searchsorted(boundaries[1:], local_coords)
    stripe_idx = np.clip(stripe_idx, 0, n_stripes - 1)
    base = brightness[stripe_idx] 

    for freq, amp in zip(TEX_FREQS, TEX_AMPS):
        phase_a = rng.uniform(0, 2 * np.pi)
        along_mod = np.cos(2 * np.pi * freq * local_coords + phase_a)
        base = base * (1.0 + amp * 0.6 * along_mod)

    pattern = np.clip(base, 0.05, 1.5)
    return pattern.astype(np.float32)

def _generate_surface_textures(obstacles, room_seed, sym_period=None):
    """Per-segment barcode textures. When sym_period is set (perceptual-aliasing course),
    obstacle textures are seeded by the WITHIN-SECTOR index (obs_idx % sym_period) so that
    rotationally-replicated landmarks are texturally identical -- genuine visual aliasing,
    not merely symmetric geometry."""
    rng = np.random.RandomState(int(room_seed) & 0xFFFFFFFF)
    textures = {}

    wall_seeds = [rng.randint(0, 2**31) for _ in range(4)]
    wall_coords = np.linspace(0, 1, BARCODE_RESOLUTION)
    for i, seed in enumerate(wall_seeds):
        textures[i] = _barcode_texture(seed, wall_coords)

    n_obstacles = obstacles.shape[0]
    coords = np.linspace(0.0, 1.0, BARCODE_RESOLUTION)
    for obs_idx in range(n_obstacles):
        tex_idx = obs_idx if sym_period is None else (obs_idx % sym_period)
        for side in range(4):
            seg_idx = 4 + obs_idx * 4 + side
            if sym_period is None:
                seed = int(rng.randint(0, 2**31))
            else:
                # deterministic per (within-sector landmark, side) so replicas share texture
                seed = int(np.random.RandomState(
                    (int(room_seed) + 1315423911 * tex_idx + 2654435761 * side) & 0xFFFFFFFF
                ).randint(0, 2**31))
            textures[seg_idx] = _barcode_texture(seed, coords)

    return textures

def compute_tof_distance(robot_pos, robot_heading, segments, include_back=False):
    if include_back:
        angles = robot_heading + jnp.array([-jnp.pi/4, 0.0, jnp.pi/4, jnp.pi])
        n_rays = 4
    else:
        angles = robot_heading + jnp.array([-jnp.pi/4, 0.0, jnp.pi/4])
        n_rays = 3
    origins = jnp.broadcast_to(robot_pos, (n_rays, 2))
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

    dists, _ = cast_rays(origins, directions, segments) 
    min_dists = jnp.min(dists, axis=-1) 

    max_range = 2.83  # max diagonal of 2m×2m room = √(2²+2²)
    tof_dists = jnp.clip(min_dists, 0.0, max_range)

    return tof_dists

# Removed the convoluted 'stripe_edges' system entirely.
def _precompute_barcode_tensors(surface_textures, obstacles):
    """Pre-compute texture tensors for fast vectorized lookup."""
    seg_ids = sorted(surface_textures.keys())
    n_surf = max(seg_ids) + 1 if seg_ids else 0

    tex_rows = []
    for seg_id in range(n_surf):
        if seg_id in surface_textures:
            tex = np.array(surface_textures[seg_id], dtype=np.float32)
            tex_rows.append(tex)
        else:
            tex_rows.append(np.zeros(BARCODE_RESOLUTION, dtype=np.float32))

    return jnp.stack([jnp.array(t) for t in tex_rows]) if tex_rows else jnp.zeros((0, BARCODE_RESOLUTION))


def _sample_barcode_fast(min_idx, nearest, min_dist, obstacles, tex_tensor):
    n_pix = min_idx.shape[0]
    hx = nearest[:, 0]
    hy = nearest[:, 1]

    is_wall = min_idx < 4
    t_wall = jnp.stack([
        hx / ROOM_W,   
        hy / ROOM_H,   
        hx / ROOM_W,   
        hy / ROOM_H,   
    ], axis=1) 

    obs_seg_ids = min_idx - 4 
    obs_idx = obs_seg_ids // 4   
    side = obs_seg_ids % 4      

    # Use mode='clip' to prevent out-of-bounds JAX crashes
    max_obs_idx = max(0, obstacles.shape[0] - 1)
    safe_obs_idx = jnp.clip(obs_idx, 0, max_obs_idx)
    
    _obs_x0 = jnp.take(obstacles[:, 0], safe_obs_idx, mode='clip') if obstacles.shape[0] > 0 else jnp.zeros(n_pix)
    _obs_y0 = jnp.take(obstacles[:, 1], safe_obs_idx, mode='clip') if obstacles.shape[0] > 0 else jnp.zeros(n_pix)
    _obs_x1 = jnp.take(obstacles[:, 2], safe_obs_idx, mode='clip') if obstacles.shape[0] > 0 else jnp.ones(n_pix)
    _obs_y1 = jnp.take(obstacles[:, 3], safe_obs_idx, mode='clip') if obstacles.shape[0] > 0 else jnp.ones(n_pix)
    
    dx = _obs_x1 - _obs_x0 + 1e-8
    dy = _obs_y1 - _obs_y0 + 1e-8

    t_obs = jnp.where(
        side == 0, (hx - _obs_x0) / dx,
        jnp.where(
            side == 1, (hy - _obs_y0) / dy,
            jnp.where(
                side == 2, (hx - _obs_x0) / dx,
                (hy - _obs_y0) / dy
            )
        )
    ) 

    # Clip min_idx to [0, 3] to prevent out-of-bounds lookup on t_wall (which only has 4 columns)
    safe_min_idx = jnp.clip(min_idx, 0, 3)
    t_wall_selected = jnp.take_along_axis(t_wall, safe_min_idx[:, None], axis=1)[:, 0]
    t_all = jnp.where(is_wall, t_wall_selected, t_obs) 

    tex_for_pix = jnp.take(tex_tensor, min_idx, axis=0, mode='clip') 

    # Map coordinate natively (0.0 - 1.0) directly to the 512-dim pixel array
    tex_idx = jnp.clip(jnp.floor(t_all * BARCODE_RESOLUTION).astype(jnp.int32), 0, BARCODE_RESOLUTION - 1) 
    
    batch_idx = jnp.arange(n_pix) 
    intensity = tex_for_pix[batch_idx, tex_idx]

    t_depth = min_dist / 2.83   # normalise by room diagonal (was 14.14=√(10²+10²); now 2.83=√(2²+2²))
    
    for freq, amp in zip(TEX_FREQS, TEX_AMPS):
        phase = jnp.pi * freq 
        depth_mod = jnp.cos(2.0 * jnp.pi * freq * 6.0 * t_depth + phase)
        intensity = intensity * (1.0 + 0.6 * amp * depth_mod)

    return jnp.clip(intensity, 0.05, 1.5)


def _sample_barcode_textures(min_idx, nearest, min_dist, surface_textures, obstacles=None, tex_tensor=None):
    if tex_tensor is not None:
        return _sample_barcode_fast(min_idx, nearest, min_dist, obstacles, tex_tensor)
    return jnp.zeros(N_PIXELS, dtype=jnp.float32) 


def compute_pixel_readings(robot_pos, robot_heading, segments, surface_textures=None, obstacles=None, tex_tensor=None):
    fov_rad = jnp.radians(FOV_DEG)
    angles = robot_heading + jnp.linspace(-fov_rad/2, fov_rad/2, N_PIXELS)
    origins = jnp.broadcast_to(robot_pos, (N_PIXELS, 2))
    dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    
    dists, hit_pts = cast_rays(origins, dirs, segments)
    min_idx = jnp.argmin(dists, axis=-1)
    min_dist = jnp.min(dists, axis=-1)
    nearest = hit_pts[jnp.arange(N_PIXELS), min_idx]
    hit_type = (min_idx >= 4).astype(jnp.float32)

    intensities = _sample_barcode_textures(min_idx, nearest, min_dist, surface_textures, obstacles, tex_tensor)

    return intensities, min_dist, hit_type, nearest


# =============================================================================
# Generate Safe Sample
# =============================================================================
def generate_sample(key, time_steps=TIME_STEPS, dt=DT):
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)

    for room_attempt in range(MAX_ROOM_ATTEMPTS):
        k_obs = jax.random.PRNGKey(rng.randint(0, 2**31))
        obstacles = generate_obstacles(k_obs)
        segments = obstacles_to_segments(obstacles)

        for traj_attempt in range(MAX_TRAJ_ATTEMPTS):
            k_traj = jax.random.PRNGKey(rng.randint(0, 2**31))
            
            traj_result = _make_trajectory(k_traj, time_steps, dt, obstacles)
            if traj_result is None: continue 
                
            positions, headings, vx, vy, omega = traj_result

            spawn_ok = bool(jnp.all(_is_clear(positions[0, 0], positions[0, 1], obstacles)))
            traj_ok = bool(jnp.all(_trajectory_clear(positions, obstacles)))

            if spawn_ok and traj_ok:
                room_seed = rng.randint(0, 2**31)
                surface_textures = _generate_surface_textures(np.array(obstacles), room_seed)
                tex_t = _precompute_barcode_tensors(surface_textures, np.array(obstacles))

                jax_obstacles = jnp.asarray(obstacles)
                
                readings = jax.vmap(
                    lambda p, h: compute_pixel_readings(
                        p, h, segments, surface_textures, jax_obstacles, tex_t
                    )
                )(positions, headings)
                intensities = readings[0]
                distances = readings[1]

                tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(positions, headings, segments)

                prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
                delta = intensities - prev
                events = jnp.where(delta > THRESHOLD, 1.0,
                          jnp.where(delta < -THRESHOLD, -1.0, 0.0))
                events = events.at[0].set(0.0)

                min_clear_arr = jnp.min(distances, axis=1)          

                if vx.ndim == 0:
                    vx_arr = jnp.broadcast_to(vx, (time_steps,)).astype(jnp.float32)
                    vy_arr = jnp.broadcast_to(vy, (time_steps,)).astype(jnp.float32)
                    omega_arr = jnp.broadcast_to(omega, (time_steps,)).astype(jnp.float32)
                else:
                    vx_arr = vx.astype(jnp.float32)
                    vy_arr = vy.astype(jnp.float32)
                    omega_arr = omega.astype(jnp.float32)

                clearance_norm = jnp.tanh(min_clear_arr / 0.4)          # 2m room: half of 0.8m mid-room clearance (was /2.0 in 10m room)
                labels = jnp.stack([
                    vx_arr     / abs(VX_RANGE[1]),
                    vy_arr     / abs(VY_RANGE[1]),
                    omega_arr  / abs(OMEGA_RANGE[1]),
                    clearance_norm,
                ], axis=1)                                              

                info = {
                    'obstacles': obstacles,
                    'segments': segments,
                    'positions': positions,
                    'headings': headings,
                    'vx': vx_arr, 'vy': vy_arr, 'omega': omega_arr,   
                    'vx_mean': float(jnp.mean(vx_arr)),               
                    'vy_mean': float(jnp.mean(vy_arr)),
                    'omega_mean': float(jnp.mean(omega_arr)),
                    'intensities': intensities,
                    'distances': distances,
                    'tof': tof_dists,
                    'room_attempts': room_attempt + 1,
                    'traj_attempts': traj_attempt + 1,
                }
                return events, labels, info

    empty_obs = jnp.zeros((0, 4), dtype=jnp.float32)
    empty_segs = jnp.array([
        [[0, 0], [ROOM_W, 0]], [[ROOM_W, 0], [ROOM_W, ROOM_H]],
        [[ROOM_W, ROOM_H], [0, ROOM_H]], [[0, ROOM_H], [0, 0]],
    ], dtype=jnp.float32)

    k_fb = jax.random.PRNGKey(rng.randint(0, 2**31))
    positions, headings, vx, vy, omega = _make_trajectory(k_fb, time_steps, dt)

    fb_room_seed = rng.randint(0, 2**31)
    fb_textures = _generate_surface_textures(np.zeros((0, 4), dtype=np.float32), fb_room_seed)
    fb_tex_t = _precompute_barcode_tensors(fb_textures, np.zeros((0, 4), dtype=np.float32))

    readings = jax.vmap(
        lambda p, h: compute_pixel_readings(
            p, h, empty_segs, fb_textures, empty_obs, fb_tex_t
        )
    )(positions, headings)
    
    intensities = readings[0]
    distances = readings[1]

    tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(positions, headings, empty_segs)

    prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
    delta = intensities - prev
    events = jnp.where(delta > THRESHOLD, 1.0, jnp.where(delta < -THRESHOLD, -1.0, 0.0))
    events = events.at[0].set(0.0)

    min_clear_arr = jnp.min(distances, axis=1)
    vx_arr = jnp.broadcast_to(vx, (time_steps,)).astype(jnp.float32)
    vy_arr = jnp.broadcast_to(vy, (time_steps,)).astype(jnp.float32)
    omega_arr = jnp.broadcast_to(omega, (time_steps,)).astype(jnp.float32)
    clearance_norm = jnp.tanh(min_clear_arr / 2.0)
    labels = jnp.stack([
        vx_arr     / abs(VX_RANGE[1]),
        vy_arr     / abs(VY_RANGE[1]),
        omega_arr  / abs(OMEGA_RANGE[1]),
        clearance_norm,
    ], axis=1)

    info = {
        'obstacles': empty_obs,
        'segments': empty_segs,
        'positions': positions,
        'headings': headings,
        'vx': vx_arr, 'vy': vy_arr, 'omega': omega_arr,
        'vx_mean': float(jnp.mean(vx_arr)),
        'vy_mean': float(jnp.mean(vy_arr)),
        'omega_mean': float(jnp.mean(omega_arr)),
        'intensities': intensities,
        'distances': distances,
        'tof': tof_dists,
        'room_attempts': MAX_ROOM_ATTEMPTS,
        'traj_attempts': MAX_TRAJ_ATTEMPTS,
        'fallback': True,
    }
    return events, labels, info


def generate_fixed_room_dataset(key, n_samples, obstacles=None, time_steps=TIME_STEPS, dt=DT,
                                course_type='random'):
    key_np = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.RandomState(key_np)

    if course_type == 'circuit':
        # Part B: revisit-rich closed-loop course (collision-free by construction).
        return _circuit_room_dataset(rng, n_samples, obstacles, time_steps, dt)

    if course_type == 'explore':
        # View-rich, place-rich room-filling repeated-Lissajous course (bounded-window / two-layer
        # study). Many distinct places + genuine same-heading revisits once per lap.
        return _explore_room_dataset(rng, n_samples, obstacles, time_steps, dt)

    if course_type == 'l2probe':
        # Two-scale revisit schedule for the bounded-L1 / weight-L2 examination (see
        # _make_l2probe_trajectory): short-gap revisits L1 can close, then a long excursion
        # that evicts them, then a return only the L2 weight memory can serve.
        return _explore_room_dataset(rng, n_samples, obstacles, time_steps, dt,
                                     traj_fn=_make_l2probe_trajectory)

    if course_type == 'circuit_alias':
        # Tier-4 stress test: revisit-rich loop with a k-fold symmetric (perceptually
        # aliasing) landmark ring, so spatially-distant places look alike.
        kfold = int(os.environ.get('SLAM_ALIAS_KFOLD', '2'))
        sym_period = N_CIRCUIT_OBS // kfold
        if obstacles is None:
            k_obs = jax.random.PRNGKey(rng.randint(0, 2**31))
            obstacles = generate_circuit_obstacles_aliased(k_obs, kfold=kfold)
        return _circuit_room_dataset(rng, n_samples, obstacles, time_steps, dt, sym_period=sym_period)

    if obstacles is None:
        k_obs = jax.random.PRNGKey(rng.randint(0, 2**31))
        obstacles = generate_obstacles(k_obs)
        segments = obstacles_to_segments(obstacles)
        room_seed = rng.randint(0, 2**31)
    else:
        segments = obstacles_to_segments(obstacles)
        # # Hash the obstacles geometry to create a completely deterministic room texture seed!
        import hashlib
        obs_bytes = np.array(obstacles).tobytes()
        room_seed = int(hashlib.md5(obs_bytes).hexdigest(), 16) % (2**31 - 1)

    obstacles_np = np.array(obstacles) if hasattr(obstacles, 'dtype') else obstacles
    surface_textures = _generate_surface_textures(obstacles_np, room_seed)
    tex_t = _precompute_barcode_tensors(surface_textures, obstacles_np)

    events_list = []
    labels_list = []
    tof_list = []
    positions_list = []
    headings_list = []
    intensities_list = []
    total_attempts = 0
    max_attempts = n_samples * MAX_TRAJ_ATTEMPTS

    jax_obstacles = jnp.asarray(obstacles)

    while len(events_list) < n_samples and total_attempts < max_attempts:
        k_traj = jax.random.PRNGKey(rng.randint(0, 2**31))
        traj_result = _make_trajectory(k_traj, time_steps, dt, obstacles)
        
        if traj_result is None:
            total_attempts += 1
            continue
            
        positions, headings, vx, vy, omega = traj_result

        spawn_ok = bool(jnp.all(_is_clear(positions[0, 0], positions[0, 1], obstacles)))
        traj_ok = bool(jnp.all(_trajectory_clear(positions, obstacles)))
        total_attempts += 1

        if spawn_ok and traj_ok:
            readings = jax.vmap(
                lambda p, h: compute_pixel_readings(
                    p, h, segments, surface_textures, jax_obstacles, tex_t
                )
            )(positions, headings)
            
            intensities = readings[0]
            distances = readings[1]
            intensities_list.append(intensities)

            tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(positions, headings, segments)

            prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
            delta = intensities - prev
            events = jnp.where(delta > THRESHOLD, 1.0, jnp.where(delta < -THRESHOLD, -1.0, 0.0))
            events = events.at[0].set(0.0)

            min_clear_arr = jnp.min(distances, axis=1)  
            if vx.ndim == 0:
                vx_arr = jnp.broadcast_to(vx, (time_steps,)).astype(jnp.float32)
                vy_arr = jnp.broadcast_to(vy, (time_steps,)).astype(jnp.float32)
                omega_arr = jnp.broadcast_to(omega, (time_steps,)).astype(jnp.float32)
            else:
                vx_arr = vx.astype(jnp.float32)
                vy_arr = vy.astype(jnp.float32)
                omega_arr = omega.astype(jnp.float32)
            clearance_norm = jnp.tanh(min_clear_arr / 2.0)
            labels = jnp.stack([
                vx_arr     / abs(VX_RANGE[1]),
                vy_arr     / abs(VY_RANGE[1]),
                omega_arr  / abs(OMEGA_RANGE[1]),
                clearance_norm,
            ], axis=1)  

            events_list.append(events)
            labels_list.append(labels)
            tof_list.append(tof_dists)
            positions_list.append(positions)
            headings_list.append(headings)

    if len(events_list) == 0:
        # Fallback: ignore traj_ok (allow bounces) if we can't find a perfectly clear trajectory
        for attempt in range(max_attempts):
            k_traj = jax.random.PRNGKey(rng.randint(0, 2**31))
            traj_result = _make_trajectory(k_traj, time_steps, dt, obstacles)
            if traj_result is None: continue
            positions, headings, vx, vy, omega = traj_result
            spawn_ok = bool(jnp.all(_is_clear(positions[0, 0], positions[0, 1], obstacles)))
            if spawn_ok or attempt == max_attempts - 1:
                readings = jax.vmap(
                    lambda p, h: compute_pixel_readings(
                        p, h, segments, surface_textures, jax_obstacles, tex_t
                    )
                )(positions, headings)
                intensities = readings[0]
                distances = readings[1]
                intensities_list.append(intensities)
                tof_dists = jax.vmap(compute_tof_distance, in_axes=(0, 0, None))(positions, headings, segments)
                prev = jnp.concatenate([intensities[:1], intensities[:-1]], axis=0)
                delta = intensities - prev
                events = jnp.where(delta > THRESHOLD, 1.0, jnp.where(delta < -THRESHOLD, -1.0, 0.0))
                events = events.at[0].set(0.0)
                min_clear_arr = jnp.min(distances, axis=1)
                vx_arr = vx.astype(jnp.float32)
                vy_arr = vy.astype(jnp.float32)
                omega_arr = omega.astype(jnp.float32)
                clearance_norm = jnp.tanh(min_clear_arr / 2.0)
                labels = jnp.stack([
                    vx_arr     / abs(VX_RANGE[1]),
                    vy_arr     / abs(VY_RANGE[1]),
                    omega_arr  / abs(OMEGA_RANGE[1]),
                    clearance_norm,
                ], axis=1)
                events_list.append(events)
                labels_list.append(labels)
                tof_list.append(tof_dists)
                positions_list.append(positions)
                headings_list.append(headings)
                break

    return (jnp.stack(events_list), jnp.stack(labels_list), jnp.stack(tof_list),
            jnp.stack(positions_list), jnp.stack(headings_list), obstacles, segments,
            jnp.stack(intensities_list))


def generate_batch(key, batch_size=BATCH_SIZE, time_steps=TIME_STEPS, dt=DT):
    keys = jax.random.split(key, batch_size)
    events_list, labels_list, info_list = [], [], []
    for i in range(batch_size):
        ev, lb, inf = generate_sample(keys[i], time_steps, dt)
        events_list.append(ev)
        labels_list.append(lb)
        info_list.append(inf)
    return jnp.stack(events_list), jnp.stack(labels_list), info_list


# =============================================================================
# Main (test)
# =============================================================================
def main():
    print("=" * 60)
    print("  🌲 Sparse Forest — Collision-Free Event Camera")
    print("=" * 60)
    
    key = jax.random.PRNGKey(SEED)
    import time as _time

    print("\n  ⚡ Generating single sample...")
    t0 = _time.time()
    events, labels, info = generate_sample(key)
    print(f"  Time: {_time.time()-t0:.3f}s")
    print(f"  Events: {int(jnp.sum(jnp.abs(events)))}/{N_PIXELS*TIME_STEPS}")

    print(f"\n  ⚡ Batch (B={BATCH_SIZE})...")
    t0 = _time.time()
    key2 = jax.random.split(key, 2)[0]
    ev_b, lb_b, info_b = generate_batch(key2, BATCH_SIZE)
    elapsed = _time.time()-t0
    print(f"  Time: {elapsed:.3f}s ({elapsed/BATCH_SIZE:.3f}s/sample)")

if __name__ == "__main__":
    main()