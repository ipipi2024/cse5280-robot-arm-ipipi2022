# robot.py
# Robot obstacle physics, near-exit particle detection, k-means clustering,
# EMA centroid smoothing, cluster-motion prediction, and the Phase 3
# point-robot simulation loop.
# Imports crowd physics from crowd.py; has no knowledge of arm kinematics.

import numpy as np
from config import EPS
from crowd import total_gradient_with_particles_softmin


# ─────────────────────────────────────────────
# Robot obstacle cost + analytic gradient
# ─────────────────────────────────────────────

def robot_obstacle_cost(x, robot_pos, R_robot, w_robot):
    """
    Quadratic-band repulsion from the robot end-effector.

    Identical in form to particle_repulsion_cost — the robot is simply
    treated as a dynamic obstacle whose position updates each timestep.

        d        = ||x - robot_pos||
        C_robot  = 0.5 * w_robot * (R_robot - d)^2   if d < R_robot
                 = 0                                   otherwise
    """
    d = np.linalg.norm(x - robot_pos)
    if d < R_robot:
        return 0.5 * w_robot * (R_robot - d) ** 2
    return 0.0


def grad_robot_obstacle(x, robot_pos, R_robot, w_robot):
    """
    Analytic gradient of C_robot w.r.t. particle position x.

    Derivation — same chain rule as grad_particle_repulsion:
        dC/dd  = -w_robot * (R_robot - d)
        dd/dx  = (x - robot_pos) / d

        grad = -w_robot * (R_robot - d) * (x - robot_pos) / max(d, EPS)

    Pushes the particle away from the robot point.
    """
    d = np.linalg.norm(x - robot_pos)
    if d >= R_robot:
        return np.zeros_like(x)
    return -w_robot * (R_robot - d) * (x - robot_pos) / max(d, EPS)


# ─────────────────────────────────────────────
# Near-exit detection
# ─────────────────────────────────────────────

def find_particles_near_exits(positions, active, exits, detection_radius):
    """
    Return indices of active particles within detection_radius of any exit.

    Parameters
    ----------
    positions        : (N, 2) current particle positions
    active           : (N,) bool array — False for particles that converged
    exits            : (K, 2) exit positions
    detection_radius : scalar — detection radius around each exit

    Returns
    -------
    indices : list of ints — particles currently flowing near exits
    """
    indices = []
    for i in range(len(positions)):
        if not active[i]:
            continue
        dist_to_any_exit = np.min(np.linalg.norm(exits - positions[i], axis=1))
        if dist_to_any_exit < detection_radius:
            indices.append(i)
    return indices


# ─────────────────────────────────────────────
# K-means clustering
# ─────────────────────────────────────────────

def kmeans(points, k, n_iter=10):
    """
    Minimal k-means for (M, 2) point arrays.

    Parameters
    ----------
    points : (M, 2) positions to cluster
    k      : number of clusters
    n_iter : Lloyd iterations

    Returns
    -------
    centroids : (k, 2) cluster centres
    labels    : (M,) int cluster index per point
    """
    rng = np.random.default_rng(seed=0)          # fixed seed → deterministic
    idx = rng.choice(len(points), size=k, replace=False)
    centroids = points[idx].copy()

    labels = np.zeros(len(points), dtype=int)
    for _ in range(n_iter):
        # Assignment: each point → nearest centroid
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Update: recompute centroid for each cluster
        for c in range(k):
            members = points[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
            # else: keep old centroid — avoids NaN for empty cluster

    return centroids, labels


# ─────────────────────────────────────────────
# EMA centroid smoothing
# ─────────────────────────────────────────────

def smooth_centroid(current, prev_smoothed, lambda_smooth):
    """
    Exponential moving average (EMA) over the dominant cluster centroid.

    smoothed = (1 - lambda_) * prev_smoothed + lambda_ * current

    Why smooth before predicting?
      The raw dominant centroid jumps each step because k-means re-assigns
      particles and occasionally swaps which cluster is "dominant".  Taking
      an EMA with a small lambda_ (e.g. 0.3) damps these frame-to-frame
      discontinuities so the velocity estimate used for prediction is a
      stable trend rather than noise.

    Parameters
    ----------
    current       : (2,) raw dominant centroid this step
    prev_smoothed : (2,) EMA value from the previous step, or None
    lambda_smooth : float in (0, 1] — blend weight for the new observation.
                    Smaller → more smoothing, larger lag.
                    Larger → tracks faster, less smoothing.

    Returns
    -------
    smoothed : (2,) updated EMA value
    """
    if prev_smoothed is None:
        return current.copy()
    return (1.0 - lambda_smooth) * prev_smoothed + lambda_smooth * current


# ─────────────────────────────────────────────
# Cluster-motion prediction
# ─────────────────────────────────────────────

def predict_cluster_target(smoothed_centroid, prev_smoothed, horizon,
                           room_bounds=(0, 10)):
    """
    Predict where the dominant cluster will be after `horizon` steps,
    using the smoothed centroid for velocity estimation. Clamps the result
    to room bounds to prevent the robot from targeting outside the room.

    velocity  = smoothed_centroid - prev_smoothed
    predicted = smoothed_centroid + horizon * velocity
    predicted = clamp(predicted, room_bounds)

    Parameters
    ----------
    smoothed_centroid : (2,) EMA-smoothed dominant centroid this step
    prev_smoothed     : (2,) EMA-smoothed dominant centroid previous step, or None
    horizon           : int — steps to look ahead (0 → return smoothed_centroid)
    room_bounds       : (lo, hi) scalar bounds applied to both x and y

    Returns
    -------
    predicted : (2,) clamped predicted position
    """
    if prev_smoothed is None or horizon == 0:
        return smoothed_centroid.copy()

    velocity  = smoothed_centroid - prev_smoothed
    raw       = smoothed_centroid + horizon * velocity
    lo, hi    = room_bounds
    return np.clip(raw, lo, hi)


# ─────────────────────────────────────────────
# Robot targeting: cluster → smooth → predict
# ─────────────────────────────────────────────

def update_robot_target(positions, active, exits, detection_radius, prev_target,
                        k=2, prev_smoothed=None, horizon=3, lambda_smooth=0.3):
    """
    Compute the robot's new target using clustering + EMA smoothing + prediction.

    Pipeline per step
    -----------------
    1. Find particles near any exit.
    2. If none detected: hold previous target.
    3. If fewer than k particles: fall back to mean (no smoothing/prediction).
    4. Run k-means(k), pick dominant cluster → raw centroid.
    5. Smooth raw centroid with EMA (lambda_smooth).
    6. Predict smoothed centroid H steps ahead; clamp to room.
    7. Use predicted position as robot target.

    Parameters
    ----------
    positions        : (N, 2) current positions
    active           : (N,) bool array
    exits            : (K, 2) exit positions
    detection_radius : float
    prev_target      : (2,) last known target (fallback)
    k                : number of clusters
    prev_smoothed    : (2,) EMA value from the previous step, or None
    horizon          : int — prediction look-ahead steps
    lambda_smooth    : float — EMA blend weight (0 < lambda_ <= 1)

    Returns
    -------
    target            : (2,) robot target (predicted, clamped)
    centroids         : (k, 2) all k-means centroids, or None if skipped
    dom_centroid      : (2,) raw dominant centroid, or None
    smoothed          : (2,) EMA-smoothed dominant centroid, or None
    predicted_target  : (2,) predicted target (same as target when active)
    """
    near = find_particles_near_exits(positions, active, exits, detection_radius)
    if len(near) == 0:
        return prev_target, None, None, None, None

    pts = positions[near]
    if len(pts) < k:
        mean_pos = pts.mean(axis=0)
        return mean_pos, None, mean_pos, mean_pos, mean_pos

    centroids, labels = kmeans(pts, k=k)
    sizes         = np.bincount(labels, minlength=k)
    dominant      = np.argmax(sizes)
    dom_centroid  = centroids[dominant]

    smoothed  = smooth_centroid(dom_centroid, prev_smoothed, lambda_smooth)
    predicted = predict_cluster_target(smoothed, prev_smoothed, horizon)
    return predicted, centroids, dom_centroid, smoothed, predicted


# ─────────────────────────────────────────────
# Phase 3 — point-robot simulation loop
# ─────────────────────────────────────────────

def run_evacuation_with_robot_phase1(starts, exits, walls, beta=4.0,
                                     alpha=0.04, n_steps=1000, tol=0.15,
                                     R_p=0.6, w_p=30.0,
                                     robot_start=None,
                                     robot_alpha=0.08,
                                     detection_radius=2.0,
                                     R_robot=0.9, w_robot=50.0,
                                     horizon=3, lambda_smooth=0.3):
    """
    Evacuation simulation with Phase 3 robot interference agent
    (clustering + EMA smoothing + clamped prediction).

    Each timestep:
      a. cluster near-exit particles → raw dominant centroid
      b. smooth centroid with EMA (lambda_smooth)
      c. predict smoothed centroid H steps ahead; clamp to room
      d. move robot toward predicted position proportionally
      e. compute all particle gradients from snapshot (synchronous):
           soft-min exit attraction
         + wall repulsion
         + inter-particle repulsion
         + robot obstacle repulsion
      f. update all particle positions

    Parameters
    ----------
    robot_start      : (2,) initial robot position.  Defaults to room centre.
    robot_alpha      : robot movement speed toward target (proportional gain)
    detection_radius : radius around each exit that counts as "near exit"
    R_robot          : robot obstacle influence radius
    w_robot          : robot obstacle penalty weight
    horizon          : int — prediction look-ahead steps (0 = reactive only)
    lambda_smooth    : float — EMA blend weight for centroid smoothing (0 < λ ≤ 1)

    Returns
    -------
    trajectories          : list of N arrays (T_i, 2) — particle paths
    robot_traj            : (n_steps, 2) array — robot positions
    robot_targets         : (n_steps, 2) array — robot targets (predicted)
    cluster_centroids_log : list of (k,2) or None — all k-means centroids per step
    dominant_centroid_log : list of (2,) or None — raw dominant centroid per step
    smoothed_centroid_log : list of (2,) or None — EMA-smoothed centroid per step
    predicted_target_log  : list of (2,) or None — predicted targets per step
    """
    exits     = np.array(exits)
    N         = len(starts)
    positions = np.array(starts, dtype=float)
    active    = np.ones(N, dtype=bool)
    trajs     = [[p.copy()] for p in positions]

    robot_pos    = np.array(robot_start if robot_start is not None
                            else [5.0, 5.0], dtype=float)
    robot_target = robot_pos.copy()
    robot_traj            = [robot_pos.copy()]
    robot_targets_log     = [robot_target.copy()]
    cluster_centroids_log = [None]
    dominant_centroid_log = [None]
    smoothed_centroid_log = [None]
    predicted_target_log  = [None]
    prev_smoothed         = None

    for _ in range(n_steps):
        if not np.any(active):
            break

        # ── a. Cluster → smooth → predict ─────────────────────────────────
        robot_target, centroids, dom_centroid, smoothed, predicted = \
            update_robot_target(
                positions, active, exits, detection_radius, robot_target,
                prev_smoothed=prev_smoothed, horizon=horizon,
                lambda_smooth=lambda_smooth
            )
        cluster_centroids_log.append(centroids)
        dominant_centroid_log.append(dom_centroid)
        smoothed_centroid_log.append(smoothed)
        predicted_target_log.append(predicted)
        if smoothed is not None:
            prev_smoothed = smoothed

        # ── b. Move robot toward target (proportional control) ────────────
        robot_pos = robot_pos + robot_alpha * (robot_target - robot_pos)

        # ── c. Snapshot all positions for synchronous gradient computation ─
        snapshot = positions.copy()

        # ── d. Compute gradients (soft-min goal + walls + repulsion + robot)─
        grads = np.zeros_like(positions)
        for i in range(N):
            if active[i]:
                xi = snapshot[i]
                grads[i] = total_gradient_with_particles_softmin(
                    i, snapshot, exits, walls, R_p, w_p, beta
                )
                grads[i] = grads[i] + grad_robot_obstacle(
                    xi, robot_pos, R_robot, w_robot
                )

        # ── e. Update all particles simultaneously ─────────────────────────
        for i in range(N):
            if active[i]:
                positions[i] = positions[i] - alpha * grads[i]
                trajs[i].append(positions[i].copy())
                if np.min(np.linalg.norm(exits - positions[i], axis=1)) < tol:
                    active[i] = False

        robot_traj.append(robot_pos.copy())
        robot_targets_log.append(robot_target.copy())

    return ([np.array(t) for t in trajs],
            np.array(robot_traj),
            np.array(robot_targets_log),
            cluster_centroids_log,
            dominant_centroid_log,
            smoothed_centroid_log,
            predicted_target_log)
