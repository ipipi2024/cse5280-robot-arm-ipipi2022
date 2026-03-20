# ik_arm.py
# 2-link planar arm: forward kinematics, analytic Jacobian, Jacobian-transpose
# IK, and the Phase 4 simulation loop that replaces the point-robot with the
# arm end-effector.
# Imports crowd physics from crowd.py and robot obstacle/targeting from robot.py.

import numpy as np
from crowd import total_gradient_with_particles_softmin
from robot import update_robot_target, grad_robot_obstacle


# ─────────────────────────────────────────────
# Forward kinematics
# ─────────────────────────────────────────────

def arm_forward_kinematics(base, angles, lengths):
    """
    Compute joint positions for a 2-link planar arm.

    Convention
    ----------
    theta1 : absolute angle of link 1 from the +x axis (radians)
    theta2 : angle of link 2 relative to link 1 (radians)

    Joint positions
    ---------------
        joint0 = base                                (root, fixed)
        joint1 = base + L1 * [cos(θ1), sin(θ1)]     (elbow)
        ee     = joint1 + L2 * [cos(θ1+θ2), sin(θ1+θ2)]  (end-effector)

    Parameters
    ----------
    base    : (2,) fixed root position
    angles  : [theta1, theta2]
    lengths : [L1, L2]

    Returns
    -------
    joints : list of three (2,) arrays — [root, elbow, end_effector]
    """
    t1, t2 = angles
    L1, L2 = lengths
    elbow = np.array(base) + L1 * np.array([np.cos(t1), np.sin(t1)])
    ee    = elbow           + L2 * np.array([np.cos(t1 + t2), np.sin(t1 + t2)])
    return [np.array(base, dtype=float), elbow, ee]


# ─────────────────────────────────────────────
# Analytic Jacobian
# ─────────────────────────────────────────────

def arm_jacobian(angles, lengths):
    """
    Analytic 2×2 Jacobian of the end-effector w.r.t. joint angles [θ1, θ2].

    Derivation
    ----------
        ee_x = L1·cos(θ1) + L2·cos(θ1+θ2)
        ee_y = L1·sin(θ1) + L2·sin(θ1+θ2)

        ∂ee_x/∂θ1 = -L1·sin(θ1) - L2·sin(θ1+θ2)
        ∂ee_x/∂θ2 =             - L2·sin(θ1+θ2)
        ∂ee_y/∂θ1 =  L1·cos(θ1) + L2·cos(θ1+θ2)
        ∂ee_y/∂θ2 =               L2·cos(θ1+θ2)

    Returns
    -------
    J : (2, 2) — columns correspond to θ1, θ2
    """
    t1, t2 = angles
    L1, L2 = lengths
    s1,  c1  = np.sin(t1),      np.cos(t1)
    s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)
    return np.array([
        [-L1 * s1 - L2 * s12,  -L2 * s12],
        [ L1 * c1 + L2 * c12,   L2 * c12],
    ])


# ─────────────────────────────────────────────
# Jacobian-transpose IK step
# ─────────────────────────────────────────────

def arm_ik_step(base, angles, lengths, target, alpha_ik=0.05):
    """
    One Jacobian-transpose IK step nudging the end-effector toward target.

    Why Jacobian transpose?
      J^T is O(1) to compute (no inversion), always numerically stable, and
      sufficient for tracking a slow-moving target.  The update is a gradient
      step on the squared end-effector error:

          L(θ) = 0.5 · ||ee(θ) - target||²
          ∂L/∂θ = J^T · (ee - target)   →   θ -= alpha_ik · J^T · (ee - target)
                                         ↔   θ += alpha_ik · J^T · e

    Parameters
    ----------
    base     : (2,) arm root (fixed)
    angles   : [θ1, θ2] current joint angles
    lengths  : [L1, L2]
    target   : (2,) desired end-effector position
    alpha_ik : gradient step size (tune alongside robot_alpha)

    Returns
    -------
    new_angles : [θ1, θ2] updated joint angles (list, not numpy array,
                 to match the mutable state carried through the loop)
    """
    joints = arm_forward_kinematics(base, angles, lengths)
    ee     = joints[-1]
    e      = target - ee
    J      = arm_jacobian(angles, lengths)
    dtheta = alpha_ik * (J.T @ e)
    return [angles[0] + dtheta[0], angles[1] + dtheta[1]]


# ─────────────────────────────────────────────
# Phase 4 — IK arm simulation loop
# ─────────────────────────────────────────────

def run_evacuation_with_robot_arm(starts, exits, walls, beta=4.0,
                                  alpha=0.04, n_steps=1000, tol=0.15,
                                  R_p=0.6, w_p=30.0,
                                  arm_base=None,
                                  arm_angles=None,
                                  arm_lengths=None,
                                  alpha_ik=0.05,
                                  detection_radius=2.0,
                                  R_robot=0.9, w_robot=50.0,
                                  horizon=3, lambda_smooth=0.3):
    """
    Evacuation simulation where the robot is a 2-link planar arm.

    The targeting pipeline (clustering → EMA smoothing → prediction) is
    identical to Phase 3.  The only difference:

      Phase 3:  robot_pos += robot_alpha * (target - robot_pos)
      Phase 4:  angles     = arm_ik_step(base, angles, lengths, target, alpha_ik)
                ee_pos      = arm_forward_kinematics(base, angles, lengths)[-1]

    Particles see the end-effector position `ee_pos` as the dynamic
    obstacle — exactly the same quadratic-band repulsion cost as before.

    Parameters
    ----------
    arm_base    : (2,) fixed root of the arm.  Default: [5.0, 0.2]
    arm_angles  : [θ1, θ2] initial joint angles.  Default: [π/2, -π/6]
    arm_lengths : [L1, L2] link lengths.  Default: [3.0, 3.0]
    alpha_ik    : Jacobian-transpose IK step size
    (all other parameters identical to run_evacuation_with_robot_phase1)

    Returns
    -------
    trajectories          : list of N arrays (T_i, 2)
    ee_traj               : (n_steps, 2) end-effector positions over time
    arm_angles_log        : list of [θ1, θ2] at each step
    robot_targets         : (n_steps, 2) predicted targets used each step
    cluster_centroids_log : list of (k,2) or None
    dominant_centroid_log : list of (2,) or None
    smoothed_centroid_log : list of (2,) or None
    predicted_target_log  : list of (2,) or None
    """
    exits     = np.array(exits)
    N         = len(starts)
    positions = np.array(starts, dtype=float)
    active    = np.ones(N, dtype=bool)
    trajs     = [[p.copy()] for p in positions]

    base    = np.array(arm_base    if arm_base    is not None else [5.0, 0.2])
    angles  = list(arm_angles      if arm_angles  is not None else [np.pi / 2, -np.pi / 6])
    lengths = list(arm_lengths     if arm_lengths is not None else [3.0, 3.0])

    ee_pos  = arm_forward_kinematics(base, angles, lengths)[-1]

    ee_traj           = [ee_pos.copy()]
    arm_angles_log    = [list(angles)]
    robot_targets_log = [ee_pos.copy()]

    cluster_centroids_log = [None]
    dominant_centroid_log = [None]
    smoothed_centroid_log = [None]
    predicted_target_log  = [None]
    prev_smoothed         = None
    prev_target           = ee_pos.copy()

    for _ in range(n_steps):
        if not np.any(active):
            break

        # ── a. Cluster → smooth → predict target ──────────────────────────
        target, centroids, dom_centroid, smoothed, predicted = \
            update_robot_target(
                positions, active, exits, detection_radius, prev_target,
                prev_smoothed=prev_smoothed, horizon=horizon,
                lambda_smooth=lambda_smooth
            )
        cluster_centroids_log.append(centroids)
        dominant_centroid_log.append(dom_centroid)
        smoothed_centroid_log.append(smoothed)
        predicted_target_log.append(predicted)
        if smoothed is not None:
            prev_smoothed = smoothed
        prev_target = target

        # ── b. IK step: move arm end-effector toward predicted target ──────
        angles = arm_ik_step(base, angles, lengths, target, alpha_ik)
        ee_pos = arm_forward_kinematics(base, angles, lengths)[-1]

        # ── c. Snapshot for synchronous gradient computation ───────────────
        snapshot = positions.copy()

        # ── d. Compute gradients — ee_pos replaces robot_pos as obstacle ──
        grads = np.zeros_like(positions)
        for i in range(N):
            if active[i]:
                xi = snapshot[i]
                grads[i] = total_gradient_with_particles_softmin(
                    i, snapshot, exits, walls, R_p, w_p, beta
                )
                grads[i] += grad_robot_obstacle(xi, ee_pos, R_robot, w_robot)

        # ── e. Update all particles simultaneously ─────────────────────────
        for i in range(N):
            if active[i]:
                positions[i] -= alpha * grads[i]
                trajs[i].append(positions[i].copy())
                if np.min(np.linalg.norm(exits - positions[i], axis=1)) < tol:
                    active[i] = False

        ee_traj.append(ee_pos.copy())
        arm_angles_log.append(list(angles))
        robot_targets_log.append(target.copy())

    return ([np.array(t) for t in trajs],
            np.array(ee_traj),
            arm_angles_log,
            np.array(robot_targets_log),
            cluster_centroids_log,
            dominant_centroid_log,
            smoothed_centroid_log,
            predicted_target_log)
