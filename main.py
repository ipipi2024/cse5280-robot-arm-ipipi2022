# main.py
# Entry point: defines scenes and runs all demos.
# All simulation logic lives in crowd.py, robot.py, ik_arm.py.
# All plotting logic lives in visualization.py.

import numpy as np

from crowd import (
    run_simulation,
    run_multi_particle_simulation,
    run_multi_particle_simulation_with_repulsion,
    run_evacuation_simulation,
)
from robot import run_evacuation_with_robot_phase1
from ik_arm import run_evacuation_with_robot_arm
from visualization import (
    plot_results,
    plot_multi_particle_results,
    plot_cost_field_and_vectors,
    plot_evacuation_with_robot,
    plot_evacuation_with_robot_arm,
    animate_evacuation,
)


if __name__ == "__main__":
    g = np.array([5.5, 7.0])   # goal — inside the n-shaped enclosure

    # ── Outer square boundary [0,10]×[0,10] ──────────────────────────────────
    boundary_R = 1.0
    boundary_w = 80.0
    boundary_walls = [
        {'a': np.array([ 0.0,  0.0]), 'b': np.array([10.0,  0.0]),  # bottom
         'R': boundary_R, 'w': boundary_w},
        {'a': np.array([10.0,  0.0]), 'b': np.array([10.0, 10.0]),  # right
         'R': boundary_R, 'w': boundary_w},
        {'a': np.array([10.0, 10.0]), 'b': np.array([ 0.0, 10.0]),  # top
         'R': boundary_R, 'w': boundary_w},
        {'a': np.array([ 0.0, 10.0]), 'b': np.array([ 0.0,  0.0]),  # left
         'R': boundary_R, 'w': boundary_w},
    ]

    # ── n-shaped interior enclosure ───────────────────────────────────────────
    #
    #   (3.5,8.5)────────────(7.5,8.5)   ← top bar
    #      │                     │
    #      │    goal (5.5,7.0)   │
    #      │                     │
    #   (3.5,3.0)           (7.5,3.0)    ← open bottom gap  (y < 3.0)
    #
    interior_R = 1.0
    interior_w = 120.0
    interior_walls = [
        {'a': np.array([3.5, 3.0]), 'b': np.array([3.5, 8.5]),  # left leg
         'R': interior_R, 'w': interior_w},
        {'a': np.array([3.5, 8.5]), 'b': np.array([7.5, 8.5]),  # top bar
         'R': interior_R, 'w': interior_w},
        {'a': np.array([7.5, 8.5]), 'b': np.array([7.5, 3.0]),  # right leg
         'R': interior_R, 'w': interior_w},
    ]

    walls = boundary_walls + interior_walls

    # ── Single-particle run (kept for reference + cost-field debug) ───────────
    x0 = np.array([2.0, 2.0])
    trajectory = run_simulation(x0, g, walls, alpha=0.05, n_steps=800)
    plot_results(trajectory, x0, g, walls)
    plot_cost_field_and_vectors(x0, g, walls, trajectory=trajectory, grid_n=60)

    # ── Multi-particle run ────────────────────────────────────────────────────
    N = 25
    np.random.seed(42)   # fix seed for reproducibility

    # Sample starts from three regions outside the n-shape enclosure:
    #   left  — x∈[1,3],   y∈[2,8]
    #   right — x∈[7,9],   y∈[2,8]
    #   bottom— x∈[3.5,7.5], y∈[1,2.5]
    n_left   = N // 3
    n_right  = N // 3
    n_bottom = N - n_left - n_right

    left_starts   = np.column_stack([
        np.random.uniform(1.0, 3.0, n_left),
        np.random.uniform(2.0, 8.0, n_left),
    ])
    right_starts  = np.column_stack([
        np.random.uniform(7.0, 9.0, n_right),
        np.random.uniform(2.0, 8.0, n_right),
    ])
    bottom_starts = np.column_stack([
        np.random.uniform(3.5, 7.5, n_bottom),
        np.random.uniform(1.0, 2.5, n_bottom),
    ])

    starts = np.vstack([left_starts, right_starts, bottom_starts])

    # Independent (no repulsion)
    trajectories = run_multi_particle_simulation(
        starts, g, walls, alpha=0.04, n_steps=1000, tol=0.05
    )
    plot_multi_particle_results(trajectories, starts, g, walls, repulsion=False)

    # With pairwise repulsion — synchronous updates, same walls and goal
    trajectories_repel = run_multi_particle_simulation_with_repulsion(
        starts, g, walls, alpha=0.04, n_steps=1000, tol=0.05,
        R_p=0.6, w_p=30.0
    )
    plot_multi_particle_results(trajectories_repel, starts, g, walls, repulsion=True)

    # ── Evacuation scenario: soft-min goal toward two exits ───────────────────
    #
    # Particles start randomly across the room.
    # Two exits are placed at the bottom — left and right of centre.
    # No hard assignment: each particle's pull toward each exit is weighted
    # by exp(-beta * C_i), so nearby particles naturally prefer the closer exit.
    # Repulsion spreads particles across both exits to avoid crowding.
    #
    beta = 4.0     # sharpness: higher → stronger preference for the closer exit

    exits = np.array([
        [2.0, 1.2],   # left exit
        [8.0, 1.2],   # right exit
    ])

    # For this scenario use the same boundary walls but drop the n-shape:
    evac_walls = boundary_walls

    np.random.seed(42)
    evac_starts = np.column_stack([
        np.random.uniform(1.0, 9.0, N),
        np.random.uniform(3.0, 9.0, N),
    ])

    evac_trajectories = run_evacuation_simulation(
        evac_starts, exits, evac_walls, beta=beta,
        alpha=0.04, n_steps=1000, tol=0.15,
        R_p=0.6, w_p=30.0
    )
    plot_multi_particle_results(
        evac_trajectories, evac_starts, g=None, walls=evac_walls,
        repulsion=True, exits=exits
    )

    # ── Phase 1: evacuation with robot interference ───────────────────────────
    # Same scene as the soft-min evacuation above (open room, two exits).
    # A robot point starts at the centre of the room, detects particles
    # flowing toward the exits, and moves to intercept that flow.
    # Particles treat the robot as a dynamic obstacle — same quadratic-band
    # repulsion as inter-particle repulsion, just anchored to robot_pos.
    np.random.seed(42)
    phase1_starts = np.column_stack([
        np.random.uniform(1.0, 9.0, N),
        np.random.uniform(3.0, 9.0, N),
    ])

    (phase1_trajs, robot_traj, robot_targets,
     cluster_log, dominant_log, smoothed_log, predicted_log) = \
        run_evacuation_with_robot_phase1(
            phase1_starts, exits, evac_walls,
            beta=4.0, alpha=0.04, n_steps=1000, tol=0.15,
            R_p=0.6,   w_p=30.0,
            robot_start=[5.0, 5.0],
            robot_alpha=0.08,
            detection_radius=2.0,
            R_robot=0.9, w_robot=50.0,
            horizon=3, lambda_smooth=0.3,
        )
    plot_evacuation_with_robot(
        phase1_trajs, robot_traj, robot_targets,
        phase1_starts, exits, evac_walls,
        R_robot=0.9,
        cluster_centroids_log=cluster_log,
        dominant_centroid_log=dominant_log,
        smoothed_centroid_log=smoothed_log,
        predicted_target_log=predicted_log,
    )

    # ── Phase 4: evacuation with 2-link planar arm IK ─────────────────────
    # Same crowd scene as Phase 3. The robot point is replaced by the
    # end-effector of a 2-link arm rooted at (5, 0.2) — just above the
    # bottom wall, centred horizontally. The arm uses Jacobian-transpose IK
    # to track the predicted dominant cluster target each timestep.
    # Particles repel from the end-effector position exactly as before.
    np.random.seed(42)
    arm_starts = np.column_stack([
        np.random.uniform(1.0, 9.0, N),
        np.random.uniform(3.0, 9.0, N),
    ])

    arm_base    = [5.0, 0.2]          # fixed root just above bottom wall
    arm_lengths = [3.0, 3.0]          # total reach = 6.0 — covers both exits
    arm_angles0 = [np.pi / 2, -np.pi / 6]   # initial: roughly pointing up

    (arm_trajs, ee_traj, arm_angles_log,
     arm_robot_targets, arm_cluster_log,
     arm_dominant_log, arm_smoothed_log, arm_predicted_log) = \
        run_evacuation_with_robot_arm(
            arm_starts, exits, evac_walls,
            beta=4.0, alpha=0.04, n_steps=1000, tol=0.15,
            R_p=0.6, w_p=30.0,
            arm_base=arm_base,
            arm_angles=arm_angles0,
            arm_lengths=arm_lengths,
            alpha_ik=0.05,
            detection_radius=2.0,
            R_robot=0.9, w_robot=50.0,
            horizon=3, lambda_smooth=0.3,
        )
    plot_evacuation_with_robot_arm(
        arm_trajs, ee_traj, arm_angles_log,
        arm_robot_targets, arm_starts, exits, evac_walls,
        arm_base=arm_base,
        arm_lengths=arm_lengths,
        R_robot=0.9,
        dominant_centroid_log=arm_dominant_log,
        smoothed_centroid_log=arm_smoothed_log,
        predicted_target_log=arm_predicted_log,
    )

    # ── Animation: Phase 4 IK arm ─────────────────────────────────────────────
    # Assign to a variable — FuncAnimation stops if the object is garbage-collected
    anim = animate_evacuation(
        arm_trajs, exits, evac_walls,
        ee_traj=ee_traj,
        arm_angles_log=arm_angles_log,
        arm_base=arm_base,
        arm_lengths=arm_lengths,
        predicted_target_log=arm_predicted_log,
        interval=30,
        title="Phase 4 — IK arm evacuation (animated)",
    )
