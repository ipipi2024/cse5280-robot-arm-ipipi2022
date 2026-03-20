# visualization.py
# All matplotlib plotting functions for the crowd-evacuation simulation.
# Imports only numpy, matplotlib, and crowd physics needed for cost-field plots.
# Has no knowledge of robot internals or arm kinematics.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from crowd import total_cost, total_gradient
from ik_arm import arm_forward_kinematics


# ─────────────────────────────────────────────
# Single-particle trajectory plot
# ─────────────────────────────────────────────

def plot_results(trajectory, x0, g, walls):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(trajectory[:, 0], trajectory[:, 1],
            color='steelblue', linewidth=1.5, zorder=2)

    ax.scatter(*x0, color='green', s=100, zorder=5)
    ax.scatter(*g,  color='red',   s=100, zorder=5)

    for wall in walls:
        ax.plot([wall['a'][0], wall['b'][0]],
                [wall['a'][1], wall['b'][1]],
                color='black', linewidth=3, zorder=4)

    legend_handles = [
        Line2D([0], [0], color='steelblue', label='Trajectory'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label='Start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='Goal'),
        Line2D([0], [0], color='black', linewidth=3, label='Wall'),
    ]
    ax.legend(handles=legend_handles)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title("Gradient Descent Particle Simulation\n(analytic gradients)")
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Multi-particle trajectory plot
# ─────────────────────────────────────────────

def plot_multi_particle_results(trajectories, starts, g, walls,
                                repulsion=False, exits=None):
    """
    Plot all particle trajectories together on one axes.

    Parameters
    ----------
    trajectories : list of (T_i, 2) arrays
    starts       : (N, 2) array used as start positions
    g            : single goal position (pass None when using exits)
    walls        : list of wall dicts
    repulsion    : bool — controls subtitle text
    exits        : optional (K, 2) array — if provided, drawn instead of g
    """
    N = len(trajectories)
    colours = plt.cm.hsv(np.linspace(0, 0.85, max(N, 1)))

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, traj in enumerate(trajectories):
        colour = colours[i % len(colours)]
        ax.plot(traj[:, 0], traj[:, 1], color=colour, linewidth=0.9,
                alpha=0.75, zorder=2)
        ax.scatter(*traj[0], color=colour, edgecolors='black',
                   s=50, zorder=5, linewidths=0.6)

    legend_handles = [
        Line2D([0], [0], color='grey', linewidth=0.9,
               label=f'Trajectories (N={N})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markeredgecolor='black', markersize=7, label='Start positions'),
        Line2D([0], [0], color='black', linewidth=3, label='Wall'),
    ]
    if exits is not None:
        for k, ex in enumerate(exits):
            ax.scatter(*ex, color='lime', s=200, zorder=6,
                       marker='*', edgecolors='black', linewidths=0.8)
            ax.annotate(f'Exit {k+1}', xy=ex,
                        xytext=(ex[0] + 0.15, ex[1] + 0.15),
                        fontsize=9, color='darkgreen', fontweight='bold')
        legend_handles.append(
            Line2D([0], [0], marker='*', color='w', markerfacecolor='lime',
                   markeredgecolor='black', markersize=14, label='Exits'))
    elif g is not None:
        ax.scatter(*g, color='red', s=140, zorder=6, marker='*')
        legend_handles.append(
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markersize=14, label='Goal'))

    for wall in walls:
        ax.plot([wall['a'][0], wall['b'][0]],
                [wall['a'][1], wall['b'][1]],
                color='black', linewidth=3, zorder=4)

    ax.legend(handles=legend_handles, loc='upper left')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title(
        f"Multi-particle gradient descent  (N={len(trajectories)})\n"
        f"{'Inter-particle repulsion enabled' if repulsion else 'Each particle simulated independently — no inter-particle forces'}"
    )
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Cost field + gradient vector visualisation
# ─────────────────────────────────────────────

def plot_cost_field_and_vectors(x0, g, walls, trajectory=None, grid_n=60):
    """
    Visualise the total cost landscape and the descent direction field.

    Parameters
    ----------
    x0         : start position (plotted in green)
    g          : goal position  (plotted in red)
    walls      : list of wall dicts with 'a','b','R','w'
    trajectory : optional (N,2) array — overlaid in cyan if provided
    grid_n     : number of grid points per axis (higher = finer but slower)
    """
    xs = np.linspace(0, 10, grid_n)
    ys = np.linspace(0, 10, grid_n)
    XX, YY = np.meshgrid(xs, ys)

    Z = np.zeros_like(XX)
    U = np.zeros_like(XX)
    V = np.zeros_like(XX)

    for i in range(grid_n):
        for j in range(grid_n):
            pos     = np.array([XX[i, j], YY[i, j]])
            Z[i, j] = total_cost(pos, g, walls)
            grad    = total_gradient(pos, g, walls)
            U[i, j] = -grad[0]
            V[i, j] = -grad[1]

    fig, ax = plt.subplots(figsize=(8, 8))

    Z_display = np.clip(Z, 0, np.percentile(Z, 95))
    cf = ax.contourf(XX, YY, Z_display, levels=40, cmap='plasma')
    plt.colorbar(cf, ax=ax, label='Total cost (clipped at 95th percentile)')

    step = max(1, grid_n // 20)
    ax.quiver(XX[::step, ::step], YY[::step, ::step],
              U[::step, ::step],  V[::step, ::step],
              color='white', alpha=0.6, width=0.003, headwidth=4)

    for wall in walls:
        ax.plot([wall['a'][0], wall['b'][0]],
                [wall['a'][1], wall['b'][1]],
                color='black', linewidth=3, zorder=4)

    ax.scatter(*x0, color='lime', s=120, zorder=6)
    ax.scatter(*g,  color='red',  s=120, zorder=6)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
               markersize=10, label='Start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='Goal'),
        Line2D([0], [0], color='black', linewidth=3, label='Wall'),
        Line2D([0], [0], color='white', alpha=0.6, label='Descent direction'),
    ]

    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                color='cyan', linewidth=1.5, zorder=5)
        legend_handles.insert(0,
            Line2D([0], [0], color='cyan', linewidth=1.5, label='Trajectory'))

    ax.legend(handles=legend_handles, loc='upper left',
              facecolor='#222', labelcolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title("Total cost field and descent directions\n"
                 "(arrows point downhill — the direction gradient descent moves)")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Phase 3 — point-robot evacuation plot
# ─────────────────────────────────────────────

def plot_evacuation_with_robot(trajectories, robot_traj, robot_targets,
                               starts, exits, walls, R_robot=0.9,
                               cluster_centroids_log=None,
                               dominant_centroid_log=None,
                               smoothed_centroid_log=None,
                               predicted_target_log=None):
    """
    Plot the Phase 3 evacuation scene:
      - particle trajectories (hsv colour per particle)
      - start markers (same colour, black edge)
      - exit markers (lime stars, labelled)
      - robot trajectory (dashed magenta)
      - robot influence-radius circles (faint magenta) at each step
      - robot final position (magenta X)
      - robot target trail (small grey dots)
      - cluster centroids trail (black circles, optional)
      - raw dominant centroid trail (blue squares, optional)
      - smoothed centroid trail (cyan triangles, optional)
      - predicted target trail (orange diamonds, optional)
      - faint arrow from smoothed → predicted at each sampled step (optional)
      - walls (black)
    """
    N       = len(trajectories)
    colours = plt.cm.hsv(np.linspace(0, 0.85, max(N, 1)))

    fig, ax = plt.subplots(figsize=(9, 9))

    # Particle trajectories + start markers
    for i, traj in enumerate(trajectories):
        c = colours[i % len(colours)]
        ax.plot(traj[:, 0], traj[:, 1], color=c,
                linewidth=0.9, alpha=0.7, zorder=2)
        ax.scatter(*traj[0], color=c, edgecolors='black',
                   s=45, zorder=5, linewidths=0.6)

    # Robot influence-radius circles at each step (faint)
    step = max(1, len(robot_traj) // 40)
    for pos in robot_traj[::step]:
        ax.add_patch(Circle(pos, R_robot, color='magenta',
                            fill=False, linewidth=0.6, alpha=0.18, zorder=3))
    ax.add_patch(Circle(robot_traj[-1], R_robot, color='magenta',
                        fill=False, linewidth=1.2, alpha=0.55, zorder=6))

    # Robot trajectory + final marker
    ax.plot(robot_traj[:, 0], robot_traj[:, 1],
            color='magenta', linewidth=1.6, linestyle='--',
            zorder=6, label='Robot trajectory')
    ax.scatter(*robot_traj[-1], color='magenta', marker='X',
               s=160, zorder=7, edgecolors='black', linewidths=0.8,
               label='Robot (final)')

    # Robot target trail
    ax.scatter(robot_targets[:, 0], robot_targets[:, 1],
               color='grey', s=6, alpha=0.4, zorder=3,
               label='Robot target trail')

    # Cluster centroids trail
    if cluster_centroids_log is not None:
        all_centroids = [c for c in cluster_centroids_log if c is not None]
        if all_centroids:
            step = max(1, len(all_centroids) // 60)
            for c_arr in all_centroids[::step]:
                ax.scatter(c_arr[:, 0], c_arr[:, 1],
                           color='black', s=18, alpha=0.35,
                           marker='o', zorder=4, linewidths=0)

    # Raw dominant, smoothed, predicted centroid trails + connector arrows
    have_dom  = dominant_centroid_log is not None
    have_smo  = smoothed_centroid_log is not None
    have_pred = predicted_target_log is not None

    if have_dom or have_smo or have_pred:
        steps = [i for i, d in enumerate(dominant_centroid_log or [])
                 if d is not None]
        if steps:
            samp = max(1, len(steps) // 40)

            if have_dom:
                dom_arr = np.array([dominant_centroid_log[i] for i in steps])
                ax.scatter(dom_arr[:, 0], dom_arr[:, 1],
                           color='royalblue', s=20, alpha=0.40,
                           marker='s', zorder=5, linewidths=0,
                           label='Raw dominant centroid')

            if have_smo:
                smo_valid = [smoothed_centroid_log[i] for i in steps
                             if smoothed_centroid_log[i] is not None]
                if smo_valid:
                    smo_arr = np.array(smo_valid)
                    ax.scatter(smo_arr[:, 0], smo_arr[:, 1],
                               color='cyan', s=20, alpha=0.50,
                               marker='^', zorder=5, linewidths=0,
                               label='Smoothed centroid (EMA)')

            if have_smo and have_pred:
                for i in steps[::samp]:
                    smo  = smoothed_centroid_log[i]
                    pred = predicted_target_log[i]
                    if smo is not None and pred is not None:
                        ax.annotate('', xy=pred, xytext=smo,
                                    arrowprops=dict(arrowstyle='->',
                                                    color='orangered',
                                                    lw=0.8, alpha=0.35),
                                    zorder=4)

            if have_pred:
                pred_valid = [predicted_target_log[i] for i in steps
                              if predicted_target_log[i] is not None]
                if pred_valid:
                    pred_arr = np.array(pred_valid)
                    ax.scatter(pred_arr[:, 0], pred_arr[:, 1],
                               color='orangered', s=28, alpha=0.55,
                               marker='D', zorder=5, linewidths=0,
                               label='Predicted target')

    # Exits
    for k, ex in enumerate(exits):
        ax.scatter(*ex, color='lime', s=200, zorder=8,
                   marker='*', edgecolors='black', linewidths=0.8)
        ax.annotate(f'Exit {k+1}', xy=ex,
                    xytext=(ex[0] + 0.15, ex[1] + 0.15),
                    fontsize=9, color='darkgreen', fontweight='bold')

    # Walls
    for wall in walls:
        ax.plot([wall['a'][0], wall['b'][0]],
                [wall['a'][1], wall['b'][1]],
                color='black', linewidth=3, zorder=4)

    legend_handles = [
        Line2D([0], [0], color='grey', linewidth=0.9,
               label=f'Particle trajectories (N={N})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markeredgecolor='black', markersize=7, label='Starts'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='lime',
               markeredgecolor='black', markersize=13, label='Exits'),
        Line2D([0], [0], color='magenta', linewidth=1.6,
               linestyle='--', label='Robot trajectory'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='magenta',
               markeredgecolor='black', markersize=11, label='Robot (final)'),
        Line2D([0], [0], color='magenta', linewidth=1.2, alpha=0.55,
               label=f'Robot influence radius (R={R_robot})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markersize=5, alpha=0.5, label='Robot target trail'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
               markersize=5, alpha=0.5, label='Cluster centroids'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='royalblue',
               markersize=6, alpha=0.6, label='Raw dominant centroid'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='cyan',
               markersize=6, alpha=0.6, label='Smoothed centroid (EMA)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='orangered',
               markersize=6, alpha=0.7, label='Predicted target'),
        Line2D([0], [0], color='black', linewidth=3, label='Wall'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title(
        f"Phase 3 — Robot interference agent with clustering + prediction  (N={N})\n"
        "Robot targets predicted dominant cluster (k-means k=2, horizon=H steps)"
    )
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Phase 4 — IK arm evacuation plot
# ─────────────────────────────────────────────

def plot_evacuation_with_robot_arm(trajectories, ee_traj, arm_angles_log,
                                   robot_targets, starts, exits, walls,
                                   arm_base, arm_lengths,
                                   R_robot=0.9,
                                   dominant_centroid_log=None,
                                   smoothed_centroid_log=None,
                                   predicted_target_log=None):
    """
    Plot the Phase 4 evacuation scene with a 2-link planar arm.

    Shows all Phase 3 layers plus:
      - end-effector trajectory (solid magenta line)
      - arm links at sampled timesteps (faint dark red)
      - arm links at final configuration (bold dark red)
      - joint positions (filled circles)
      - influence-radius circle around final end-effector
    """
    N       = len(trajectories)
    colours = plt.cm.hsv(np.linspace(0, 0.85, max(N, 1)))
    base    = np.array(arm_base)

    fig, ax = plt.subplots(figsize=(9, 9))

    # Particle trajectories + start markers
    for i, traj in enumerate(trajectories):
        c = colours[i % len(colours)]
        ax.plot(traj[:, 0], traj[:, 1], color=c,
                linewidth=0.9, alpha=0.7, zorder=2)
        ax.scatter(*traj[0], color=c, edgecolors='black',
                   s=45, zorder=5, linewidths=0.6)

    # Arm snapshots at sampled timesteps (faint)
    samp = max(1, len(arm_angles_log) // 30)
    for ang in arm_angles_log[::samp]:
        joints = arm_forward_kinematics(base, ang, arm_lengths)
        xs = [j[0] for j in joints]
        ys = [j[1] for j in joints]
        ax.plot(xs, ys, color='darkred', linewidth=1.0, alpha=0.15, zorder=3)

    # Final arm configuration (bold)
    final_joints = arm_forward_kinematics(base, arm_angles_log[-1], arm_lengths)
    fxs = [j[0] for j in final_joints]
    fys = [j[1] for j in final_joints]
    ax.plot(fxs, fys, color='darkred', linewidth=3.0, alpha=0.9,
            zorder=7, label='Arm (final)')
    ax.scatter(fxs[:-1], fys[:-1], color='darkred', s=60,
               zorder=8, edgecolors='black', linewidths=0.8)
    ax.scatter(*base, color='black', s=80, marker='s',
               zorder=9, edgecolors='black', label='Arm base')

    # End-effector trajectory
    ax.plot(ee_traj[:, 0], ee_traj[:, 1],
            color='magenta', linewidth=1.4, linestyle='-',
            alpha=0.7, zorder=6, label='End-effector trajectory')
    ax.scatter(*ee_traj[-1], color='magenta', marker='X',
               s=160, zorder=9, edgecolors='black', linewidths=0.8,
               label='End-effector (final)')

    # Influence radius around final end-effector
    ax.add_patch(Circle(ee_traj[-1], R_robot,
                        color='magenta', fill=False,
                        linewidth=1.4, alpha=0.6, zorder=6))

    # Predicted target trail
    if predicted_target_log is not None:
        preds = [p for p in predicted_target_log if p is not None]
        if preds:
            pa = np.array(preds)
            ax.scatter(pa[:, 0], pa[:, 1],
                       color='orangered', s=20, alpha=0.45,
                       marker='D', zorder=4, linewidths=0,
                       label='Predicted target')

    # Smoothed centroid trail
    if smoothed_centroid_log is not None:
        smos = [s for s in smoothed_centroid_log if s is not None]
        if smos:
            sa = np.array(smos)
            ax.scatter(sa[:, 0], sa[:, 1],
                       color='cyan', s=16, alpha=0.45,
                       marker='^', zorder=4, linewidths=0,
                       label='Smoothed centroid (EMA)')

    # Exits
    for k, ex in enumerate(exits):
        ax.scatter(*ex, color='lime', s=200, zorder=8,
                   marker='*', edgecolors='black', linewidths=0.8)
        ax.annotate(f'Exit {k+1}', xy=ex,
                    xytext=(ex[0] + 0.15, ex[1] + 0.15),
                    fontsize=9, color='darkgreen', fontweight='bold')

    # Walls
    for wall in walls:
        ax.plot([wall['a'][0], wall['b'][0]],
                [wall['a'][1], wall['b'][1]],
                color='black', linewidth=3, zorder=4)

    legend_handles = [
        Line2D([0], [0], color='grey', linewidth=0.9,
               label=f'Particle trajectories (N={N})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markeredgecolor='black', markersize=7, label='Starts'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='lime',
               markeredgecolor='black', markersize=13, label='Exits'),
        Line2D([0], [0], color='darkred', linewidth=2.5, label='Arm (final)'),
        Line2D([0], [0], color='darkred', linewidth=1.0, alpha=0.3,
               label='Arm (history, sampled)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
               markersize=7, label='Arm base'),
        Line2D([0], [0], color='magenta', linewidth=1.4,
               label='End-effector trajectory'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='magenta',
               markeredgecolor='black', markersize=11, label='End-effector (final)'),
        Line2D([0], [0], color='magenta', linewidth=1.2, alpha=0.6,
               label=f'Influence radius (R={R_robot})'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='cyan',
               markersize=6, alpha=0.6, label='Smoothed centroid (EMA)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='orangered',
               markersize=6, alpha=0.7, label='Predicted target'),
        Line2D([0], [0], color='black', linewidth=3, label='Wall'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title(
        f"Phase 4 — 2-link planar arm IK  (N={N})\n"
        "End-effector tracks predicted dominant cluster (Jacobian-transpose IK)"
    )
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
