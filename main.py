import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EPS = 1e-8   # safeguard against division by zero when x lies on the wall


# ─────────────────────────────────────────────
# 1. Goal cost + analytic gradient
# ─────────────────────────────────────────────

def goal_cost(x, g):
    """C_goal = 0.5 * ||x - g||²"""
    diff = x - g
    return 0.5 * np.dot(diff, diff)


def grad_goal(x, g):
    """
    Analytic gradient of C_goal.

    Derivation:
        C_goal = 0.5 * (x - g)·(x - g)
        dC/dx  = (x - g)

    Descending this gradient moves x toward g.
    """
    return x - g


# ─────────────────────────────────────────────
# 2. Point-to-segment: distance + closest point
# ─────────────────────────────────────────────

def point_to_segment(x, a, b):
    """
    Returns (d, p) where:
      d = shortest distance from x to segment [a, b]
      p = closest point on the segment to x

    Steps:
      1. Project (x - a) onto (b - a) to get scalar t.
      2. Clamp t to [0, 1] so p stays on the segment.
      3. p = a + t*(b - a),  d = ||x - p||
    """
    v = b - a
    t = np.dot(x - a, v) / np.dot(v, v)
    t = np.clip(t, 0.0, 1.0)
    p = a + t * v
    d = np.linalg.norm(x - p)
    return d, p


# ─────────────────────────────────────────────
# 3. Wall penalty + analytic gradient
# ─────────────────────────────────────────────

def wall_cost(x, a, b, R, w):
    """
    C_wall = 0.5 * w * (R - d)²   if d < R
           = 0                      otherwise
    """
    d, _ = point_to_segment(x, a, b)
    if d < R:
        return 0.5 * w * (R - d) ** 2
    return 0.0


def grad_wall_penalty(x, a, b, R, w):
    """
    Analytic gradient of C_wall via the chain rule.

    Derivation:
        d      = ||x - p||          (distance to closest point p on segment)

        dC/dd  = -w * (R - d)       (outer derivative of the quadratic)

        dd/dx  = (x - p) / d        (gradient of Euclidean distance)

                 This formula holds whether p is in the segment interior
                 (t clamped) or at an endpoint — in both cases p is fixed
                 relative to the perpendicular direction of x, so the
                 along-segment component vanishes.

        Chain rule:
        grad C_wall = dC/dd * dd/dx
                    = -w * (R - d) * (x - p) / max(d, eps)   if d < R
                    = 0                                        otherwise
    """
    d, p = point_to_segment(x, a, b)
    if d >= R:
        return np.zeros_like(x)
    return -w * (R - d) * (x - p) / max(d, EPS)


# ─────────────────────────────────────────────
# 4. Total cost + total gradient
# ─────────────────────────────────────────────

def total_cost(x, g, walls):
    """Sum of goal cost and all wall penalties."""
    cost = goal_cost(x, g)
    for wall in walls:
        cost += wall_cost(x, wall['a'], wall['b'], wall['R'], wall['w'])
    return cost


def total_gradient(x, g, walls):
    """
    Analytic gradient of the total cost:
        grad C = grad C_goal + sum_i grad C_wall_i
    """
    grad = grad_goal(x, g)
    for wall in walls:
        grad = grad + grad_wall_penalty(x, wall['a'], wall['b'],
                                        wall['R'], wall['w'])
    return grad


# ─────────────────────────────────────────────
# 5. Simulation loop
# ─────────────────────────────────────────────

def run_simulation(x0, g, walls, alpha=0.001, n_steps=500):
    """
    Gradient descent:   x <- x - alpha * grad C(x)

    Uses the fully analytic gradient — no finite differences.
    Wall avoidance emerges purely from cost minimisation.
    """
    x = x0.copy()
    trajectory = [x.copy()]

    for _ in range(n_steps):
        grad = total_gradient(x, g, walls)
        x = x - alpha * grad
        trajectory.append(x.copy())

    return np.array(trajectory)


# ─────────────────────────────────────────────
# 6. Multi-particle simulation
# ─────────────────────────────────────────────

def run_multi_particle_simulation(starts, g, walls,
                                  alpha=0.05, n_steps=800, tol=0.05):
    """
    Run independent gradient-descent simulations for N particles.

    Each particle sees the same walls and goal and is simulated separately —
    no particle-particle interaction of any kind.

    Parameters
    ----------
    starts  : (N, 2) array or list of 2-element arrays — one start per particle
    g       : goal position shared by all particles
    walls   : list of wall dicts (same format as single-particle version)
    alpha   : gradient descent step size
    n_steps : maximum number of steps per particle
    tol     : early-stop when ||x - g|| < tol (particle has reached the goal)

    Returns
    -------
    trajectories : list of N arrays, each of shape (T_i, 2)
                   T_i may be shorter than n_steps+1 if the particle converged.
    """
    trajectories = []

    for x0 in starts:
        x = np.array(x0, dtype=float)
        traj = [x.copy()]

        for _ in range(n_steps):
            if np.linalg.norm(x - g) < tol:
                break                          # particle reached the goal
            grad = total_gradient(x, g, walls)
            x = x - alpha * grad
            traj.append(x.copy())

        trajectories.append(np.array(traj))

    return trajectories


# ─────────────────────────────────────────────
# 7. Plotting
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
# 8. Multi-particle plot
# ─────────────────────────────────────────────

def plot_multi_particle_results(trajectories, starts, g, walls):
    """
    Plot all particle trajectories together on one axes.

    Parameters
    ----------
    trajectories : list of (T_i, 2) arrays returned by run_multi_particle_simulation
    starts       : (N, 2) array or list used as start positions
    g            : goal position
    walls        : list of wall dicts
    """
    # One colour per particle, cycling through a qualitative palette
    colours = plt.cm.tab10(np.linspace(0, 1, max(len(trajectories), 1)))

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, traj in enumerate(trajectories):
        colour = colours[i % len(colours)]
        ax.plot(traj[:, 0], traj[:, 1], color=colour, linewidth=1.4,
                alpha=0.85, zorder=2)
        # Mark each start with a small filled circle in the same colour
        ax.scatter(*traj[0], color=colour, edgecolors='black',
                   s=80, zorder=5, linewidths=0.8)

    # Goal
    ax.scatter(*g, color='red', s=140, zorder=6, marker='*')

    # Walls
    for wall in walls:
        ax.plot([wall['a'][0], wall['b'][0]],
                [wall['a'][1], wall['b'][1]],
                color='black', linewidth=3, zorder=4)

    legend_handles = [
        Line2D([0], [0], color='steelblue', linewidth=1.4,
               label=f'Particle trajectories (N={len(trajectories)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markeredgecolor='black', markersize=9, label='Start positions'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=14, label='Goal'),
        Line2D([0], [0], color='black', linewidth=3, label='Wall'),
    ]
    ax.legend(handles=legend_handles)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title(f"Multi-particle gradient descent  (N={len(trajectories)})\n"
                 "Each particle simulated independently — no inter-particle forces")
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 9. Cost field + gradient vector visualisation
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

    What each layer shows
    ─────────────────────
    contourf   : total cost value — bright = high cost (near walls or far from
                 goal), dark = low cost (near goal, away from walls)
    quiver     : negative gradient i.e. the direction the particle is pushed
                 at each point. Arrows point downhill on the cost surface.
    """
    xs = np.linspace(0, 10, grid_n)
    ys = np.linspace(0, 10, grid_n)
    XX, YY = np.meshgrid(xs, ys)

    Z = np.zeros_like(XX)
    U = np.zeros_like(XX)   # -grad[0]  (x component of descent direction)
    V = np.zeros_like(XX)   # -grad[1]  (y component of descent direction)

    for i in range(grid_n):
        for j in range(grid_n):
            pos     = np.array([XX[i, j], YY[i, j]])
            Z[i, j] = total_cost(pos, g, walls)
            grad    = total_gradient(pos, g, walls)
            U[i, j] = -grad[0]
            V[i, j] = -grad[1]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Cost contours — clip so wall spikes don't collapse the colour scale
    Z_display = np.clip(Z, 0, np.percentile(Z, 95))
    cf = ax.contourf(XX, YY, Z_display, levels=40, cmap='plasma')
    plt.colorbar(cf, ax=ax, label='Total cost (clipped at 95th percentile)')

    # Descent direction arrows — subsampled so they stay readable
    step = max(1, grid_n // 20)
    ax.quiver(XX[::step, ::step], YY[::step, ::step],
              U[::step, ::step],  V[::step, ::step],
              color='white', alpha=0.6, width=0.003, headwidth=4)

    # Walls
    for wall in walls:
        ax.plot([wall['a'][0], wall['b'][0]],
                [wall['a'][1], wall['b'][1]],
                color='black', linewidth=3, zorder=4)

    # Start and goal
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

    # Optional trajectory overlay
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
# 10. Main
# ─────────────────────────────────────────────

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
    # Particles start from different positions around the n-shape.
    # Each is simulated independently — same walls, same goal, no interaction.
    starts = np.array([
        [2.0, 2.0],   # lower-left  (must go around left leg)
        [8.0, 2.0],   # lower-right (must go around right leg)
        [5.5, 1.5],   # bottom-centre (aligned with the gap)
        [1.5, 5.0],   # left side
        [8.5, 5.0],   # right side
    ])

    trajectories = run_multi_particle_simulation(
        starts, g, walls, alpha=0.05, n_steps=800, tol=0.05
    )
    plot_multi_particle_results(trajectories, starts, g, walls)
