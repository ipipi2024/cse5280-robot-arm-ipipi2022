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
# 6. Plotting
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
# 7. Cost field + gradient vector visualisation
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
# 8. Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Start is outside and offset so the direct path to the goal is blocked
    # by the left leg of the n-shape.  The particle must go around the outside
    # and enter through the open bottom gap.
    x0 = np.array([2.0, 2.0])
    g  = np.array([5.5, 7.0])

    # ── Outer square boundary [0,10]×[0,10] ──────────────────────────────────
    # Same penalty framework as every other wall — no special boundary logic.
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
    # Legs reach down to y=3.0 — well below the goal at y=7.0.
    # The gap width is 4 units (x: 3.5→7.5).  With R=1.0 the influence bands
    # consume 1 unit on each side, leaving ~2 units of navigable gap width.
    #
    # Start (2.0, 2.0) is to the lower-left.  The direct path to the goal
    # crosses the left leg, so the particle is forced to detour around the
    # outside of the n-shape and enter through the bottom opening.
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

    trajectory = run_simulation(x0, g, walls, alpha=0.05, n_steps=800)
    plot_results(trajectory, x0, g, walls)

    # Debug: show the cost landscape and descent vectors, with trajectory overlaid.
    # Use a coarser grid (grid_n=40) for speed, or increase to 80+ for detail.
    plot_cost_field_and_vectors(x0, g, walls, trajectory=trajectory, grid_n=60)
