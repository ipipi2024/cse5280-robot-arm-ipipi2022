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
# 7. Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    x0 = np.array([1.0, 1.0])
    g  = np.array([9.0, 9.0])

    # Outer square boundary [0,10]x[0,10] — same penalty framework as interior walls.
    # The particle stays inside because crossing into the influence band (R) raises cost.
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

    # Interior obstacle
    interior_walls = [
        {'a': np.array([4.0, 2.0]),
         'b': np.array([4.0, 7.0]),
         'R': 1.5,
         'w': 80.0},
    ]

    walls = boundary_walls + interior_walls

    trajectory = run_simulation(x0, g, walls, alpha=0.05, n_steps=400)
    plot_results(trajectory, x0, g, walls)
