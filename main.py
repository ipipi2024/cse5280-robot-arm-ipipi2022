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
# 6. Particle-particle repulsion cost + gradient
# ─────────────────────────────────────────────

def particle_repulsion_cost(xi, xj, R_p, w_p):
    """
    Quadratic-band repulsion between two particles.

    C_repel = 0.5 * w_p * (R_p - d)²   if d < R_p
            = 0                          otherwise

    where d = ||xi - xj||.

    Same functional form as the wall penalty — just applied to the
    distance between two particles instead of a point and a segment.
    """
    d = np.linalg.norm(xi - xj)
    if d < R_p:
        return 0.5 * w_p * (R_p - d) ** 2
    return 0.0


def grad_particle_repulsion(xi, xj, R_p, w_p):
    """
    Analytic gradient of C_repel with respect to xi.

    Derivation (chain rule — identical structure to grad_wall_penalty):
        d       = ||xi - xj||

        dC/dd   = -w_p * (R_p - d)          (outer derivative)

        dd/dxi  = (xi - xj) / d             (gradient of Euclidean distance)

        grad_xi C_repel = -w_p * (R_p - d) * (xi - xj) / max(d, EPS)

    Direction: points from xj toward xi, i.e. pushes xi away from xj.
    By Newton's third law symmetry, the gradient w.r.t. xj is the negative.
    """
    d = np.linalg.norm(xi - xj)
    if d >= R_p:
        return np.zeros_like(xi)
    return -w_p * (R_p - d) * (xi - xj) / max(d, EPS)


def total_gradient_with_particles(i, positions, g, walls, R_p, w_p):
    """
    Full gradient for particle i, including inter-particle repulsion.

    grad C_i = grad_goal(x_i, g)
             + sum over all walls:  grad_wall_penalty(x_i, ...)
             + sum over all j ≠ i:  grad_particle_repulsion(x_i, x_j, R_p, w_p)

    Parameters
    ----------
    i         : index of the particle whose gradient we are computing
    positions : (N, 2) array of current particle positions (snapshot)
    g         : goal position
    walls     : list of wall dicts
    R_p       : particle influence radius
    w_p       : particle repulsion weight
    """
    xi   = positions[i]
    grad = total_gradient(xi, g, walls)          # goal + wall terms (existing)

    for j, xj in enumerate(positions):           # add repulsion from every other particle
        if j != i:
            grad = grad + grad_particle_repulsion(xi, xj, R_p, w_p)

    return grad


# ─────────────────────────────────────────────
# 7. Multi-particle simulation (independent)
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
# 8. Multi-particle simulation WITH repulsion
# ─────────────────────────────────────────────

def run_multi_particle_simulation_with_repulsion(starts, g, walls,
                                                 alpha=0.05, n_steps=800,
                                                 tol=0.05, R_p=0.8, w_p=60.0):
    """
    Gradient-descent simulation for N particles with pairwise repulsion.

    Each step is **synchronous**:
      1. Snapshot all current positions.
      2. Compute every particle's gradient using that snapshot.
      3. Apply all updates simultaneously.

    Why synchronous?
    ----------------
    If we updated particle 1 first and then used its new position to compute
    particle 2's gradient, particle 2 would react to a position that didn't
    exist at the start of the step. This breaks the physical symmetry:
    particle 1 pushes particle 2 based on position A, but particle 2 pushes
    back based on position B (already moved). Synchronous updates ensure both
    particles react to the same shared state.

    Parameters
    ----------
    starts : (N, 2) array — one start position per particle
    g      : shared goal position
    walls  : list of wall dicts
    alpha  : step size
    n_steps: maximum steps
    tol    : stop a particle when ||x - g|| < tol
    R_p    : particle influence radius (repulsion activates when d < R_p)
    w_p    : particle repulsion weight

    Returns
    -------
    trajectories : list of N arrays, each (T_i, 2)
    """
    N = len(starts)
    positions = np.array(starts, dtype=float)       # (N, 2) — current state
    active    = np.ones(N, dtype=bool)              # True while particle still moving

    # Store full trajectory for every particle
    trajectories = [[p.copy()] for p in positions]

    for _ in range(n_steps):
        if not np.any(active):
            break

        # ── Step 1: snapshot ───────────────────────────────────────────────
        snapshot = positions.copy()                 # all gradients read from here

        # ── Step 2: compute all gradients from the snapshot ────────────────
        grads = np.zeros_like(positions)
        for i in range(N):
            if active[i]:
                grads[i] = total_gradient_with_particles(
                    i, snapshot, g, walls, R_p, w_p
                )

        # ── Step 3: apply all updates simultaneously ───────────────────────
        for i in range(N):
            if active[i]:
                positions[i] = positions[i] - alpha * grads[i]
                trajectories[i].append(positions[i].copy())
                if np.linalg.norm(positions[i] - g) < tol:
                    active[i] = False               # this particle has converged

    return [np.array(t) for t in trajectories]


# ─────────────────────────────────────────────
# 9. Plotting
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
# 10. Multi-particle plot
# ─────────────────────────────────────────────

def plot_multi_particle_results(trajectories, starts, g, walls, repulsion=False):
    """
    Plot all particle trajectories together on one axes.

    Parameters
    ----------
    trajectories : list of (T_i, 2) arrays returned by run_multi_particle_simulation
    starts       : (N, 2) array or list used as start positions
    g            : goal position
    walls        : list of wall dicts
    """
    N = len(trajectories)
    # Use a continuous colormap so N=25 particles all get distinct colours
    colours = plt.cm.hsv(np.linspace(0, 0.85, max(N, 1)))

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, traj in enumerate(trajectories):
        colour = colours[i % len(colours)]
        ax.plot(traj[:, 0], traj[:, 1], color=colour, linewidth=0.9,
                alpha=0.75, zorder=2)
        # Mark each start with a small filled circle in the same colour
        ax.scatter(*traj[0], color=colour, edgecolors='black',
                   s=50, zorder=5, linewidths=0.6)

    # Goal
    ax.scatter(*g, color='red', s=140, zorder=6, marker='*')

    # Walls
    for wall in walls:
        ax.plot([wall['a'][0], wall['b'][0]],
                [wall['a'][1], wall['b'][1]],
                color='black', linewidth=3, zorder=4)

    legend_handles = [
        Line2D([0], [0], color='grey', linewidth=0.9,
               label=f'Trajectories (N={N})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markeredgecolor='black', markersize=7, label='Start positions'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=14, label='Goal'),
        Line2D([0], [0], color='black', linewidth=3, label='Wall'),
    ]
    ax.legend(handles=legend_handles, loc='upper left')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title(f"Multi-particle gradient descent  (N={len(trajectories)})\n"
                 f"{'Inter-particle repulsion enabled' if repulsion else 'Each particle simulated independently — no inter-particle forces'}")
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 11. Cost field + gradient vector visualisation
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
# 12. Main
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
