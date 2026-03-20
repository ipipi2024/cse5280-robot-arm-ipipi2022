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
# 2. Soft-min goal cost + gradient (two exits)
# ─────────────────────────────────────────────
#
# Why soft-min instead of hard min?
# ──────────────────────────────────
# hard min:  C = min(C1, C2)
#   - non-differentiable at the switching surface where C1 == C2
#   - gradient jumps discontinuously → particle snaps between exits, causing
#     oscillation or getting stuck at the indifference boundary
#
# soft-min:  C = -(1/beta) * log( exp(-beta*C1) + exp(-beta*C2) )
#   - fully smooth and differentiable everywhere
#   - gradient is a weighted average of the two exit gradients
#   - weights are proportional to exp(-beta*Ci): the cheaper exit gets more pull
#   - as beta → ∞ it approaches the hard min; beta=4 gives a smooth but
#     sharp-enough preference that nearby particles clearly favour the closer exit

def softmin_goal_cost(x, exits, beta):
    """
    Soft-min over quadratic costs to each exit.

        C_i   = 0.5 * ||x - g_i||²
        C_soft = -(1/beta) * log( sum_i exp(-beta * C_i) )

    Uses the log-sum-exp trick for numerical stability:
        shift the exponents by max(-beta*C_i) before summing so no value
        overflows, then correct the log afterwards.

    Parameters
    ----------
    x     : current position (2,)
    exits : (K, 2) array of exit positions
    beta  : sharpness parameter (higher = closer to hard min)
    """
    costs  = np.array([0.5 * np.dot(x - g, x - g) for g in exits])
    logits = -beta * costs
    shift  = np.max(logits)                        # numerical stability
    log_sum = shift + np.log(np.sum(np.exp(logits - shift)))
    return -(1.0 / beta) * log_sum


def grad_softmin_goal(x, exits, beta):
    """
    Analytic gradient of the soft-min goal cost.

    Derivation:
        Let C_i = 0.5 ||x - g_i||²,   grad C_i = x - g_i

        Differentiating C_soft w.r.t. x:

            d/dx [-(1/beta) log Z]  where  Z = sum_i exp(-beta * C_i)

            = -(1/beta) * (1/Z) * sum_i [ -beta * grad_C_i * exp(-beta * C_i) ]

            = sum_i  [ exp(-beta * C_i) / Z ]  *  grad_C_i

            = sum_i  w_i * (x - g_i)

        where w_i = softmax(-beta * C_i) are the exit weights.
        w_i is large when C_i is small (x is close to exit i).

    This is a weighted average of the individual goal gradients — no
    discontinuity at the indifference surface between exits.

    Parameters
    ----------
    x     : current position (2,)
    exits : (K, 2) array of exit positions
    beta  : sharpness parameter
    """
    costs  = np.array([0.5 * np.dot(x - g, x - g) for g in exits])
    logits = -beta * costs
    shift  = np.max(logits)
    exp_vals = np.exp(logits - shift)
    weights  = exp_vals / np.sum(exp_vals)         # softmax = exit weights

    grad = np.zeros_like(x)
    for w, g in zip(weights, exits):
        grad = grad + w * (x - g)
    return grad


# ─────────────────────────────────────────────
# 3. Point-to-segment: distance + closest point
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


def total_gradient_softmin(x, exits, walls, beta):
    """
    Total gradient using the soft-min goal term (multi-exit, no repulsion).

        grad C = grad_softmin_goal(x, exits, beta)
               + sum over walls: grad_wall_penalty(x, ...)
    """
    grad = grad_softmin_goal(x, exits, beta)
    for wall in walls:
        grad = grad + grad_wall_penalty(x, wall['a'], wall['b'],
                                        wall['R'], wall['w'])
    return grad


def total_gradient_with_particles_softmin(i, positions, exits, walls,
                                          R_p, w_p, beta):
    """
    Full gradient for particle i using soft-min goal + repulsion.

        grad C_i = grad_softmin_goal(x_i, exits, beta)
                 + sum over walls:    grad_wall_penalty(x_i, ...)
                 + sum over j ≠ i:   grad_particle_repulsion(x_i, x_j, R_p, w_p)
    """
    xi   = positions[i]
    grad = total_gradient_softmin(xi, exits, walls, beta)
    for j, xj in enumerate(positions):
        if j != i:
            grad = grad + grad_particle_repulsion(xi, xj, R_p, w_p)
    return grad


# ─────────────────────────────────────────────
# 6. Simulation loop
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
# 9. Evacuation simulation (soft-min + repulsion)
# ─────────────────────────────────────────────

def run_evacuation_simulation(starts, exits, walls, beta=4.0,
                              alpha=0.04, n_steps=1000, tol=0.05,
                              R_p=0.6, w_p=30.0):
    """
    Multi-particle gradient descent toward the soft-min of two (or more) exits,
    with pairwise repulsion and synchronous updates.

    Each particle is attracted to the cheaper exit — not by a hard if/else
    switch, but because the soft-min gradient smoothly weights the pull from
    each exit by exp(-beta * C_i). Nearby particles naturally spread across
    exits, preventing crowding, since repulsion pushes them apart while the
    goal pull steers them toward whichever exit is least costly from their
    current position.

    A particle is considered to have reached an exit when it is within `tol`
    of ANY exit.

    Parameters
    ----------
    starts : (N, 2) array — start positions
    exits  : (K, 2) array — exit positions
    walls  : list of wall dicts
    beta   : soft-min sharpness (higher = sharper exit preference)
    alpha  : step size
    n_steps: maximum steps
    tol    : convergence threshold (distance to nearest exit)
    R_p    : particle repulsion radius
    w_p    : particle repulsion weight

    Returns
    -------
    trajectories : list of N arrays, each (T_i, 2)
    """
    exits     = np.array(exits)
    N         = len(starts)
    positions = np.array(starts, dtype=float)
    active    = np.ones(N, dtype=bool)
    trajs     = [[p.copy()] for p in positions]

    for _ in range(n_steps):
        if not np.any(active):
            break

        snapshot = positions.copy()

        grads = np.zeros_like(positions)
        for i in range(N):
            if active[i]:
                grads[i] = total_gradient_with_particles_softmin(
                    i, snapshot, exits, walls, R_p, w_p, beta
                )

        for i in range(N):
            if active[i]:
                positions[i] = positions[i] - alpha * grads[i]
                trajs[i].append(positions[i].copy())
                # Stop when particle reaches any exit
                if np.min(np.linalg.norm(exits - positions[i], axis=1)) < tol:
                    active[i] = False

    return [np.array(t) for t in trajs]


# ─────────────────────────────────────────────
# 10. Plotting
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

def plot_multi_particle_results(trajectories, starts, g, walls,
                                repulsion=False, exits=None):
    """
    Plot all particle trajectories together on one axes.

    Parameters
    ----------
    trajectories : list of (T_i, 2) arrays returned by run_multi_particle_simulation
    starts       : (N, 2) array or list used as start positions
    g            : single goal position (pass None when using exits)
    walls        : list of wall dicts
    repulsion    : bool — controls subtitle text
    exits        : optional (K, 2) array — if provided, drawn instead of g
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

    # Goal(s)
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

    # Walls
    for wall in walls:
        ax.plot([wall['a'][0], wall['b'][0]],
                [wall['a'][1], wall['b'][1]],
                color='black', linewidth=3, zorder=4)

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
# 13. Cost field + gradient vector visualisation
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


# ═════════════════════════════════════════════
# PHASE 1 — Robot interference agent
# ═════════════════════════════════════════════
#
# Why this is a valid Phase 1 baseline:
#   The robot is represented as a moving point obstacle.  Particles
#   already treat walls and each other as repulsion sources via the same
#   quadratic-band cost; adding the robot just plugs into that same
#   framework.  The robot tracks where evacuation flow is accumulating
#   near exits and moves there proportionally — a minimal but physically
#   meaningful interference strategy that can be extended later with
#   clustering, prediction, and inverse kinematics.

# ─────────────────────────────────────────────
# 14. Robot obstacle cost + analytic gradient
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
# 15. Robot detection and targeting
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
        return current.copy()               # cold start — initialise to first observation
    return (1.0 - lambda_smooth) * prev_smoothed + lambda_smooth * current


def predict_cluster_target(smoothed_centroid, prev_smoothed, horizon, room_bounds=(0, 10)):
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
        return smoothed_centroid.copy()     # no history or no look-ahead

    velocity  = smoothed_centroid - prev_smoothed   # one-step displacement (smoothed)
    raw       = smoothed_centroid + horizon * velocity
    lo, hi    = room_bounds
    return np.clip(raw, lo, hi)             # stay inside [0, 10] × [0, 10]


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
        return prev_target, None, None, None, None   # no flow — hold

    pts = positions[near]
    if len(pts) < k:
        mean_pos = pts.mean(axis=0)
        return mean_pos, None, mean_pos, mean_pos, mean_pos  # too few — plain mean

    centroids, labels = kmeans(pts, k=k)
    sizes         = np.bincount(labels, minlength=k)
    dominant      = np.argmax(sizes)
    dom_centroid  = centroids[dominant]

    smoothed  = smooth_centroid(dom_centroid, prev_smoothed, lambda_smooth)
    predicted = predict_cluster_target(smoothed, prev_smoothed, horizon)
    return predicted, centroids, dom_centroid, smoothed, predicted


# ─────────────────────────────────────────────
# 16. Evacuation simulation with robot (Phase 1)
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

    # Robot state
    robot_pos    = np.array(robot_start if robot_start is not None
                            else [5.0, 5.0], dtype=float)
    robot_target = robot_pos.copy()
    robot_traj            = [robot_pos.copy()]
    robot_targets_log     = [robot_target.copy()]
    cluster_centroids_log = [None]
    dominant_centroid_log = [None]
    smoothed_centroid_log = [None]
    predicted_target_log  = [None]
    prev_smoothed         = None   # EMA state — updated each step clustering fires

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
            prev_smoothed = smoothed              # advance EMA state for next step

        # ── b. Move robot toward target (proportional control) ────────────
        robot_pos = robot_pos + robot_alpha * (robot_target - robot_pos)

        # ── c. Snapshot all positions for synchronous gradient computation ─
        snapshot = positions.copy()

        # ── d. Compute gradients (soft-min goal + walls + repulsion + robot)─
        grads = np.zeros_like(positions)
        for i in range(N):
            if active[i]:
                xi = snapshot[i]
                # Existing terms: soft-min goal + walls + inter-particle
                grads[i] = total_gradient_with_particles_softmin(
                    i, snapshot, exits, walls, R_p, w_p, beta
                )
                # New term: robot as dynamic obstacle
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


# ─────────────────────────────────────────────
# 17. Phase 1 visualisation
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
    step = max(1, len(robot_traj) // 40)   # draw at most ~40 circles
    for pos in robot_traj[::step]:
        circle = plt.Circle(pos, R_robot, color='magenta',
                            fill=False, linewidth=0.6, alpha=0.18, zorder=3)
        ax.add_patch(circle)
    # Final position gets a slightly more visible circle
    circle_final = plt.Circle(robot_traj[-1], R_robot, color='magenta',
                               fill=False, linewidth=1.2, alpha=0.55, zorder=6)
    ax.add_patch(circle_final)

    # Robot trajectory
    ax.plot(robot_traj[:, 0], robot_traj[:, 1],
            color='magenta', linewidth=1.6, linestyle='--',
            zorder=6, label='Robot trajectory')
    # Robot final position
    ax.scatter(*robot_traj[-1], color='magenta', marker='X',
               s=160, zorder=7, edgecolors='black', linewidths=0.8,
               label='Robot (final)')
    # Robot target trail
    ax.scatter(robot_targets[:, 0], robot_targets[:, 1],
               color='grey', s=6, alpha=0.4, zorder=3,
               label='Robot target trail')

    # Cluster centroids trail — every centroid from every timestep where
    # clustering fired, sampled for readability.
    if cluster_centroids_log is not None:
        all_centroids = [c for c in cluster_centroids_log if c is not None]
        if all_centroids:
            step = max(1, len(all_centroids) // 60)
            for c_arr in all_centroids[::step]:
                ax.scatter(c_arr[:, 0], c_arr[:, 1],
                           color='black', s=18, alpha=0.35,
                           marker='o', zorder=4, linewidths=0)

    # Raw dominant, smoothed, predicted centroid trails + connector arrows
    # Arrows go from smoothed centroid → predicted target to show the prediction
    # offset produced by EMA velocity extrapolation.
    have_dom  = dominant_centroid_log is not None
    have_smo  = smoothed_centroid_log is not None
    have_pred = predicted_target_log is not None

    if have_dom or have_smo or have_pred:
        # Collect steps where at least dominant fired
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
                        # faint arrow: smoothed → predicted (shows prediction offset)
                        ax.annotate('', xy=pred, xytext=smo,
                                    arrowprops=dict(arrowstyle='->', color='orangered',
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
    ax.set_title(f"Phase 3 — Robot interference agent with clustering + prediction  (N={N})\n"
                 "Robot targets predicted dominant cluster (k-means k=2, horizon=H steps)")
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 18. Main
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
