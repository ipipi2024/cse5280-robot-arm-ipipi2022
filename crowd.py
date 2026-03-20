# crowd.py
# Core crowd-simulation physics: cost functions, analytic gradients, wall
# penalties, particle repulsion, and simulation loops.
# Nothing in this module knows about robots or arm kinematics.

import numpy as np
from config import EPS


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
# 4. Wall penalty + analytic gradient
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
# 5. Total cost + total gradient (single goal)
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
    grad = total_gradient(xi, g, walls)          # goal + wall terms

    for j, xj in enumerate(positions):
        if j != i:
            grad = grad + grad_particle_repulsion(xi, xj, R_p, w_p)

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
# 7. Single-particle simulation loop
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
# 8. Multi-particle simulation (independent)
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
                break
            grad = total_gradient(x, g, walls)
            x = x - alpha * grad
            traj.append(x.copy())

        trajectories.append(np.array(traj))

    return trajectories


# ─────────────────────────────────────────────
# 9. Multi-particle simulation WITH repulsion
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
    positions = np.array(starts, dtype=float)
    active    = np.ones(N, dtype=bool)
    trajectories = [[p.copy()] for p in positions]

    for _ in range(n_steps):
        if not np.any(active):
            break

        snapshot = positions.copy()

        grads = np.zeros_like(positions)
        for i in range(N):
            if active[i]:
                grads[i] = total_gradient_with_particles(
                    i, snapshot, g, walls, R_p, w_p
                )

        for i in range(N):
            if active[i]:
                positions[i] = positions[i] - alpha * grads[i]
                trajectories[i].append(positions[i].copy())
                if np.linalg.norm(positions[i] - g) < tol:
                    active[i] = False

    return [np.array(t) for t in trajectories]


# ─────────────────────────────────────────────
# 10. Evacuation simulation (soft-min + repulsion)
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
                if np.min(np.linalg.norm(exits - positions[i], axis=1)) < tol:
                    active[i] = False

    return [np.array(t) for t in trajs]
