# CSE 5280 - Robot Arm: Gradient Descent Particle Simulation

## Overview

A 2D multi-particle simulation where particles navigate from start positions
toward goal(s) while avoiding wall obstacles inside a bounded room. Motion
emerges purely from **cost minimisation** via gradient descent — there is no
collision detection, no path-finding (no A*), and no hard routing logic.

Four simulation modes are supported:

| Mode | Goal term | Repulsion | Robot |
|------|-----------|-----------|-------|
| Independent | Single goal | No | No |
| Pairwise repulsion | Single goal | Yes, synchronous | No |
| Evacuation (soft-min) | Two exits via soft-min | Yes, synchronous | No |
| Phase 1 — robot interference | Two exits via soft-min | Yes, synchronous | Yes (centroid) |
| Phase 2 — clustering targeting | Two exits via soft-min | Yes, synchronous | Yes (k-means) |
| Phase 3 — prediction | Two exits via soft-min | Yes, synchronous | Yes (k-means + prediction) |

All gradients are derived analytically from first principles.

## Cost Function

### Single-goal term

```
C_goal(x) = 0.5 * ||x - g||²
grad C_goal = x - g
```

### Soft-min goal term (two exits)

Each particle is smoothly attracted toward the cheaper of two (or more) exits.

```
C_i   = 0.5 * ||x - g_i||²        (per-exit quadratic cost)

C_soft(x) = -(1/beta) * log( sum_i exp(-beta * C_i) )
```

**Why soft-min instead of hard min?**

`hard min = min(C1, C2)` is non-differentiable where `C1 == C2` — the gradient
jumps discontinuously at the indifference surface, causing oscillation or
particles getting stuck. Soft-min is smooth everywhere and its gradient is a
**weighted average** of the per-exit gradients:

```
w_i      = exp(-beta * C_i) / sum_j exp(-beta * C_j)   (softmax weights)

grad C_soft = sum_i  w_i * (x - g_i)
```

`w_i` is large when `C_i` is small — the closer exit gets more pull. As
`beta → ∞` the result approaches the hard min. `beta = 4.0` gives a sharp but
smooth preference. The log-sum-exp trick is used internally for numerical
stability.

### Wall penalty term (per wall segment)

```
C_wall(x) = 0.5 * w * (R - d(x))²   if d(x) < R,   else 0

grad C_wall = -w * (R - d) * (x - p) / max(d, eps)   if d < R,   else 0
```

`p` = closest point on the segment, `d` = distance to segment.

### Particle-particle repulsion term

Same quadratic-band form as the wall penalty, applied to pairwise distances.

```
C_repel(xi, xj) = 0.5 * w_p * (R_p - d_ij)²   if d_ij < R_p,   else 0

grad_xi C_repel = -w_p * (R_p - d_ij) * (xi - xj) / max(d_ij, eps)
```

Pushes `xi` away from `xj`. By symmetry the gradient w.r.t. `xj` is the
negative — equal and opposite repulsion.

## Distance: Point to Segment

```
t = dot(x - a, b - a) / dot(b - a, b - a)   # project onto line
t = clamp(t, 0, 1)                            # constrain to segment
p = a + t * (b - a)
d = ||x - p||
```

## Simulation Loops

### Single particle

```
x = x - alpha * grad C(x)
```

### Multiple particles — independent

```
for each x0:
    for each step:
        if ||x - g|| < tol: break
        x = x - alpha * total_gradient(x, g, walls)
```

### Multiple particles — synchronous (repulsion or evacuation)

```
snapshot = positions.copy()                      # 1. freeze current state
grads[i] = total_gradient_with_particles*(...)   # 2. all from snapshot
positions[i] -= alpha * grads[i]                 # 3. all updates together
```

**Why synchronous?** All gradients read the same position snapshot before any
particle moves, preserving the physical symmetry of pairwise repulsion.

### Evacuation convergence criterion

A particle stops when it reaches **any** exit:

```
if min_k ||x - exit_k|| < tol: stop
```

## Environment Layouts

### n-shape scenario (single goal)

```
(0,10)──────────────────────────────(10,10)
  │                                     │
  │    (3.5,8.5)──────────(7.5,8.5)     │
  │        │   goal(5.5,7.0)  │         │
  │        │                  │         │
  │    (3.5,3.0)          (7.5,3.0)     │
  │             ↑ open gap              │
  │   N=25 starts scattered outside     │
(0,0)──────────────────────────────(10,0)
```

Particles must detour around the n-shape legs and enter through the bottom gap.

### Evacuation scenario (soft-min, two exits)

```
(0,10)──────────────────────────────(10,10)
  │   N=25 starts   y∈[3,9]            │
  │   x∈[1,9]  scattered               │
  │                                     │
  │                                     │
  │                                     │
  │                                     │
  │  ★ Exit 1              Exit 2 ★    │
(0,0)──────────────────────────────(10,0)
      (2,1.2)              (8,1.2)
```

No interior walls. Particles near the left naturally prefer Exit 1; particles
near the right prefer Exit 2. Repulsion spreads load across both exits.

## Particle Start Positions

### n-shape runs (`np.random.seed(42)`, N=25)

| Region | x range | y range | Count |
|--------|---------|---------|-------|
| Left   | [1, 3]     | [2, 8]   | N // 3 |
| Right  | [7, 9]     | [2, 8]   | N // 3 |
| Bottom | [3.5, 7.5] | [1, 2.5] | remainder |

### Evacuation run (`np.random.seed(42)`, N=25)

Uniform random in `x∈[1,9], y∈[3,9]`.

## Parameters

| Parameter    | Value | Description |
|--------------|-------|-------------|
| `boundary_R` | 1.0   | Influence radius for outer boundary walls |
| `boundary_w` | 80.0  | Penalty weight for outer boundary walls |
| `interior_R` | 1.0   | Influence radius for n-shape walls |
| `interior_w` | 120.0 | Penalty weight for n-shape walls |
| `R_p`        | 0.6   | Particle repulsion radius |
| `w_p`        | 30.0  | Particle repulsion weight |
| `alpha`      | 0.04  | Gradient descent step size |
| `n_steps`    | 1000  | Maximum steps per particle |
| `tol`        | 0.05 / 0.15 | Convergence threshold (single goal / evacuation) |
| `beta`       | 4.0   | Soft-min sharpness — higher → sharper exit preference |

**Tuning `beta`:** low values (≈1) give a very smooth blend between exits;
high values (≈10+) approach a hard switch and may introduce gradient instability
near the indifference surface.

## Phase 1 — Robot Interference Agent

### Concept

The robot is a **moving point obstacle** in the same 2D environment. It has no
arms, no IK, no clustering, and no prediction — this is intentionally the
minimal baseline. The robot:

1. Detects particles that are currently near an exit (within `detection_radius`)
2. Moves toward the centroid of those particles (proportional control)
3. Is felt by all particles as a dynamic repulsive obstacle

Particles treat the robot exactly like they treat each other or a wall —
via the same quadratic-band cost, no special rules.

### Robot obstacle cost + gradient

```
d         = ||x - robot_pos||

C_robot   = 0.5 * w_robot * (R_robot - d)²   if d < R_robot,   else 0

grad_xi C_robot = -w_robot * (R_robot - d) * (x - robot_pos) / max(d, eps)
```

Identical derivation to the particle repulsion term. The robot position is
fixed for the full step (read before any particle moves), consistent with
synchronous updates.

### Robot motion (Phase 1)

```
target    = mean position of active particles within detection_radius of any exit
           (if none detected, hold previous target)

robot_pos = robot_pos + robot_alpha * (target - robot_pos)
```

### Per-step order

```
1. detect near-exit particles  →  update robot target
2. move robot (proportional)
3. snapshot particle positions
4. compute all gradients from snapshot:
       soft-min goal + walls + inter-particle repulsion + robot repulsion
5. apply all particle updates simultaneously
```

### Why this is a valid Phase 1 baseline

The robot reacts only to where particles **currently are** near exits — no
grouping of flow streams, no forecasting of future positions, no arm
kinematics. It is physically meaningful (it interferes with the highest-density
evacuation flow) while remaining the simplest possible extension. Clustering,
prediction, and IK can all be layered on top without changing this core loop.

### Robot parameters

| Parameter          | Value | Description |
|--------------------|-------|-------------|
| `robot_start`      | [5,5] | Initial robot position (room centre) |
| `robot_alpha`      | 0.08  | Proportional gain toward target |
| `detection_radius` | 2.0   | Radius around each exit for flow detection |
| `R_robot`          | 0.9   | Robot obstacle influence radius |
| `w_robot`          | 50.0  | Robot obstacle penalty weight |

---

## Phase 2 — Clustering-Based Targeting

### Concept

Phase 2 replaces the global centroid heuristic with **k-means clustering** of
near-exit particles. The robot targets the centroid of the *dominant* (largest)
cluster instead of the mean of all detected particles.

**Why clustering?**

With two exits the global mean drifts to the midpoint between the two exit
flows and never truly intercepts either one. K-means separates the flows into
per-exit groups; targeting the largest cluster sends the robot into the densest
stream of approaching particles — the one most worth disrupting.

### k-means algorithm

```
Initialise k centroids by sampling k points at random (fixed seed)

repeat n_iter times:
    labels[i] = argmin_c  ||x_i - centroid_c||      (assignment)
    centroid_c = mean of all x_i with labels[i] == c  (update)
    # empty cluster → centroid unchanged (avoids NaN)

return centroids, labels
```

Lloyd's algorithm with `n_iter=10` and `k=2` (one cluster per exit).
Fixed random seed (`seed=0`) makes results deterministic across runs.

### Dominant-cluster targeting

```
centroids, labels = kmeans(near_exit_positions, k=2)
sizes     = bincount(labels)
dominant  = argmax(sizes)          # cluster with the most particles
target    = centroids[dominant]
```

### Edge cases

| Situation | Behaviour |
|-----------|-----------|
| No particles near exits | Hold previous target |
| Fewer than k particles near exits | Fall back to plain mean |
| Cluster becomes empty mid-run | Centroid held at previous value |

### Robot motion (Phase 2)

```
target    = centroid of dominant k-means cluster among near-exit particles
           (fallback to mean / previous target — see edge cases above)

robot_pos = robot_pos + robot_alpha * (target - robot_pos)
```

The obstacle cost, gradient, and per-step order are **identical to Phase 1** —
only the targeting logic changes.

### Visualisation additions (Phase 2)

| Element | What it shows |
|---------|--------------|
| Faint magenta circles | Robot influence radius `R_robot` swept along trajectory |
| Small black dots | All k-means cluster centroids at sampled timesteps |

---

## Phase 3 — Prediction of Cluster Motion

### Concept

Phase 3 extends Phase 2 by estimating where the dominant cluster will be
**H steps in the future** and sending the robot there instead of the current
centroid.

**Why prediction helps:**

A purely reactive robot always targets where the cluster *was*. Because
particles move continuously toward exits, a reactive robot lags behind and
may arrive after the cluster has already passed through the exit. Predicting H
steps ahead gives the robot a **lead** — it intercepts the flow before it
reaches the exit rather than chasing it from behind.

**How `horizon` affects behaviour:**

| horizon | Effect |
|---------|--------|
| 0 | Equivalent to Phase 2 reactive targeting |
| 3–8 | Sweet spot — meaningful lead without overshooting |
| >10 | Predicted target may jump past the exit or leave the room |

### Prediction formula

```
v         = current_dominant_centroid - previous_dominant_centroid   (velocity estimate)

predicted = current_dominant_centroid + horizon * v
```

One-step finite difference gives a linear extrapolation. No smoothing or
filtering is applied — the motion of the dominant centroid is already smooth
because it is the mean of many particles.

### predict_cluster_target helper

```python
predicted = predict_cluster_target(current_centroid, prev_centroid, horizon)
```

Falls back to `current_centroid` when `prev_centroid is None` (first step) or
`horizon == 0`.

### update_robot_target (Phase 3)

Now returns 4 values:

```
target            — predicted dominant centroid (robot moves here)
centroids         — (k, 2) all k-means centroids, or None
dominant_centroid — (2,) current dominant centroid before prediction, or None
predicted_target  — (2,) same as target when clustering fires, else None
```

### Per-step order (Phase 3)

```
1. cluster near-exit particles  →  dominant centroid (current)
2. predict dominant centroid    →  predicted target  (current + H * v)
3. move robot toward predicted target (proportional)
4. snapshot particle positions
5. compute all gradients:
       soft-min goal + walls + inter-particle repulsion + robot repulsion
6. apply all particle updates simultaneously
7. save current dominant centroid as prev_dominant for next step
```

### Phase 3 parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `horizon` | 5 | Prediction look-ahead in steps (0 = reactive) |

All other robot parameters are unchanged from Phase 1/2.

### Visualisation additions (Phase 3)

| Element | What it shows |
|---------|--------------|
| Blue squares | Dominant cluster centroid at sampled timesteps |
| Orange diamonds | Predicted target at sampled timesteps |
| Faint orange arrows | Direction and magnitude of prediction offset |

## Cost Field Visualisation

```python
plot_cost_field_and_vectors(x0, g, walls, trajectory=None, grid_n=60)
```

| Layer | What it shows |
|-------|--------------|
| `contourf` (plasma) | Total cost — bright = high, dark = low (near goal). |
| `quiver` (white) | `−∇C(x)` — descent direction at each grid point. |
| Cyan line | Trajectory overlaid (when passed). |

## Project Structure

```
main.py
  goal_cost(x, g)                                        — single-goal cost
  grad_goal(x, g)                                        — single-goal gradient
  softmin_goal_cost(x, exits, beta)                      — soft-min over K exits
  grad_softmin_goal(x, exits, beta)                      — weighted-average gradient
  point_to_segment(x, a, b)                              — distance + closest point
  wall_cost(x, a, b, R, w)                               — wall penalty value
  grad_wall_penalty(x, a, b, R, w)                       — analytic wall gradient
  total_cost(x, g, walls)                                — single-goal + walls
  total_gradient(x, g, walls)                            — single-goal + wall gradients
  total_gradient_softmin(x, exits, walls, beta)          — soft-min + wall gradients
  particle_repulsion_cost(xi, xj, R_p, w_p)             — pairwise repulsion cost
  grad_particle_repulsion(xi, xj, R_p, w_p)             — repulsion gradient
  total_gradient_with_particles(i, positions, ...)       — single-goal + walls + repulsion
  total_gradient_with_particles_softmin(i, pos, ...)     — soft-min + walls + repulsion
  run_simulation(x0, g, walls, ...)                      — single-particle loop
  run_multi_particle_simulation(...)                     — N independent loops
  run_multi_particle_simulation_with_repulsion(...)      — N synchronous loops
  run_evacuation_simulation(starts, exits, walls, ...)        — soft-min evacuation loop
  robot_obstacle_cost(x, robot_pos, R_robot, w_robot)         — robot point obstacle cost
  grad_robot_obstacle(x, robot_pos, R_robot, w_robot)         — analytic robot gradient
  find_particles_near_exits(positions, active, exits, radius) — detect near-exit flow
  kmeans(points, k, n_iter)                                   — Lloyd's k-means clustering
  predict_cluster_target(current, prev, horizon)              — linear extrapolation of cluster motion
  update_robot_target(positions, active, exits, radius, prev) — clustering + prediction targeting
  run_evacuation_with_robot_phase1(...)                       — Phase 1/2/3 simulation loop
  plot_results(trajectory, ...)                               — single-particle plot
  plot_multi_particle_results(trajectories, ..., exits)       — all trajectories + exits
  plot_cost_field_and_vectors(x0, g, ...)                     — cost landscape + vectors
  plot_evacuation_with_robot(trajs, robot_traj, ...)          — Phase 3 visualisation
```

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`

```bash
pip install numpy matplotlib
```

## Usage

```bash
python main.py
```

Six windows open in sequence:

1. **Single-particle trajectory** — one particle from `(2,2)` routing around
   the n-shape to the single goal.
2. **Cost field** — plasma cost landscape with descent arrows and trajectory.
3. **Multi-particle independent** — 25 particles, no inter-particle forces,
   single goal inside the n-shape.
4. **Multi-particle with repulsion** — 25 particles with pairwise repulsion,
   synchronous updates, single goal.
5. **Evacuation (soft-min)** — 25 particles in an open room with two exits.
   Each particle smoothly steers toward the cheaper exit; repulsion spreads
   load across both exits without any explicit assignment logic.
6. **Phase 3 — robot with clustering + prediction** — same open room and
   exits. A robot point starts at `(5,5)`, clusters near-exit particles with
   k-means (k=2), predicts the dominant cluster's position 5 steps ahead, and
   moves to intercept that predicted position. Robot trajectory shown as dashed
   magenta; influence-radius circles as faint magenta rings; dominant centroid
   trail as blue squares; predicted target trail as orange diamonds with
   directional arrows.
