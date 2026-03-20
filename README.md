# CSE 5280 - Robot Arm: Gradient Descent Particle Simulation

## Overview

A 2D multi-particle simulation where particles navigate from start positions
toward goal(s) while avoiding wall obstacles inside a bounded room. Motion
emerges purely from **cost minimisation** via gradient descent — there is no
collision detection, no path-finding (no A*), and no hard routing logic.

Three simulation modes are supported:

| Mode | Goal term | Repulsion |
|------|-----------|-----------|
| Independent | Single goal | No |
| Pairwise repulsion | Single goal | Yes, synchronous |
| Evacuation (soft-min) | Two exits via soft-min | Yes, synchronous |

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
  run_evacuation_simulation(starts, exits, walls, ...)   — soft-min evacuation loop
  plot_results(trajectory, ...)                          — single-particle plot
  plot_multi_particle_results(trajectories, ..., exits)  — all trajectories + exits
  plot_cost_field_and_vectors(x0, g, ...)                — cost landscape + vectors
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

Five windows open in sequence:

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
