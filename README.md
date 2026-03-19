# CSE 5280 - Robot Arm: Gradient Descent Particle Simulation

## Overview

A 2D multi-particle simulation where particles navigate from start positions to
a shared goal while avoiding wall obstacles inside a bounded room. Motion
emerges purely from **cost minimisation** via gradient descent — there is no
collision detection, no path-finding (no A*), and no hard "if hit wall then
stop" logic.

The environment consists of:
- An **outer square boundary** `[0,10] × [0,10]` made of 4 wall segments
- An **n-shaped interior enclosure** made of 3 wall segments (two vertical
  legs + one top bar), open at the bottom

The goal is placed **inside** the n-shaped enclosure. Particles start from
various positions **outside** the enclosure. The direct path to the goal is
blocked by the enclosure walls, so each particle must detour around the outside
and enter through the open bottom gap — behaviour that emerges entirely from
the cost landscape.

Two simulation modes are supported:
- **Independent** — each particle only feels goal attraction and wall repulsion
- **With pairwise repulsion** — particles additionally repel each other using
  the same quadratic-band penalty framework, updated synchronously each step

All gradients are derived analytically from first principles.

## Cost Function

The total cost for particle `i` is:

```
C_i(x) = C_goal(x_i, g)
        + sum over walls:       C_wall(x_i, wall)
        + sum over j ≠ i:       C_repel(x_i, x_j)     ← only with repulsion enabled
```

### Goal term

```
C_goal(x) = 0.5 * ||x - g||²

grad C_goal = x - g
```

### Wall penalty term (per wall segment)

```
C_wall(x) = 0.5 * w * (R - d(x))²   if d(x) < R
           = 0                         otherwise

grad C_wall = -w * (R - d) * (x - p) / max(d, eps)   if d < R
            = 0                                         otherwise
```

where `p` is the closest point on the segment to `x`.

### Particle-particle repulsion term

Same quadratic-band form as the wall penalty, applied to pairwise distances.

```
C_repel(xi, xj) = 0.5 * w_p * (R_p - d_ij)²   if d_ij < R_p
                = 0                               otherwise

where d_ij = ||xi - xj||
```

**Analytic gradient w.r.t. xi (chain rule — identical derivation to wall gradient):**

```
dC/d(d_ij)  = -w_p * (R_p - d_ij)          (outer derivative)
d(d_ij)/dxi = (xi - xj) / d_ij             (gradient of Euclidean distance)

grad_xi C_repel = -w_p * (R_p - d_ij) * (xi - xj) / max(d_ij, eps)   if d_ij < R_p
                = 0                                                       otherwise
```

The direction pushes `xi` away from `xj`. By symmetry, the gradient w.r.t.
`xj` is the negative — equal and opposite repulsion.

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

Each particle runs its own loop with no awareness of others:

```
for each x0 in starts:
    for each step:
        if ||x - g|| < tol: break
        x = x - alpha * total_gradient(x, g, walls)
```

### Multiple particles — with pairwise repulsion (synchronous)

```
for each step:
    snapshot = positions.copy()                    # 1. freeze current state
    for each particle i:
        grads[i] = total_gradient_with_particles(  # 2. compute from snapshot
                       i, snapshot, g, walls, R_p, w_p)
    for each particle i:
        positions[i] -= alpha * grads[i]           # 3. apply all at once
```

**Why synchronous?** All gradients are computed from the same position snapshot
before any particle moves. If particle 1 moved first, particle 2 would react to
a position that didn't exist at the start of the step, breaking the physical
symmetry of the pairwise repulsion.

## Environment Layout

```
(0,10)──────────────────────────────(10,10)
  │                                     │
  │    (3.5,8.5)──────────(7.5,8.5)     │
  │        │   goal(5.5,7.0)  │         │
  │        │                  │         │
  │        │                  │         │
  │    (3.5,3.0)          (7.5,3.0)     │
  │             ↑ open gap              │
  │   starts scattered outside          │
(0,0)──────────────────────────────(10,0)
```

## Particle Start Positions

`N = 25` particles are sampled randomly across three regions outside the
n-shape enclosure (`np.random.seed(42)` for reproducibility):

| Region | x range | y range | Count |
|--------|---------|---------|-------|
| Left   | [1, 3]     | [2, 8]   | N // 3 |
| Right  | [7, 9]     | [2, 8]   | N // 3 |
| Bottom | [3.5, 7.5] | [1, 2.5] | remainder |

Left and right particles must detour around the corresponding n-shape leg.
Bottom particles are roughly aligned with the gap and enter more directly.

## Wall & Repulsion Parameters

| Parameter    | Value | Description |
|--------------|-------|-------------|
| `boundary_R` | 1.0   | Influence radius for outer boundary walls |
| `boundary_w` | 80.0  | Penalty weight for outer boundary walls |
| `interior_R` | 1.0   | Influence radius for n-shape walls |
| `interior_w` | 120.0 | Penalty weight for n-shape walls (stronger to force detour) |
| `R_p`        | 0.6   | Particle influence radius — smaller bubble lets 25 particles pass through the gap |
| `w_p`        | 30.0  | Particle repulsion weight — softer so particles can still converge near the goal |
| `alpha`      | 0.04  | Slightly reduced from 0.05 for stability with 25 interacting particles |
| `n_steps`    | 1000  | Increased to give all particles time to converge |

**Tuning `R_p` and `w_p`:** too large `R_p` spreads particles so far apart they
may not all fit through the gap; too large `w_p` prevents particles converging
near the goal. Reduce `w_p` if particles are pushed away before reaching `tol`.

## Cost Field Visualisation

```python
plot_cost_field_and_vectors(x0, g, walls, trajectory=None, grid_n=60)
```

| Layer | What it shows |
|-------|--------------|
| `contourf` (plasma) | Total cost — bright = high (near walls, far from goal), dark = low (near goal). |
| `quiver` (white arrows) | `−∇C(x)` — the direction gradient descent moves the particle. |
| Cyan line | Actual trajectory overlaid (when `trajectory` is passed). |

## Project Structure

```
main.py
  goal_cost(x, g)                                    — goal cost
  grad_goal(x, g)                                    — analytic gradient of goal cost
  point_to_segment(x, a, b)                          — distance + closest point on segment
  wall_cost(x, a, b, R, w)                           — wall penalty value
  grad_wall_penalty(x, a, b, R, w)                   — analytic gradient of wall penalty
  total_cost(x, g, walls)                            — goal + wall costs
  total_gradient(x, g, walls)                        — goal + wall gradients
  particle_repulsion_cost(xi, xj, R_p, w_p)          — pairwise repulsion cost
  grad_particle_repulsion(xi, xj, R_p, w_p)          — analytic gradient of repulsion
  total_gradient_with_particles(i, positions, ...)   — goal + wall + repulsion gradient
  run_simulation(x0, g, walls, ...)                  — single-particle loop
  run_multi_particle_simulation(...)                 — N independent loops
  run_multi_particle_simulation_with_repulsion(...)  — N synchronous loops with repulsion
  plot_results(trajectory, ...)                      — single-particle plot
  plot_multi_particle_results(trajectories, ...)     — all trajectories on one plot
  plot_cost_field_and_vectors(x0, g, ...)            — cost landscape + vector field
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

Four windows open in sequence:

1. **Single-particle trajectory** — one particle from `(2,2)` routing around
   the n-shape to the goal at `(5.5, 7.0)`.
2. **Cost field plot** — plasma cost landscape with descent arrows and the
   single-particle trajectory in cyan.
3. **Multi-particle (independent)** — all 5 particles, no inter-particle forces.
4. **Multi-particle (with repulsion)** — all 5 particles with pairwise
   quadratic-band repulsion, synchronous updates. Trajectories spread apart
   as particles push each other away while converging to the goal.
