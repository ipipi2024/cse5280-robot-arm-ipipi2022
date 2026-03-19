# CSE 5280 - Robot Arm: Gradient Descent Particle Simulation

## Overview

A 2D particle simulation where particles navigate from start positions to a
shared goal while avoiding wall obstacles inside a bounded room. Motion emerges
purely from **cost minimisation** via gradient descent — there is no collision
detection, no path-finding (no A*), and no hard "if hit wall then stop" logic.

The environment consists of:
- An **outer square boundary** `[0,10] × [0,10]` made of 4 wall segments
- An **n-shaped interior enclosure** made of 3 wall segments (two vertical
  legs + one top bar), open at the bottom

The goal is placed **inside** the n-shaped enclosure. Particles start from
various positions **outside** the enclosure. The direct path to the goal is
blocked by the enclosure walls, so each particle must detour around the outside
and enter through the open bottom gap — behaviour that emerges entirely from
the cost landscape, with no explicit routing rules.

Both single-particle and multi-particle modes are supported. In the
multi-particle case each particle is simulated independently using the same
cost function — there is no inter-particle repulsion.

All gradients are derived analytically from first principles.

## Cost Function

The total cost the particle minimises at each step is:

```
C(x) = C_goal(x) + sum_i C_wall_i(x)
```

### Goal term

```
C_goal(x) = 0.5 * ||x - g||²

grad C_goal = x - g
```

Descending this gradient pulls the particle toward goal `g`.

### Wall penalty term (per wall segment)

Applies to every wall — boundary or interior — identically.

```
C_wall(x) = 0.5 * w * (R - d(x))²   if d(x) < R
           = 0                         otherwise
```

where `d(x)` is the shortest distance from `x` to the wall segment `[a, b]`,
`R` is the influence radius, and `w` is the penalty weight.

**Analytic gradient via chain rule:**

```
dC_wall/dd  = -w * (R - d)           (outer derivative)
dd/dx       = (x - p) / d            (gradient of Euclidean distance,
                                       p = closest point on segment)

grad C_wall = -w * (R - d) * (x - p) / max(d, eps)   if d < R
            = 0                                         otherwise
```

The `max(d, eps)` guard prevents division by zero when the particle sits
exactly on the wall.

## Distance: Point to Segment

The closest point `p` on segment `[a, b]` to point `x` is found by:

```
t = dot(x - a, b - a) / dot(b - a, b - a)   # project onto line
t = clamp(t, 0, 1)                            # constrain to segment
p = a + t * (b - a)
d = ||x - p||
```

Clamping `t` handles the endpoint cases (before `a` or past `b`) automatically.
The gradient formula `dd/dx = (x - p) / d` is valid in all cases.

## Simulation Loop

### Single particle

```
x = x - alpha * grad C(x)
```

### Multiple particles

```python
for each start position x0:
    for each step:
        if ||x - g|| < tol: break   # early stop on convergence
        x = x - alpha * total_gradient(x, g, walls)
```

Each particle runs the same loop independently. No particle sees any other.
The result is a list of trajectory arrays, one per particle.

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

The direct path from any start outside the left leg to the goal crosses that
leg. Each particle is repelled and must route around the outside before
aligning with the gap and entering from below.

## Multi-particle Start Positions

```python
starts = [
    [2.0, 2.0],   # lower-left  — must go around the left leg
    [8.0, 2.0],   # lower-right — must go around the right leg
    [5.5, 1.5],   # bottom-centre — aligned with the gap
    [1.5, 5.0],   # left side
    [8.5, 5.0],   # right side
]
```

## Wall Parameters

| Wall group     | `R`  | `w`   | Purpose |
|----------------|------|-------|---------|
| Boundary walls | 1.0  |  80.0 | Keep particles inside the room |
| n-shape walls  | 1.0  | 120.0 | Force a full detour; prevent cutting through |

## Cost Field Visualisation

A debug plot is produced alongside the trajectory plot:

```python
plot_cost_field_and_vectors(x0, g, walls, trajectory=None, grid_n=60)
```

| Layer | What it shows |
|-------|--------------|
| `contourf` (plasma) | Total cost — bright = high (near walls, far from goal), dark = low (near goal). |
| `quiver` (white arrows) | Negative gradient `−∇C(x)` — the direction gradient descent moves the particle. |
| Cyan line | Actual trajectory overlaid on the field (when `trajectory` is passed). |

**How to use it for debugging:**
- Arrows near the bottom gap pointing inward and upward → cost corridor is viable.
- Arrows pointing sideways or outward near the gap → wall `R` or `w` is too high; the gap is blocked.

`grid_n` controls resolution: `40` is fast, `60` is the default, `80+` gives finer contours.

## Project Structure

```
main.py
  goal_cost(x, g)                          — goal cost value
  grad_goal(x, g)                          — analytic gradient of goal cost
  point_to_segment(x, a, b)               — returns (d, p): distance + closest point
  wall_cost(x, a, b, R, w)                — wall penalty value
  grad_wall_penalty(x, a, b, R, w)        — analytic gradient of wall penalty
  total_cost(x, g, walls)                 — combined cost
  total_gradient(x, g, walls)             — combined analytic gradient
  run_simulation(x0, g, walls, ...)       — single-particle gradient descent loop
  run_multi_particle_simulation(...)      — N independent gradient descent loops
  plot_results(trajectory, ...)           — single-particle trajectory plot
  plot_multi_particle_results(...)        — all trajectories on one plot
  plot_cost_field_and_vectors(x0, g, ...) — cost landscape + descent vector field
```

## Parameters

| Parameter    | Description |
|--------------|-------------|
| `alpha`      | Step size (learning rate). Too large → overshoot; too small → slow. |
| `n_steps`    | Maximum number of gradient descent iterations per particle. |
| `tol`        | Convergence threshold — particle stops when `‖x − g‖ < tol`. |
| `R`          | Wall influence radius. Repulsion activates when `d < R`. |
| `w`          | Wall penalty weight. Higher = harder wall. |
| `boundary_R` | Influence radius for the outer boundary walls. |
| `boundary_w` | Penalty weight for the outer boundary walls. |
| `interior_R` | Influence radius for the n-shape walls. |
| `interior_w` | Penalty weight for the n-shape walls. |
| `grid_n`     | Grid resolution for cost field visualisation. |

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

Three windows open in sequence:

1. **Single-particle trajectory** — one particle from `(2,2)` routing around
   the n-shape to reach the goal at `(5.5, 7.0)`.

2. **Cost field plot** — plasma cost landscape with white descent arrows and
   the single-particle trajectory in cyan. Use this to debug the cost corridor.

3. **Multi-particle trajectories** — all 5 particles shown together in
   distinct colours, each routed independently through the same environment.
