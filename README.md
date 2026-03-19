# CSE 5280 - Robot Arm: Gradient Descent Particle Simulation

## Overview

A 2D particle simulation where a particle navigates from a start position to a
goal while avoiding wall obstacles inside a bounded room. Motion emerges purely
from **cost minimisation** via gradient descent — there is no collision
detection, no path-finding (no A*), and no hard "if hit wall then stop" logic.

The environment consists of:
- An **outer square boundary** `[0,10] × [0,10]` made of 4 wall segments
- An **n-shaped interior enclosure** made of 3 wall segments (two vertical
  legs + one top bar), open at the bottom

The goal is placed **inside** the n-shaped enclosure. The start is placed
**outside and offset** so the direct path to the goal crosses the left leg.
The particle must therefore detour around the outside of the n-shape and enter
through the open bottom gap — behaviour that emerges entirely from the cost
landscape, with no explicit routing rules.

All gradients are derived analytically from first principles and implemented
directly in code.

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

```
x = x - alpha * grad C(x)
```

At each step the total analytic gradient is computed and the particle moves
one step downhill. Wall avoidance (including room boundary containment and
n-shape navigation) is a side-effect of cost minimisation.

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
  │   start(2,2)                        │
(0,0)──────────────────────────────(10,0)
```

The direct path from start `(2,2)` to goal `(5.5,7.0)` crosses the left leg.
The particle is repelled by that leg and must route around the outside of the
n-shape before aligning with the gap and entering from below.

## Wall Parameters

| Wall group      | `R`  | `w`   | Purpose |
|-----------------|------|-------|---------|
| Boundary walls  | 1.0  |  80.0 | Keep particle inside the room |
| n-shape walls   | 1.0  | 120.0 | Strong enough to force a full detour around the enclosure |

Higher `w` on the interior walls ensures the particle does not cut
unrealistically close to the legs; it arcs cleanly around the outside before
entering through the gap.

## Cost Field Visualisation

A second plot is produced for debugging the cost landscape:

```python
plot_cost_field_and_vectors(x0, g, walls, trajectory=None, grid_n=60)
```

| Layer | What it shows |
|-------|--------------|
| `contourf` (plasma) | Total cost at every grid point. Bright = high cost (near walls, far from goal). Dark = low cost (near goal). |
| `quiver` (white arrows) | Negative gradient `−∇C(x)` — the direction gradient descent moves the particle from each point. |
| Cyan line | Actual trajectory overlaid on the field (when `trajectory` is passed). |

**How to use it for debugging:**
- If arrows near the bottom gap point inward and upward → the landscape correctly guides the particle through.
- If arrows point sideways or outward near the gap → the wall parameters are too strong and the gap is not a viable corridor.
- Bright cost bands around the n-shape legs show the influence radius `R` visually.

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
  run_simulation(x0, g, walls, ...)       — gradient descent loop
  plot_results(trajectory, ...)           — trajectory visualisation
  plot_cost_field_and_vectors(x0, g, ...) — cost landscape + descent vector field
```

## Parameters

| Parameter    | Description |
|--------------|-------------|
| `alpha`      | Step size (learning rate). Too large → overshoot; too small → slow. |
| `n_steps`    | Number of gradient descent iterations. |
| `R`          | Wall influence radius. Repulsion activates when `d < R`. |
| `w`          | Wall penalty weight. Higher = harder wall. |
| `boundary_R` | Influence radius for the outer boundary walls. |
| `boundary_w` | Penalty weight for the outer boundary walls. |
| `interior_R` | Influence radius for the n-shape walls. |
| `interior_w` | Penalty weight for the n-shape walls. |

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

Two windows open:

1. **Trajectory plot** — particle path (blue) routing around the outside of the
   n-shaped enclosure (black), entering through the bottom gap, and reaching
   the goal (red) inside.

2. **Cost field plot** — plasma-coloured cost landscape with white descent
   arrows and the trajectory overlaid in cyan. Use this to debug why the
   particle routes the way it does.
