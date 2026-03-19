# CSE 5280 - Robot Arm: Gradient Descent Particle Simulation

## Overview

A 2D particle simulation where a particle navigates from a start position to a
goal while avoiding wall obstacles. Motion emerges purely from **cost
minimisation** via gradient descent — there is no collision detection, no
path-finding (no A*), and no hard "if hit wall then stop" logic.

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
one step downhill. Wall avoidance is a side-effect of cost minimisation — the
particle steers away from walls because getting closer raises the cost.

## Project Structure

```
main.py
  goal_cost(x, g)                  — goal cost value
  grad_goal(x, g)                  — analytic gradient of goal cost
  point_to_segment(x, a, b)        — returns (d, p): distance + closest point
  wall_cost(x, a, b, R, w)         — wall penalty value
  grad_wall_penalty(x, a, b, R, w) — analytic gradient of wall penalty
  total_cost(x, g, walls)          — combined cost
  total_gradient(x, g, walls)      — combined analytic gradient
  run_simulation(x0, g, walls, ...) — gradient descent loop
  plot_results(trajectory, ...)    — matplotlib visualisation
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `alpha`   | Step size (learning rate). Too large → overshoot; too small → slow. |
| `n_steps` | Number of gradient descent iterations. |
| `R`       | Wall influence radius. Particle starts feeling repulsion when `d < R`. |
| `w`       | Wall penalty weight. Higher = harder wall. |

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

A window opens showing the particle trajectory (blue) bending around the wall
(black) from start (green) to goal (red).
