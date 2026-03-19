# CSE 5280 - Robot Arm: Gradient Descent Particle Simulation

## Overview

This project implements a basic **gradient descent** simulation that animates a particle moving toward a goal position in 2D space. It serves as a foundational demonstration of the iterative optimization technique used in robot arm control and motion planning.

## What It Does

- Initializes a particle at position `(2, 3)` and a goal at `(10, 8)`
- Applies gradient descent over 50 iterations to move the particle toward the goal
- Animates the motion in real time using matplotlib

## How It Works

At each step, the particle moves in the direction of the goal using:

```
x = x + alpha * (goal - x)
```

Where `alpha = 0.1` is the learning rate (step size). This is equivalent to a proportional controller or a single gradient descent step minimizing the squared distance to the goal.

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`

Install dependencies:

```bash
pip install numpy matplotlib
```

## Usage

```bash
python main.py
```

A window will open showing the particle (blue) converging toward the goal (red) over 50 animation frames.
