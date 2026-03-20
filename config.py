# config.py
# Shared numerical constants used across the simulation modules.
# Only true constants belong here — default parameter values stay in each
# function signature so callers can override them without touching this file.

EPS = 1e-8   # safeguard against division by zero (distances, norms)
