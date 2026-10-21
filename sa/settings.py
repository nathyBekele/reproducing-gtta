"""Experiment settings shared by all engines (no MLX / torch imports here)."""

# SA hyperparameters from their model_runner.py (__init__) and the WebArena setting (lr 0.1, 1 step).
ADAM = dict(beta1=0.9, beta2=0.95, eps=1e-6, weight_decay=1e-5)
DEFAULT_LR = 0.1
DEFAULT_STEPS = 1

# Rivals.
ALPHA_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
RANDOM_SEEDS = [0, 1, 2]
