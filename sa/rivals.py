"""Comparison conditions. Each is a fixed logit bias over the vocabulary (like SA's δWᵀ)."""
import mlx.core as mx

from .settings import ALPHA_GRID, RANDOM_SEEDS  # noqa: F401  (re-exported)


def copy_bias(be, prompt_ids, alpha):
    """Rival C: +α on every vocabulary token that appears anywhere in the prompt."""
    present = mx.zeros((be.vocab_size,), dtype=mx.float32)
    present[mx.array(sorted(set(prompt_ids)))] = 1.0
    return alpha * present


def random_delta(d, norm, seed, kind="gaussian"):
    """Rival D: a random δ with a given norm.

    "gaussian": random direction. "sign": random ±1 per coordinate, the shape of an
    Adam first step (δ ≈ −lr·sign(g)), so only the signs differ from SA's δ.
    """
    key = mx.random.key(seed)
    if kind == "gaussian":
        v = mx.random.normal((d,), key=key)
    elif kind == "sign":
        v = mx.where(mx.random.bernoulli(0.5, (d,), key=key), 1.0, -1.0)
    else:
        raise ValueError(kind)
    return v * (norm / (mx.linalg.norm(v) + 1e-12))
