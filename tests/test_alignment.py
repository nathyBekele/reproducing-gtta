"""Checks for the Syntactic Alignment math (see notes/decisions.md). Uses the small dev model.

Run: pytest -q tests/test_alignment.py
"""
import os

import mlx.core as mx
import numpy as np
import pytest

from sa import alignment, data
from sa.backend import Backend

DEV_MODEL = os.environ.get("SA_DEV_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")


@pytest.fixture(scope="module")
def setup():
    be = Backend(DEV_MODEL)
    ex = data.load_split("val")[0]
    ids = be.encode(ex.prompt)[-700:]  # the tail keeps function docs, the user turn and the assistant header
    return be, ids, be.hidden(ids)


def test_user_turn_is_masked(setup):
    be, ids, _ = setup
    labels = list(ids[1:])
    mask = np.array(alignment.loss_mask(be, labels))
    start = be.encode("<|im_start|>user")
    i = next(k for k in range(len(labels)) if labels[k:k + len(start)] == start)
    assert mask[i] == 0 and mask[i - 1] == 1, "the user span should start being masked at <|im_start|>user"
    assert 0 < mask.sum() < len(mask)


def test_exact_gradient_matches_autograd(setup):
    be, ids, H = setup
    delta = mx.random.normal((be.d,), key=mx.random.key(0)) * 0.01
    g_manual, _ = alignment.gradient(be, H, ids, delta, mode="exact")
    g_auto = alignment.autograd_gradient(be, H, ids, delta)
    a, b = np.array(g_manual), np.array(g_auto)
    rel = np.linalg.norm(a - b) / np.linalg.norm(b)
    assert rel < 2e-2, f"relative error {rel:.4f}"


@pytest.mark.parametrize("mask_user", [True, False])
def test_exact_step_lowers_prompt_loss(setup, mask_user):
    """With the true gradient, one step must lower the loss. (Their "code" mode need not:
    that's a property of their implementation we measure, not a bug in ours.)"""
    be, ids, H = setup
    res = alignment.align(be, ids, H=H, mode="exact", mask_user=mask_user)
    assert res.loss_after < res.loss_before, (res.loss_before, res.loss_after)


@pytest.mark.parametrize("mask_user", [True, False])
def test_code_mode_matches_their_lines(setup, mask_user):
    """Line-by-line numpy port of model_runner.py 1938–1982 vs. our "code" mode gradient."""
    be, ids, H = setup
    ids = ids[-200:]
    Hs = H[-200:]
    labels = np.array(ids[1:])
    W = np.array(be.head(mx.eye(be.d)).T)  # [V, d] recovered through the head (bf16-rounded)
    h = np.array(Hs).astype(np.float64)
    temp_logits = h @ W.T
    shift_logits = temp_logits[:-1]
    z = shift_logits - shift_logits.max(-1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    if mask_user:
        valid = np.array(alignment.loss_mask(be, list(labels), True))
    else:
        valid = (labels >= 0).astype(np.float64)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = valid
    d_logits = (probs - one_hot) / (valid.sum() + 1e-10)
    d_hidden = d_logits @ W
    theirs = d_hidden.mean(0)

    ours, _ = alignment.gradient(be, Hs, ids, mx.zeros((be.d,)), mode="code", mask_user=mask_user)
    ours = np.array(ours)
    cos = ours @ theirs / (np.linalg.norm(ours) * np.linalg.norm(theirs))
    assert cos > 0.999, cos
    assert abs(np.linalg.norm(ours) / np.linalg.norm(theirs) - 1) < 0.02


def test_vocab_bias_is_context_independent(setup):
    """(h + δ)Wᵀ − hWᵀ == δWᵀ for any h: SA adds the same logit shift in every context."""
    be, ids, H = setup
    res = alignment.align(be, ids, H=H)
    b = np.array(alignment.vocab_bias(be, res.delta))
    for t in (10, 500):
        h = H[t]
        shift = np.array(be.head((h + res.delta)[None])[0] - be.head(h[None])[0])
        # bf16 head: compare with a tolerance relative to the size of the shift
        assert np.corrcoef(shift, b)[0, 1] > 0.99
        assert np.abs(shift - b).mean() < 0.1 * np.abs(b).mean() + 0.05


def test_first_adam_step_is_lr_times_sign(setup):
    """After one Adam step from zeros, δ ≈ −lr·sign(g) wherever |g| ≫ ε."""
    be, ids, H = setup
    res = alignment.align(be, ids, H=H, steps=1, mode="exact")
    g, d = np.array(res.grads[0]), np.array(res.delta)
    big = np.abs(g) > 100 * alignment.ADAM["eps"]
    assert big.mean() > 0.5
    np.testing.assert_allclose(d[big], -alignment.DEFAULT_LR * np.sign(g[big]), rtol=0.02, atol=1e-3)
