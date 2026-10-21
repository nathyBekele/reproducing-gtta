"""Syntactic Alignment (SA), replicated from r2llab/GTTA vllm/model_runner.py.

δ ∈ R^d (zeros at the start of a task) is added to the final hidden states H:
    logits = (H + δ) W^T
and updated with Adam on the next-token loss over the prompt, with user turns masked out.

Two gradient modes:
- "code":  exactly what their implementation computes (see notes/decisions.md):
           cot = (softmax − onehot·mask) / num_valid, then mean over ALL positions.
           Masked positions still contribute +softmax, and there's an extra 1/T factor.
- "exact": the true gradient of the masked mean cross-entropy:
           cot = (softmax − onehot) · mask / num_valid, summed over positions.
"""
from dataclasses import dataclass, field

import mlx.core as mx

from . import masking

from .settings import ADAM, DEFAULT_LR, DEFAULT_STEPS  # noqa: E402,F401  (re-exported)
CHUNK = 512  # positions per logits chunk (memory: CHUNK x V floats)


def loss_mask(be, labels, mask_user=True):
    """1 for label positions that count in the loss, 0 inside user turns (see sa/masking.py)."""
    return mx.array(masking.user_mask(be.encode, labels, mask_user))


def _head_vjp(be, x, cot):
    """cot @ W, i.e. the gradient flowing back through the LM head (works for 4-bit heads)."""
    _, (gx,) = mx.vjp(lambda v: be.head(v), [x], [cot])
    return gx


def gradient(be, H, ids, delta, mode="code", mask_user=True, chunk=CHUNK):
    """Gradient of the SA objective w.r.t. δ, plus the exact masked loss at δ.

    H: [T, d] final hidden states for `ids`; position t predicts ids[t + 1].
    """
    labels = list(ids[1:])
    Hs = H[:-1]
    mask = loss_mask(be, labels, mask_user)
    num_valid = mx.sum(mask) + 1e-10
    T = len(labels)
    y_all = mx.array(labels)
    g = mx.zeros((be.d,), dtype=mx.float32)
    loss = mx.array(0.0)
    for s in range(0, T, chunk):
        x = Hs[s:s + chunk] + delta
        y, m = y_all[s:s + chunk], mask[s:s + chunk]
        logits = be.head(x)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        p = mx.exp(logp)
        onehot = (mx.arange(logits.shape[-1])[None, :] == y[:, None]).astype(mx.float32)
        if mode == "code":
            cot = (p - onehot * m[:, None]) / num_valid
        elif mode == "exact":
            cot = (p - onehot) * m[:, None] / num_valid
        else:
            raise ValueError(mode)
        gx = _head_vjp(be, x, cot)
        g = g + mx.sum(gx, axis=0)
        loss = loss - mx.sum(mx.take_along_axis(logp, y[:, None], axis=-1)[:, 0] * m)
        mx.eval(g, loss)
    if mode == "code":
        g = g / T  # their d_hidden.mean(dim=0)
    return g, float(loss / num_valid)


def autograd_gradient(be, H, ids, delta, mask_user=True, chunk=CHUNK):
    """Reference for tests: autograd of the exact masked mean cross-entropy."""
    labels = list(ids[1:])
    Hs = H[:-1]
    mask = loss_mask(be, labels, mask_user)
    num_valid = mx.sum(mask)
    y_all = mx.array(labels)
    g = mx.zeros((be.d,), dtype=mx.float32)
    for s in range(0, len(labels), chunk):
        x, y, m = Hs[s:s + chunk], y_all[s:s + chunk], mask[s:s + chunk]

        def chunk_loss(dl):
            logits = be.head(x + dl)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return -mx.sum(mx.take_along_axis(logp, y[:, None], axis=-1)[:, 0] * m) / num_valid

        g = g + mx.grad(chunk_loss)(delta)
        mx.eval(g)
    return g


@dataclass
class AlignResult:
    delta: mx.array
    loss_before: float              # exact masked loss at δ = 0
    loss_after: float               # exact masked loss at the final δ
    grads: list = field(default_factory=list)  # gradient used at each step


def align(be, ids, H=None, steps=DEFAULT_STEPS, lr=DEFAULT_LR, mode="code", mask_user=True):
    """Run SA on one prompt (their Adam, zeros init). Returns δ and before/after losses."""
    if H is None:
        H = be.hidden(ids)
    delta = mx.zeros((be.d,), dtype=mx.float32)
    m = mx.zeros_like(delta)
    v = mx.zeros_like(delta)
    b1, b2, eps, wd = ADAM["beta1"], ADAM["beta2"], ADAM["eps"], ADAM["weight_decay"]
    grads, loss_before = [], None
    for step in range(1, steps + 1):
        g, loss = gradient(be, H, ids, delta, mode, mask_user)
        if loss_before is None:
            loss_before = loss
        grads.append(g)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** step)
        v_hat = v / (1 - b2 ** step)
        delta = delta - lr * m_hat / (mx.sqrt(v_hat) + eps) - lr * wd * delta
        mx.eval(delta)
    _, loss_after = gradient(be, H, ids, delta, "exact", mask_user)
    return AlignResult(delta=delta, loss_before=loss_before, loss_after=loss_after, grads=grads)


def vocab_bias(be, delta):
    """b = δ W^T: the context-independent logit shift SA applies to every token."""
    return be.head(delta[None])[0]
