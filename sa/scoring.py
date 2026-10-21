"""Teacher-forced scoring of the gold answer under a logit bias.

With teacher forcing the tokens are fixed, so the hidden states don't depend on δ:
one forward pass over prompt + answer gives every H, and each condition is just
log_softmax(H·Wᵀ + bias) at the answer positions.
"""
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class Encoded:
    prompt_ids: list
    answer_ids: list

    @property
    def ids(self):
        return self.prompt_ids + self.answer_ids

    @property
    def answer_rows(self):
        """Rows of H that predict the answer tokens (position t predicts token t + 1)."""
        start = len(self.prompt_ids) - 1
        return slice(start, start + len(self.answer_ids))


def encode_example(be, prompt, answer):
    """Tokenize prompt and answer; check that splitting doesn't change the tokens at the boundary."""
    p, a = be.encode(prompt), be.encode(answer)
    joint = be.encode(prompt + answer)
    if joint != p + a:
        raise ValueError("prompt/answer boundary tokenizes differently when joined")
    return Encoded(p, a)


def base_logits(be, H, enc):
    """Unbiased logits [n_answer, V] at the answer positions."""
    return be.head(H[enc.answer_rows])


def answer_logprobs(logits, answer_ids, bias=None):
    """Per-token log-probabilities of the gold answer under (logits + bias)."""
    if bias is not None:
        logits = logits + bias[None, :]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    lp = mx.take_along_axis(logp, mx.array(answer_ids)[:, None], axis=-1)[:, 0]
    mx.eval(lp)
    return lp.tolist()


def summarize(lps):
    return {"sum": float(sum(lps)), "mean": float(sum(lps) / len(lps))}
