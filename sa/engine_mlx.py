"""MLX engine (Apple Silicon): one example → one result record. Reference implementation."""
import time

import mlx.core as mx
import numpy as np

from . import alignment, rivals, scoring
from .backend import Backend
from .records import VARIANTS, new_record, summarize_logprobs, token_sets, top_tokens


class MLXEngine:
    name = "mlx"

    def __init__(self, model_path):
        self.be = Backend(model_path)
        self.model_path = model_path

    def info(self):
        be = self.be
        return {"engine": self.name, "model": self.model_path, "tied": be.tied, "d": be.d, "vocab_size": be.vocab_size}

    def run_example(self, ex, k):
        be = self.be
        t0 = time.time()
        enc = scoring.encode_example(be, ex.prompt, ex.answer)
        H = be.hidden(enc.ids)                       # the single forward pass
        Hp = H[: len(enc.prompt_ids)]
        L = scoring.base_logits(be, H, enc)
        sets = token_sets(be.encode, ex, enc.prompt_ids, enc.answer_ids)
        out = new_record(ex, be.decode, enc.prompt_ids, enc.answer_ids)

        def score(name, bias=None, **extra):
            lps = scoring.answer_logprobs(L, enc.answer_ids, bias)
            out["conditions"][name] = {"logprobs": lps, **summarize_logprobs(lps), **extra}

        score("none")
        for name, kw in VARIANTS.items():
            res = alignment.align(be, enc.prompt_ids, H=Hp, **kw)
            bias = alignment.vocab_bias(be, res.delta)
            g = np.abs(np.array(res.grads[0]))
            delta_norm = float(mx.linalg.norm(res.delta))
            out["sa"][name] = {
                **kw,
                "loss_before": res.loss_before, "loss_after": res.loss_after,
                "delta_norm": delta_norm,
                "grad_median_abs": float(np.median(g)),
                "frac_grad_below_eps": float((g < alignment.ADAM["eps"]).mean()),
                "bias_std": float(np.array(bias).std()),
                "top_tokens": top_tokens(be.decode, np.array(bias), sets, k),
            }
            score(name, bias)
            for kind in ("gaussian", "sign"):
                vals = []
                for seed in rivals.RANDOM_SEEDS:
                    rb = alignment.vocab_bias(be, rivals.random_delta(be.d, delta_norm, seed, kind))
                    vals.append(scoring.answer_logprobs(L, enc.answer_ids, rb))
                mean_lps = np.mean(np.array(vals), axis=0).tolist()
                out["conditions"][f"random_{kind}_like_{name}"] = {
                    "logprobs": mean_lps, **summarize_logprobs(mean_lps), "seeds": rivals.RANDOM_SEEDS}
        for alpha in rivals.ALPHA_GRID:
            score(f"copy_alpha_{alpha}", rivals.copy_bias(be, enc.prompt_ids, alpha), alpha=alpha)
        out["seconds"] = round(time.time() - t0, 1)
        mx.clear_cache()
        return out
