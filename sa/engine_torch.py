"""PyTorch engine (NVIDIA GPU, or CPU for checks): same records as the MLX engine, faster.

Speed-up ("lever 1"): every SA variant takes its first Adam step from δ = 0, so they all
share the same probabilities p = softmax(H·Wᵀ). One pass over the vocabulary gives:
  exact gradient          g_exact = Σ_t (p_t − e_y) W / T            (no mask, mean CE)
  paper (code, no mask)   g_paper = g_exact / T                       (their extra .mean(0))
  code + mask             g_mask  = Σ_t (p_t − m_t·e_y) W / N / T
and the first Adam step from zeros is δ = −lr·g / (|g| + ε) (m̂ = g, v̂ = g²; weight decay × 0).
For steps > 1, later steps recompute each variant's gradient at its own current δ (their Adam, as in
alignment.py). lr and steps are constructor arguments (defaults: the paper's WebArena setting).
"""
import time

import numpy as np
import torch

from . import settings
from .masking import user_mask
from .records import VARIANTS, new_record, summarize_logprobs, token_sets, top_tokens

EPS = settings.ADAM["eps"]
B1, B2, WD = settings.ADAM["beta1"], settings.ADAM["beta2"], settings.ADAM["weight_decay"]
CHUNK = 2048


class TorchEngine:
    name = "torch"

    def __init__(self, model_path, device=None, dtype=torch.bfloat16, lr=None, steps=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.lr = settings.DEFAULT_LR if lr is None else float(lr)
        self.steps = settings.DEFAULT_STEPS if steps is None else int(steps)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(self.device).eval()
        cfg = self.model.config
        self.tied = bool(getattr(cfg, "tie_word_embeddings", False))
        self.d = cfg.hidden_size
        # Full-precision copy of the LM head for all logit math: [V, d].
        self.W = self.model.get_output_embeddings().weight.detach().float()
        self.vocab_size = self.W.shape[0]
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

    def info(self):
        return {"engine": self.name, "model": self.model_path, "tied": self.tied, "d": self.d,
                "vocab_size": self.vocab_size, "device": self.device, "lr": self.lr, "steps": self.steps,
                "dtype": str(next(self.model.parameters()).dtype)}

    # --- basic pieces -----------------------------------------------------------------
    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @torch.no_grad()
    def hidden(self, ids):
        """H [T, d] float32: final hidden states after the last norm."""
        x = torch.tensor([ids], device=self.device)
        return self.model.model(input_ids=x).last_hidden_state[0].float()

    def head(self, x):
        return x @ self.W.T

    def vocab_bias(self, delta):
        return self.W @ delta

    # --- SA -----------------------------------------------------------------------------------
    @torch.no_grad()
    def _sums(self, Hs, labels, mask_m, delta, need=("full", "mask")):
        """At a given δ: S_full = Σ_t (p_t − e_y)·W, S_mask = Σ_t (p_t − m_t·e_y)·W, plus NLL sums."""
        T = len(labels)
        y = torch.tensor(labels, device=self.device)
        m = torch.tensor(mask_m, device=self.device)
        S = {k: torch.zeros(self.d, device=self.device) for k in need}
        nll_all = torch.zeros((), device=self.device)
        nll_masked = torch.zeros((), device=self.device)
        for s in range(0, T, CHUNK):
            logp = torch.log_softmax(self.head(Hs[s:s + CHUNK] + delta), dim=-1)
            yc, mc = y[s:s + CHUNK], m[s:s + CHUNK]
            nll = -logp.gather(1, yc[:, None])[:, 0]
            nll_all += nll.sum()
            nll_masked += (nll * mc).sum()
            p = logp.exp()
            rows = torch.arange(len(yc), device=self.device)
            if "full" in need:
                cot = p.clone() if "mask" in need else p
                cot[rows, yc] -= 1.0
                S["full"] += (cot @ self.W).sum(0)
            if "mask" in need:
                p[rows, yc] -= mc
                S["mask"] += (p @ self.W).sum(0)
        n_mask = float(m.sum()) + 1e-10
        return S, {"no_mask": float(nll_all / T), "mask": float(nll_masked / n_mask)}, n_mask

    @staticmethod
    def _variant_grad(name, S, T, n_mask):
        if name == "sa_exact":
            return S["full"] / T
        if name == "sa_paper":
            return S["full"] / T / T          # their code with mask off: extra .mean(0)
        if name == "sa_code_mask":
            return S["mask"] / n_mask / T     # their code with mask on
        raise ValueError(name)

    @torch.no_grad()
    def _loss(self, Hs, labels, mask_m, delta):
        y = torch.tensor(labels, device=self.device)
        m = torch.tensor(mask_m, device=self.device)
        total = torch.zeros((), device=self.device)
        for s in range(0, len(labels), CHUNK):
            logp = torch.log_softmax(self.head(Hs[s:s + CHUNK] + delta), dim=-1)
            total += (-logp.gather(1, y[s:s + CHUNK, None])[:, 0] * m[s:s + CHUNK]).sum()
        return float(total / (m.sum() + 1e-10))

    def _adam(self, state, g, step):
        state["m"] = B1 * state["m"] + (1 - B1) * g
        state["v"] = B2 * state["v"] + (1 - B2) * g * g
        m_hat = state["m"] / (1 - B1 ** step)
        v_hat = state["v"] / (1 - B2 ** step)
        state["delta"] = state["delta"] - self.lr * m_hat / (v_hat.sqrt() + EPS) - self.lr * WD * state["delta"]

    def align_all(self, H_prompt, prompt_ids):
        labels = list(prompt_ids[1:])
        T = len(labels)
        Hs = H_prompt[:-1]
        masks = {False: [1.0] * T, True: user_mask(self.encode, labels, True)}
        zero = torch.zeros(self.d, device=self.device)
        S0, losses, n_mask = self._sums(Hs, labels, masks[True], zero)   # shared pass at δ = 0
        out = {}
        for name, kw in VARIANTS.items():
            g1 = self._variant_grad(name, S0, T, n_mask)
            st = {"delta": zero.clone(), "m": zero.clone(), "v": zero.clone()}
            self._adam(st, g1, 1)
            need = ("mask",) if kw["mask_user"] else ("full",)
            for step in range(2, self.steps + 1):
                S, _, _ = self._sums(Hs, labels, masks[True], st["delta"], need)
                self._adam(st, self._variant_grad(name, S, T, n_mask), step)
            mk = masks[kw["mask_user"]]
            out[name] = {
                "delta": st["delta"], "grad": g1,
                "loss_before": losses["mask" if kw["mask_user"] else "no_mask"],
                "loss_after": self._loss(Hs, labels, mk, st["delta"]),
            }
        return out

    # --- scoring --------------------------------------------------------------------------
    @staticmethod
    def answer_logprobs(L, answer_ids, bias=None):
        logits = L if bias is None else L + bias[None, :]
        logp = torch.log_softmax(logits, dim=-1)
        idx = torch.tensor(answer_ids, device=L.device)
        return logp.gather(1, idx[:, None])[:, 0].tolist()

    def random_delta(self, norm, seed, kind):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        if kind == "gaussian":
            v = torch.randn(self.d, generator=gen)
        else:
            v = torch.where(torch.rand(self.d, generator=gen) < 0.5, 1.0, -1.0)
        return (v * (norm / (v.norm() + 1e-12))).to(self.device)

    def copy_bias(self, prompt_ids, alpha):
        b = torch.zeros(self.vocab_size, device=self.device)
        b[torch.tensor(sorted(set(prompt_ids)), device=self.device)] = alpha
        return b

    # --- one example ------------------------------------------------------------------------
    @torch.no_grad()
    def run_example(self, ex, k):
        t0 = time.time()
        p_ids, a_ids = self.encode(ex.prompt), self.encode(ex.answer)
        if self.encode(ex.prompt + ex.answer) != p_ids + a_ids:
            raise ValueError("prompt/answer boundary tokenizes differently when joined")
        H = self.hidden(p_ids + a_ids)               # the single forward pass
        start = len(p_ids) - 1
        L = self.head(H[start:start + len(a_ids)])
        sets = token_sets(self.encode, ex, p_ids, a_ids)
        out = new_record(ex, self.decode, p_ids, a_ids)

        def score(name, bias=None, **extra):
            lps = self.answer_logprobs(L, a_ids, bias)
            out["conditions"][name] = {"logprobs": lps, **summarize_logprobs(lps), **extra}

        score("none")
        for name, r in self.align_all(H[:len(p_ids)], p_ids).items():
            kw = VARIANTS[name]
            bias = self.vocab_bias(r["delta"])
            g = r["grad"].abs().cpu().numpy()
            delta_norm = float(r["delta"].norm())
            out["sa"][name] = {
                **kw,
                "loss_before": r["loss_before"], "loss_after": r["loss_after"],
                "delta_norm": delta_norm,
                "grad_median_abs": float(np.median(g)),
                "frac_grad_below_eps": float((g < EPS).mean()),
                "bias_std": float(bias.std()),
                "top_tokens": top_tokens(self.decode, bias.cpu().numpy(), sets, k),
            }
            score(name, bias)
            for kind in ("gaussian", "sign"):
                vals = [self.answer_logprobs(L, a_ids, self.vocab_bias(self.random_delta(delta_norm, s, kind)))
                        for s in settings.RANDOM_SEEDS]
                mean_lps = np.mean(np.array(vals), axis=0).tolist()
                out["conditions"][f"random_{kind}_like_{name}"] = {
                    "logprobs": mean_lps, **summarize_logprobs(mean_lps), "seeds": settings.RANDOM_SEEDS}
        for alpha in settings.ALPHA_GRID:
            score(f"copy_alpha_{alpha}", self.copy_bias(p_ids, alpha), alpha=alpha)
        out["seconds"] = round(time.time() - t0, 2)
        return out
