"""Live SA demo: one request → what SA's δ looks like on a prompt, with their code vs. the true gradient.

Self-contained (torch + transformers only) so the same file runs on a Modal GPU and on a CPU Space.

What one run returns:
- curve: size of δ after SA's first Adam step, if SA saw only the first L tokens of the prompt, for a grid
  of L, under each gradient variant. Computed in ONE pass: the model reads left to right, so the hidden
  states of a prefix don't depend on what follows, and the gradient sums are prefix sums.
- the full-length run of the chosen variant / lr / steps: ‖δ‖, share of gradient coordinates below
  Adam's ε, the tokens δ·Wᵀ pushes up most, and (if an expected answer is given) how each answer
  token's log-probability changes.
- optionally: a greedy reply with and without SA.

SA as in GTTA's vllm/model_runner.py: δ added to the final hidden states (after the last norm),
Adam (β 0.9/0.95, ε 1e-6, wd 1e-5), loss = next-token cross-entropy over the prompt.
Variants: "paper" = their gradient with masking off (sum/T/T, the extra .mean(0)); "mask" = their gradient
with user turns masked; "exact" = the true gradient of the mean loss.
"""
import math
import time

import torch

B1, B2, EPS, WD = 0.9, 0.95, 1e-6, 1e-5
USER_START, TURN_END = "<|im_start|>user", "<|im_end|>"
VARIANTS = ("paper", "mask", "exact")
CHUNK = 1024


def _find(seq, pat, start=0):
    n = len(pat)
    for i in range(start, len(seq) - n + 1):
        if seq[i:i + n] == pat:
            return i
    return -1


def user_mask(encode, labels):
    """1.0 = counts in the loss, 0.0 = inside a user turn (their mask_tokens == "true" behaviour)."""
    labels = list(labels)
    mask = [1.0] * len(labels)
    s, e = encode(USER_START), encode(TURN_END)
    i = _find(labels, s)
    while i != -1:
        j = _find(labels, e, i + len(s))
        stop = len(labels) if j == -1 else j + len(e)
        for k in range(i, stop):
            mask[k] = 0.0
        i = _find(labels, s, stop)
    return mask


class LiveSA:
    def __init__(self, model_id, device=None, dtype=None, max_tokens=12000):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.W = self.model.get_output_embeddings().weight.detach().float()     # [V, d]
        self.d = self.W.shape[1]
        self.max_tokens = max_tokens
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        self.stop_ids = [i for i in (self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id) if i is not None]

    def to(self, device):
        """Move to another device (Modal snapshots: load on CPU, snapshot, then move to the GPU)."""
        self.device = device
        self.model.to(device)
        self.W = self.W.to(device)
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        return self

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    # ---------- core math ----------
    @torch.no_grad()
    def _sums(self, Hs, labels, mask, delta, record_at=None):
        """S_full = Σ (p − e_y)W and S_mask = Σ (p − m·e_y)W over positions; optionally prefix sums at record_at."""
        y = torch.tensor(labels, device=self.device)
        m = torch.tensor(mask, device=self.device)
        S_full = torch.zeros(self.d, device=self.device)
        S_mask = torch.zeros(self.d, device=self.device)
        nll = torch.zeros((), device=self.device)
        rec, marks = {}, sorted(record_at or [])
        T = len(labels)
        for s in range(0, T, CHUNK):
            e = min(T, s + CHUNK)
            logp = torch.log_softmax(Hs[s:e] @ self.W.T + (Hs.new_zeros(1) if delta is None else delta @ self.W.T), dim=-1)
            yc, mc = y[s:e], m[s:e]
            p = logp.exp()
            rows = torch.arange(e - s, device=self.device)
            nll_c = -logp[rows, yc]
            full = p.clone()
            full[rows, yc] -= 1.0
            p[rows, yc] -= mc
            G_full, G_mask = full @ self.W, p @ self.W           # [chunk, d]
            if marks:
                cf, cm = torch.cumsum(G_full, 0), torch.cumsum(G_mask, 0)
                cn = torch.cumsum(nll_c, 0)
                cmask = torch.cumsum(mc, 0)
                base_mask = float(m[:s].sum())
                for L in marks:
                    if s < L <= e:
                        rec[L] = (S_full + cf[L - s - 1], S_mask + cm[L - s - 1], base_mask + float(cmask[L - s - 1]),
                                  float(nll + cn[L - s - 1]))
            S_full += G_full.sum(0)
            S_mask += G_mask.sum(0)
            nll += nll_c.sum()
        return S_full, S_mask, float(m.sum()), float(nll), rec

    @staticmethod
    def _grad(variant, S_full, S_mask, n_mask, T):
        if variant == "exact":
            return S_full / T
        if variant == "paper":
            return S_full / T / T
        return S_mask / max(n_mask, 1e-10) / T

    @staticmethod
    def _adam(st, g, lr):
        st["step"] += 1
        st["m"] = B1 * st["m"] + (1 - B1) * g
        st["v"] = B2 * st["v"] + (1 - B2) * g * g
        mh = st["m"] / (1 - B1 ** st["step"])
        vh = st["v"] / (1 - B2 ** st["step"])
        st["delta"] = st["delta"] - lr * mh / (vh.sqrt() + EPS) - lr * WD * st["delta"]

    def _top(self, b, prompt_ids, answer_ids, k):
        in_p, in_a = set(prompt_ids), set(answer_ids)
        out = []
        for rank, idx in (("top", torch.topk(b, k).indices), ("bottom", torch.topk(-b, 5).indices)):
            for i in idx.tolist():
                out.append({"token": self.tok.decode([i]), "bias": round(float(b[i]), 6), "rank": rank,
                            "in_prompt": i in in_p, "in_answer": i in in_a})
        return out

    @torch.no_grad()
    def _generate(self, prompt_ids, bias, max_new):
        x = torch.tensor([prompt_ids], device=self.device)
        out = self.model.model(input_ids=x, use_cache=True)
        cache, h = out.past_key_values, out.last_hidden_state[0, -1].float()
        toks = []
        for _ in range(max_new):
            logits = h @ self.W.T if bias is None else h @ self.W.T + bias
            t = int(torch.argmax(logits))
            if t in self.stop_ids:
                break
            toks.append(t)
            out = self.model.model(input_ids=torch.tensor([[t]], device=self.device), past_key_values=cache, use_cache=True)
            cache, h = out.past_key_values, out.last_hidden_state[0, -1].float()
        return self.tok.decode(toks, skip_special_tokens=True)

    # ---------- one request ----------
    @torch.no_grad()
    def run(self, prompt, answer="", variant="paper", lr=0.1, steps=1, top_k=20, curve_points=12,
            generate=False, max_new=96):
        t0 = time.time()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        lr, steps = float(lr), int(steps)
        p_ids, a_ids = self.encode(prompt), self.encode(answer) if answer else []
        if len(p_ids) < 8:
            raise ValueError("prompt is too short (need at least 8 tokens)")
        if len(p_ids) + len(a_ids) > self.max_tokens:
            raise ValueError(f"prompt + answer is {len(p_ids) + len(a_ids)} tokens; the limit here is {self.max_tokens}")
        x = torch.tensor([p_ids + a_ids], device=self.device)
        H = self.model.model(input_ids=x).last_hidden_state[0].float()
        t_fwd = time.time() - t0
        labels = p_ids[1:]
        T = len(labels)
        Hs = H[:T]
        mask = user_mask(self.encode, labels)

        # curve: first Adam step from δ = 0 is −lr·g/(|g|+ε) for every prefix length L
        lo = min(64, max(8, T // 8))
        grid = sorted({int(round(math.exp(v))) for v in
                       [math.log(lo) + i * (math.log(T) - math.log(lo)) / max(1, curve_points - 1)
                        for i in range(curve_points)]} | {T})
        S_full, S_mask, n_mask, nll_sum, rec = self._sums(Hs, labels, mask, None, record_at=grid)
        curve = []
        for L in grid:
            sf, sm, nm, _ = rec[L]
            row = {"L": L}
            for v in VARIANTS:
                g = self._grad(v, sf, sm, nm, L)
                step = -lr * g / (g.abs() + EPS)
                row[v] = round(float(step.norm()), 5)
                row[v + "_below_eps"] = round(float((g.abs() < EPS).float().mean()), 4)
            curve.append(row)

        # full-length run of the chosen variant (their schedule: `steps` Adam steps, continuing δ)
        z = torch.zeros(self.d, device=self.device)
        st = {"delta": z.clone(), "m": z.clone(), "v": z.clone(), "step": 0}
        g1 = self._grad(variant, S_full, S_mask, n_mask, T)
        self._adam(st, g1, lr)
        for _ in range(steps - 1):
            sf, sm, nm, _, _ = self._sums(Hs, labels, mask, st["delta"])
            self._adam(st, self._grad(variant, sf, sm, nm, T), lr)
        delta = st["delta"]
        b = delta @ self.W.T

        def prompt_loss(dl):
            sf_, sm_, nm_, nll_, _ = self._sums(Hs, labels, mask, dl)
            return nll_ / T

        result = {
            "model": self.model_id, "device": self.device, "variant": variant, "lr": lr, "steps": steps,
            "n_prompt_tokens": len(p_ids), "n_answer_tokens": len(a_ids),
            "delta_norm": round(float(delta.norm()), 6),
            "below_eps": round(float((g1.abs() < EPS).float().mean()), 4),
            "grad_median_abs": float(g1.abs().median()),
            "bias_std": round(float(b.std()), 6),
            "prompt_loss": [round(nll_sum / T, 5), round(prompt_loss(delta), 5)],
            "curve": curve,
            "top_tokens": self._top(b, p_ids, a_ids, top_k),
        }
        if a_ids:
            start = len(p_ids) - 1
            La = H[start:start + len(a_ids)] @ self.W.T
            idx = torch.tensor(a_ids, device=self.device)[:, None]
            lp0 = torch.log_softmax(La, -1).gather(1, idx)[:, 0]
            lp1 = torch.log_softmax(La + b, -1).gather(1, idx)[:, 0]
            result["answer"] = {"tokens": [self.tok.decode([i]) for i in a_ids],
                                "logprob_none": [round(float(v), 4) for v in lp0],
                                "logprob_sa": [round(float(v), 4) for v in lp1],
                                "mean_delta": round(float((lp1 - lp0).mean()), 6)}
        if generate:
            result["generation"] = {"none": self._generate(p_ids, None, max_new), "sa": self._generate(p_ids, b, max_new)}
        result["seconds"] = {"forward": round(t_fwd, 2), "total": round(time.time() - t0, 2)}
        return result
