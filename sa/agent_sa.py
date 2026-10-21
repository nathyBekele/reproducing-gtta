"""SA inside the BFCL agent loop, measured by task success (PyTorch, greedy decoding).

How GTTA applies SA during an episode (their vllm/model_runner.py + BFCL base_handler.py, commit 24ba304):
- On the first model call of each user turn the handler writes a reset flag: δ, Adam m, v and step := 0.
- On every prefill (every model call), PA_STEPS Adam steps are taken on the whole prompt,
  continuing from the current δ and Adam state.
- δ is added to the final hidden states for every decoded token until the next reset.
Their server keeps one δ for whatever is in the batch; we keep one δ per task (the clean, intended version).

Speed: H (after the final norm) and the KV cache of earlier tokens don't depend on δ (δ only shifts the
logits), so each episode keeps its cache and hidden states and only new tokens are run through the model.
"none" uses the same loop with no bias, so the only difference between conditions is SA.
"""
import time

import torch

VARIANT_GRAD = {
    "sa_paper": lambda S, T: S / T / T,   # their code, masking off (extra .mean(0) over positions)
    "sa_exact": lambda S, T: S / T,       # true gradient of the mean next-token loss
}


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class AgentSA:
    def __init__(self, eng, variant="none", lr=0.1, steps=1, max_new=1024):
        from transformers import DynamicCache

        self.eng, self.variant, self.steps, self.max_new = eng, variant, int(steps), max_new
        eng.lr = float(lr)
        self.DynamicCache = DynamicCache
        tok = eng.tokenizer
        self.stop = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}

    def new_state(self):
        d = self.eng.d
        return {"ids": [], "cache": self.DynamicCache(), "H": torch.zeros(0, d, device=self.eng.device),
                "sa": None, "calls": []}

    @torch.no_grad()
    def _feed(self, st, new_ids):
        x = torch.tensor([new_ids], device=self.eng.device)
        out = self.eng.model.model(input_ids=x, past_key_values=st["cache"], use_cache=True)
        st["cache"] = out.past_key_values
        h = out.last_hidden_state[0].float()
        st["H"] = torch.cat([st["H"], h])
        st["ids"] = st["ids"] + list(new_ids)
        return h[-1]

    @torch.no_grad()
    def _align(self, st, prompt_ids, reset):
        eng, d = self.eng, self.eng.d
        if reset or st["sa"] is None:
            z = torch.zeros(d, device=eng.device)
            st["sa"] = {"delta": z.clone(), "m": z.clone(), "v": z.clone(), "step": 0}
        sa = st["sa"]
        labels = prompt_ids[1:]
        T = len(labels)
        Hs = st["H"][:T]
        for _ in range(self.steps):
            S, _, _ = eng._sums(Hs, labels, [1.0] * T, sa["delta"], need=("full",))
            sa["step"] += 1
            eng._adam(sa, VARIANT_GRAD[self.variant](S["full"], T), sa["step"])
        return eng.vocab_bias(sa["delta"]), float(sa["delta"].norm())

    @torch.no_grad()
    def generate(self, st, prompt_text, reset):
        """One model call: returns the generated text (special tokens stripped, like vLLM)."""
        t0 = time.time()
        eng = self.eng
        prompt_ids = eng.encode(prompt_text)
        n = min(common_prefix(st["ids"], prompt_ids), len(prompt_ids) - 1)
        if n < len(st["ids"]):
            st["cache"].crop(n - len(st["ids"]))   # transformers 5: negative = tokens to drop
            st["H"] = st["H"][:n]
            st["ids"] = st["ids"][:n]
        h = self._feed(st, prompt_ids[n:])
        bias, dnorm = (None, 0.0)
        if self.variant != "none":
            bias, dnorm = self._align(st, prompt_ids, reset)
        out = []
        for _ in range(self.max_new):
            logits = eng.head(h) if bias is None else eng.head(h) + bias
            t = int(torch.argmax(logits))
            if t in self.stop:
                break
            out.append(t)
            h = self._feed(st, [t])
        st["calls"].append({"T": len(prompt_ids), "reused": n, "delta_norm": round(dnorm, 5),
                            "n_out": len(out), "reset": bool(reset), "s": round(time.time() - t0, 2)})
        return eng.tokenizer.decode(out, skip_special_tokens=True)


class DivergenceTracker(AgentSA):
    """Run the no-SA agent and, at every decoded token, check whether each SA condition would have
    picked a different token (argmax of logits + bias_c). Until that first change, the SA run and the
    no-SA run are identical (same tokens → same prompts → same δ), so a task where no condition ever
    changes a token has the same outcome under SA. Only changed tasks need their own SA run."""

    def __init__(self, eng, conditions, max_new=1024):
        super().__init__(eng, "none", 0.1, 1, max_new)
        self.conditions = conditions          # {name: (variant, lr, steps)}

    def new_state(self):
        st = super().new_state()
        st["track"] = {c: None for c in self.conditions}      # per-condition Adam state while identical
        st["diverged"] = {}                                    # condition → (call index, token index)
        return st

    @torch.no_grad()
    def _adam(self, sa, g, lr):
        from .engine_torch import B1, B2, EPS, WD
        sa["step"] += 1
        sa["m"] = B1 * sa["m"] + (1 - B1) * g
        sa["v"] = B2 * sa["v"] + (1 - B2) * g * g
        m_hat = sa["m"] / (1 - B1 ** sa["step"])
        v_hat = sa["v"] / (1 - B2 ** sa["step"])
        sa["delta"] = sa["delta"] - lr * m_hat / (v_hat.sqrt() + EPS) - lr * WD * sa["delta"]

    @torch.no_grad()
    def generate(self, st, prompt_text, reset):
        t0 = time.time()
        eng = self.eng
        prompt_ids = eng.encode(prompt_text)
        n = min(common_prefix(st["ids"], prompt_ids), len(prompt_ids) - 1)
        if n < len(st["ids"]):
            st["cache"].crop(n - len(st["ids"]))
            st["H"] = st["H"][:n]
            st["ids"] = st["ids"][:n]
        h = self._feed(st, prompt_ids[n:])
        labels = prompt_ids[1:]
        T = len(labels)
        Hs = st["H"][:T]
        live = [c for c in self.conditions if c not in st["diverged"]]
        biases, norms = {}, {}
        S0 = None
        for c in live:
            variant, lr, steps = self.conditions[c]
            if reset or st["track"][c] is None:
                z = torch.zeros(eng.d, device=eng.device)
                st["track"][c] = {"delta": z.clone(), "m": z.clone(), "v": z.clone(), "step": 0}
            sa = st["track"][c]
            for _ in range(steps):
                if sa["step"] == 0:          # δ = 0: every condition shares this pass
                    if S0 is None:
                        S0, _, _ = eng._sums(Hs, labels, [1.0] * T, sa["delta"], need=("full",))
                    S = S0
                else:
                    S, _, _ = eng._sums(Hs, labels, [1.0] * T, sa["delta"], need=("full",))
                self._adam(sa, VARIANT_GRAD[variant](S["full"], T), lr)
            biases[c] = eng.vocab_bias(sa["delta"])
            norms[c] = round(float(sa["delta"].norm()), 5)
        call_idx = len(st["calls"])
        out = []
        for i in range(self.max_new):
            logits = eng.head(h)
            t = int(torch.argmax(logits))
            for c in list(biases):
                if int(torch.argmax(logits + biases[c])) != t:
                    st["diverged"][c] = (call_idx, i)
                    del biases[c]
            if t in self.stop:
                break
            out.append(t)
            h = self._feed(st, [t])
        st["calls"].append({"T": len(prompt_ids), "reused": n, "delta_norms": norms, "n_out": len(out),
                            "reset": bool(reset), "s": round(time.time() - t0, 2)})
        return eng.tokenizer.decode(out, skip_special_tokens=True)


def track_episode(tracker, ep):
    st = tracker.new_state()
    while not ep.done:
        ep.consume(tracker.generate(st, ep.prompt(), reset=(ep.step == 0)))
    return st["calls"], {c: list(v) for c, v in st["diverged"].items()}


def run_episode(agent, ep):
    """Drive one dg.Episode to the end. Reset δ on the first call of each user turn (their handler)."""
    st = agent.new_state()
    while not ep.done:
        ep.consume(agent.generate(st, ep.prompt(), reset=(ep.step == 0)))
    calls = st["calls"]
    del st
    return calls
