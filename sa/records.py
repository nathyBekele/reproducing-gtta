"""Shared result-record helpers, so every engine (MLX, PyTorch) writes the same JSON format."""
import numpy as np

# SA variants (see notes/decisions.md): their code with/without user masking, and the true gradient.
VARIANTS = {
    "sa_paper": dict(mode="code", mask_user=False),
    "sa_code_mask": dict(mode="code", mask_user=True),
    "sa_exact": dict(mode="exact", mask_user=False),
}


def token_sets(encode, ex, prompt_ids, answer_ids):
    system_part = ex.prompt.split("<|im_start|>user")[0]
    return {
        "prompt": set(prompt_ids),
        "system": set(encode(system_part)),
        "user": set(encode(ex.user_request)),
        "answer": set(answer_ids),
    }


def top_tokens(decode, bias, sets, k):
    """Top-k boosted and bottom-10 suppressed tokens of a vocabulary bias, with membership flags."""
    b = np.asarray(bias, dtype=np.float64)
    order = np.argsort(-b)
    rows = []
    for n, idx in enumerate(list(order[:k]) + list(order[-10:][::-1])):
        idx = int(idx)
        rows.append({
            "id": idx,
            "token": decode([idx]),
            "bias": float(b[idx]),
            "rank": "top" if n < k else "bottom",
            **{f"in_{name}": idx in s for name, s in sets.items()},
        })
    return rows


def summarize_logprobs(lps):
    return {"sum": float(sum(lps)), "mean": float(sum(lps) / len(lps))}


def new_record(ex, decode, prompt_ids, answer_ids):
    return {
        "id": ex.id, "split": ex.split, "involved_classes": ex.involved_classes,
        "user_request": ex.user_request, "answer": ex.answer,
        "answer_tokens": [decode([t]) for t in answer_ids],
        "n_prompt_tokens": len(prompt_ids), "n_answer_tokens": len(answer_ids),
        "conditions": {}, "sa": {},
    }
