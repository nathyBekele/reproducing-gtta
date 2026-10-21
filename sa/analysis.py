"""Summaries from saved per-example results (no model needed)."""
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

SA_VARIANTS = ["sa_paper", "sa_code_mask", "sa_exact"]
TOP_K = 20
N_BOOT = 10000


def load(raw_dir):
    raw_dir = Path(raw_dir)
    config = json.loads((raw_dir / "config.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted(raw_dir.glob("multi_turn_base_*.json"))]
    return config, rows


def bootstrap_ci(values, n=N_BOOT, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def best_alpha(rows, metric="mean"):
    """Choose the copy-rival α on the val split only."""
    val = [r for r in rows if r["split"] == "val"]
    alphas = sorted({c["alpha"] for r in val for c in r["conditions"].values() if "alpha" in c})
    scores = {a: np.mean([r["conditions"][f"copy_alpha_{a}"][metric] for r in val]) for a in alphas}
    return max(scores, key=scores.get), scores


def token_type(tok):
    s = tok.strip()
    if s == "":
        return "whitespace"
    if re.fullmatch(r"[\W_]+", s):
        return "punctuation"
    if re.fullmatch(r"\d+", s):
        return "number"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
        return "word/identifier"
    return "other"


def boosted_tokens(rows, variant, k=TOP_K):
    flags = ["in_prompt", "in_system", "in_user", "in_answer"]
    frac = {f: [] for f in flags}
    types, counts, sets = Counter(), Counter(), []
    for r in rows:
        top = [t for t in r["sa"][variant]["top_tokens"] if t["rank"] == "top"][:k]
        for f in flags:
            frac[f].append(np.mean([t[f] for t in top]))
        types.update(token_type(t["token"]) for t in top)
        counts.update(t["token"] for t in top)
        sets.append({t["id"] for t in top})
    pairs = list(combinations(range(len(sets)), 2))
    jac = [len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i, j in pairs] if pairs else [float("nan")]
    total = sum(types.values())
    return {
        "k": k,
        "frac_in": {f: float(np.mean(v)) for f, v in frac.items()},
        "type_share": {t: c / total for t, c in types.most_common()},
        "mean_pairwise_jaccard": float(np.mean(jac)),
        "most_common": counts.most_common(15),
    }


def condition_table(rows, conditions, metric="mean"):
    """Δ vs. 'none' (paired, per example), with bootstrap CI and % improved."""
    base = np.array([r["conditions"]["none"][metric] for r in rows])
    out = {"none": {"value": float(base.mean())}}
    for c in conditions:
        vals = np.array([r["conditions"][c][metric] for r in rows])
        diff = vals - base
        m, lo, hi = bootstrap_ci(diff)
        out[c] = {"value": float(vals.mean()), "delta": m, "ci95": [lo, hi], "pct_improved": float((diff > 0).mean())}
    return out


def paired(rows, a, b, metric="mean"):
    diff = np.array([r["conditions"][a][metric] - r["conditions"][b][metric] for r in rows])
    m, lo, hi = bootstrap_ci(diff)
    return {"a": a, "b": b, "delta": m, "ci95": [lo, hi], "pct_a_better": float((diff > 0).mean())}


def sa_diagnostics(rows, variant):
    T = np.array([r["n_prompt_tokens"] for r in rows])
    s = [r["sa"][variant] for r in rows]
    norm = np.array([x["delta_norm"] for x in s])
    dloss = np.array([x["loss_after"] - x["loss_before"] for x in s])

    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    return {
        "delta_norm_median": float(np.median(norm)),
        "delta_norm_range": [float(norm.min()), float(norm.max())],
        "frac_grad_below_eps_median": float(np.median([x["frac_grad_below_eps"] for x in s])),
        "prompt_loss_change_median": float(np.median(dloss)),
        "spearman_norm_vs_prompt_len": spearman(norm, T),
    }


def summarize(raw_dir):
    config, rows = load(raw_dir)
    alpha, alpha_scores = best_alpha(rows)
    test = [r for r in rows if r["split"] == "test"]
    copy = f"copy_alpha_{alpha}"
    conds = SA_VARIANTS + [copy] + [f"random_{k}_like_{v}" for v in SA_VARIANTS for k in ("gaussian", "sign")]
    return {
        "model": config["model"], "tag": config["tag"],
        "n": {"all": len(rows), "test": len(test), "val": len(rows) - len(test)},
        "copy_alpha_chosen_on_val": alpha, "copy_alpha_val_scores": alpha_scores,
        "q1_boosted_tokens": {v: boosted_tokens(rows, v) for v in SA_VARIANTS},
        "q2_q4_test_mean_logprob": condition_table(test, conds, "mean"),
        "q2_q4_test_sum_logprob": condition_table(test, conds, "sum"),
        "paired_test": [paired(test, "sa_exact", copy), paired(test, "sa_exact", "random_sign_like_sa_exact"),
                        paired(test, "sa_exact", "sa_paper"), paired(test, "sa_paper", "random_sign_like_sa_paper")],
        "sa_diagnostics": {v: sa_diagnostics(rows, v) for v in SA_VARIANTS},
    }


def to_markdown(s):
    lines = [f"# Results: {s['tag']} ({s['model']})", "",
             f"Examples: {s['n']['all']} (val {s['n']['val']}, test {s['n']['test']}). "
             f"Copy-rival α chosen on val: **{s['copy_alpha_chosen_on_val']}**.", "",
             "## Q1: what δ boosts (top-20 tokens of δWᵀ, all examples)", "",
             "| variant | in prompt | in system | in user | in answer | whitespace+punct | identifiers | mean pairwise Jaccard |",
             "|---|---|---|---|---|---|---|---|"]
    for v, q in s["q1_boosted_tokens"].items():
        ts = q["type_share"]
        lines.append(f"| {v} | {q['frac_in']['in_prompt']:.0%} | {q['frac_in']['in_system']:.0%} | {q['frac_in']['in_user']:.0%} | "
                     f"{q['frac_in']['in_answer']:.0%} | {ts.get('whitespace', 0) + ts.get('punctuation', 0):.0%} | "
                     f"{ts.get('word/identifier', 0):.0%} | {q['mean_pairwise_jaccard']:.2f} |")
    for v, q in s["q1_boosted_tokens"].items():
        lines.append(f"\nMost often boosted ({v}): " + ", ".join(f"`{t!r}`×{c}" for t, c in q["most_common"][:10]))
    lines += ["", "## Q2–Q4: log-prob of the gold answer (test, mean per token; Δ vs. no adaptation)", "",
              "| condition | mean log-prob | Δ | 95% CI | % examples improved |", "|---|---|---|---|---|"]
    for c, v in s["q2_q4_test_mean_logprob"].items():
        if c == "none":
            lines.append(f"| none | {v['value']:.4f} | – | – | – |")
        else:
            lines.append(f"| {c} | {v['value']:.4f} | {v['delta']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['pct_improved']:.0%} |")
    lines += ["", "## Paired comparisons (test, mean per token)", "", "| A − B | Δ | 95% CI | % A better |", "|---|---|---|---|"]
    for p in s["paired_test"]:
        lines.append(f"| {p['a']} − {p['b']} | {p['delta']:+.4f} | [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | {p['pct_a_better']:.0%} |")
    lines += ["", "## SA diagnostics (all examples)", "",
              "| variant | median ‖δ‖ | ‖δ‖ range | median % \\|g\\|<ε | median Δ prompt loss | Spearman(‖δ‖, prompt length) |",
              "|---|---|---|---|---|---|"]
    for v, d in s["sa_diagnostics"].items():
        lines.append(f"| {v} | {d['delta_norm_median']:.3f} | {d['delta_norm_range'][0]:.3f}–{d['delta_norm_range'][1]:.3f} | "
                     f"{d['frac_grad_below_eps_median']:.0%} | {d['prompt_loss_change_median']:+.5f} | {d['spearman_norm_vs_prompt_len']:+.2f} |")
    return "\n".join(lines) + "\n"
