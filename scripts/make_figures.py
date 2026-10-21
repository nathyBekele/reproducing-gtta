"""Report figures from saved results (no model needed). Output: report/figures/*.png

    python scripts/make_figures.py
"""
import glob
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
C_PAPER, C_MASK, C_EXACT, C_GREY = "#d1495b", "#edae49", "#00798c", "#8d99ae"


def load(tag):
    return {r["id"]: r for r in (json.load(open(p)) for p in glob.glob(str(ROOT / "results" / "raw" / tag / "multi_turn_base_*.json")))}


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x)
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fig_delta_vs_length():
    base, dg = load("qwen2.5-14b-bf16"), load("qwen2.5-14b-bf16-dg")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for recs, marker, suffix in ((base, "o", ""), (dg, "^", " (+DG rules)")):
        T = np.array([r["n_prompt_tokens"] for r in recs.values()])
        for v, c, lab in (("sa_paper", C_PAPER, "their code (paper setting)"), ("sa_exact", C_EXACT, "true gradient")):
            n = np.array([r["sa"][v]["delta_norm"] for r in recs.values()])
            ax.scatter(T, n, s=12, marker=marker, color=c, alpha=0.6 if suffix == "" else 0.35, label=lab + suffix)
    ax.set_yscale("log")
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("‖δ‖ after one SA step (log scale)")
    ax.set_title("Qwen2.5-14B, lr 0.1, 1 step: SA's step shrinks with prompt length")
    ax.legend(fontsize=7.5, loc="center right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "delta_vs_length.png", dpi=160)


def fig_gold_effect():
    settings = [("qwen2.5-14b-bf16", "lr 0.1, 1 step"), ("qwen2.5-14b-bf16-lr1.0-s1", "lr 1.0, 1 step"),
                ("qwen2.5-14b-bf16-lr1.0-s3", "lr 1.0, 3 steps\n(their best 14B)")]
    variants = [("sa_paper", C_PAPER, "their code, mask off"), ("sa_code_mask", C_MASK, "their code, mask on"),
                ("random_sign_like_sa_paper", C_GREY, "random δ, same size")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w = 0.25
    for j, (v, c, lab) in enumerate(variants):
        for i, (tag, _) in enumerate(settings):
            recs = [r for r in load(tag).values() if r["split"] == "test"]
            m, lo, hi = boot_ci([r["conditions"][v]["mean"] - r["conditions"]["none"]["mean"] for r in recs])
            ax.bar(i + (j - 1) * w, m, w, color=c, label=lab if i == 0 else None)
            ax.errorbar(i + (j - 1) * w, m, yerr=[[m - lo], [hi - m]], color="k", capsize=3, lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(settings)), [s for _, s in settings], fontsize=8)
    ax.set_ylabel("Δ mean log-prob of gold call\nvs. no adaptation (test, n=160)", fontsize=8)
    ax.set_title("Qwen2.5-14B: SA does not make the correct first call more likely", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "gold_effect.png", dpi=160)


def fig_dg_success():
    rows = {n: {json.loads(l)["id"]: json.loads(l)["valid"] for l in open(ROOT / "results" / "dg" / "dg-14b" / f"eval_{n}.jsonl")}
            for n in ("none", "dg")}
    ids = list(rows["none"])
    acc = {n: 100 * np.mean([rows[n][i] for i in ids]) for n in rows}
    diffs = [int(rows["dg"][i]) - int(rows["none"][i]) for i in ids]
    rng = random.Random(0)
    boot = sorted(100 * sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(10000))
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    xs = np.arange(2)
    ax.bar(xs - 0.18, [18.5, 22.0], 0.36, color=C_GREY, label="paper (Table 2)")
    ax.bar(xs + 0.18, [acc["none"], acc["dg"]], 0.36, color=C_EXACT, label="ours (n=200)")
    for x, v in zip(xs + 0.18, [acc["none"], acc["dg"]]):
        ax.text(x, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    for x, v in zip(xs - 0.18, [18.5, 22.0]):
        ax.text(x, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(xs, ["no adaptation", "+ DG"])
    ax.set_ylabel("BFCLv3 multi-turn base accuracy (%)")
    ax.set_title(f"Qwen2.5-14B: DG +{acc['dg'] - acc['none']:.1f} pts (95% CI {boot[250]:+.1f} to {boot[9750]:+.1f})", fontsize=9)
    ax.set_ylim(0, 27)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "dg_success.png", dpi=160)


# ---------- column-width figures for the 2-page note ----------
def _small():
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})


def fig_note_dg():
    _small()
    def acc(path):
        rows = [json.loads(l) for l in open(ROOT / path)]
        return 100 * sum(r["valid"] for r in rows) / len(rows)
    runs = [("paper", 18.5, 22.0),
            ("our run 1", acc("results/dg/dg-14b/eval_none.jsonl"), acc("results/dg/dg-14b/eval_dg.jsonl")),
            ("our run 2", acc("results/dg/dg-14b-noise/eval_none_rep1.jsonl"), acc("results/dg/dg-14b-noise/eval_dg_rep1.jsonl")),
            ("our run 3", acc("results/dg/dg-14b-noise/eval_none_rep2.jsonl"), acc("results/dg/dg-14b-noise/eval_dg_rep2.jsonl"))]
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    xs = np.arange(len(runs)); w = 0.36
    ax.bar(xs - w / 2, [r[1] for r in runs], w, color=C_GREY, label="no adaptation")
    ax.bar(xs + w / 2, [r[2] for r in runs], w, color=C_EXACT, label="+ DG rules")
    for x, (_, a, b) in zip(xs, runs):
        ax.text(x - w / 2, a + 0.4, f"{a:.1f}", ha="center", fontsize=6.5)
        ax.text(x + w / 2, b + 0.4, f"{b:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(xs, [r[0] for r in runs])
    ax.set_ylabel("tasks solved (%)")
    ax.set_ylim(0, 27)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "note_dg.png", dpi=220)


def fig_note_delta():
    _small()
    base = load("qwen2.5-14b-bf16")
    T = np.array([r["n_prompt_tokens"] for r in base.values()])
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for v, c, lab in (("sa_exact", C_EXACT, "true gradient"), ("sa_paper", C_PAPER, "released code")):
        ax.scatter(T, [r["sa"][v]["delta_norm"] for r in base.values()], s=6, color=c, alpha=0.7, label=lab, linewidths=0)
    ax.set_yscale("log")
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("size of SA's step, ‖δ‖")
    ax.legend(loc="center right", frameon=False, markerscale=2)
    ax.grid(alpha=0.3, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "note_delta.png", dpi=220)


if __name__ == "__main__":
    fig_delta_vs_length()
    fig_gold_effect()
    fig_dg_success()
    fig_note_dg()
    fig_note_delta()
    print("figures written to", OUT)
