"""Charts for the 2-page notes (report/note.html), read straight from saved results. No model needed.

    python scripts/make_note_figures.py      → report/figures/n_*.svg
"""
import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# One accent (the thing being tested) + greys (baseline / context). Nothing else.
ACCENT, GREY, LIGHT, INK, MUTED, GRID = "#1f4e8c", "#b5b4ae", "#dcdbd5", "#1d1d1b", "#6b6a66", "#ecebe6"
BLUE = ACCENT
plt.rcParams.update({
    "font.family": "Helvetica Neue", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
})
W, H = 3.35, 1.95   # every chart: column width, same height


def jl(path):
    return [json.loads(l) for l in open(ROOT / path)]


def pct(rows):
    return 100 * sum(r["valid"] for r in rows) / len(rows)


def save(fig, name):
    fig.tight_layout(pad=0.25)
    fig.savefig(OUT / name, format="svg", transparent=True)
    plt.close(fig)


def dg_runs():
    runs = [("paper", 18.5, 22.0)]
    for i, (a, b) in enumerate([("dg-14b/eval_none.jsonl", "dg-14b/eval_dg.jsonl"),
                                ("dg-14b-noise/eval_none_rep1.jsonl", "dg-14b-noise/eval_dg_rep1.jsonl"),
                                ("dg-14b-noise/eval_none_rep2.jsonl", "dg-14b-noise/eval_dg_rep2.jsonl")], 1):
        runs.append((f"run {i}", pct(jl("results/dg/" + a)), pct(jl("results/dg/" + b))))
    fig, ax = plt.subplots(figsize=(W, H))
    xs, w = np.arange(len(runs)), 0.36
    ax.bar(xs - w / 2 - 0.01, [r[1] for r in runs], w, color=GREY, label="no adaptation")
    ax.bar(xs + w / 2 + 0.01, [r[2] for r in runs], w, color=BLUE, label="with DG rules")
    for x, (_, a, b) in zip(xs, runs):
        ax.text(x - w / 2, a + 0.5, f"{a:.1f}", ha="center", fontsize=7, color=MUTED)
        ax.text(x + w / 2, b + 0.5, f"{b:.1f}", ha="center", fontsize=7, color=INK, fontweight="bold")
    ax.set_xticks(xs, [r[0] for r in runs])
    ax.set_ylabel("tasks solved (%)")
    ax.set_ylim(0, 28)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=2, frameon=False, bbox_to_anchor=(0, 1.0), borderaxespad=0.1,
              handlelength=1.2, columnspacing=1.2)
    save(fig, "n_dg.svg")


def sa_success():
    s = json.loads((ROOT / "results/sa_agent/sa-agent-14b/summary.json").read_text())
    n = s["n"]
    rows = [("no SA", 100 * s["none"] / n, ""),
            (f"SA, lr 0.1 × 1\n(changed {s['sa_paper:0.1:1']['changed']}/200)", 100 * s["sa_paper:0.1:1"]["success"] / n, ""),
            (f"SA, lr 1.0 × 3\n(changed {s['sa_paper:1.0:3']['changed']}/200)", 100 * s["sa_paper:1.0:3"]["success"] / n, "")]
    fig, ax = plt.subplots(figsize=(W, H))
    xs = np.arange(len(rows))
    ax.bar(xs, [r[1] for r in rows], 0.5, color=[GREY, ACCENT, ACCENT])
    for x, (_, v, note) in zip(xs, rows):
        ax.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=7, color=INK, fontweight="bold")
    ax.set_xticks(xs, [r[0] for r in rows])
    ax.set_ylabel("tasks solved (%)")
    ax.set_ylim(0, 28)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "n_sa.svg")


def step_vs_length():
    recs = [json.load(open(p)) for p in glob.glob(str(ROOT / "results/raw/qwen2.5-14b-bf16/multi_turn_base_*.json"))]
    T = np.array([r["n_prompt_tokens"] for r in recs])
    fig, ax = plt.subplots(figsize=(W, H))
    for v, c, lab in (("sa_exact", GREY, "exact gradient"), ("sa_paper", ACCENT, "released code")):
        y = np.array([r["sa"][v]["delta_norm"] for r in recs])
        ax.scatter(T, y, s=9, color=c, alpha=0.75, linewidths=0)
        ax.text(T.max() + 80, np.median(y[T > np.percentile(T, 90)]), lab, color=c, fontsize=7.5, va="center",
                fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlim(T.min() - 200, T.max() + 1700)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("size of SA's step")
    ax.yaxis.grid(True, color=GRID, lw=0.8, which="major")
    ax.set_axisbelow(True)
    ax.annotate("", xy=(6000, 0.108), xytext=(6000, 6.89),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, shrinkA=2, shrinkB=2))
    ax.text(6080, 0.86, "≈64× smaller\n(median)", fontsize=6.8, color=INK, va="center")
    save(fig, "n_step.svg")


def top_words():
    recs = [json.load(open(p)) for p in glob.glob(str(ROOT / "results/raw/qwen2.5-14b-bf16/multi_turn_base_*.json"))]
    cnt, gold = Counter(), Counter()
    for r in recs:
        for t in [x for x in r["sa"]["sa_exact"]["top_tokens"] if x["rank"] == "top"][:15]:
            cnt[t["token"]] += 1
            gold[t["token"]] += int(t["in_answer"])
    top = cnt.most_common(9)[::-1]
    show = lambda s: repr(s)[1:-1].replace(" ", "␣").replace("\\n", "⏎") or "∅"  # noqa: E731
    fig, ax = plt.subplots(figsize=(W, H))
    ys = np.arange(len(top))
    ax.barh(ys, [n for _, n in top], 0.62, color=GREY)
    ax.barh(ys, [gold[t] for t, _ in top], 0.62, color=ACCENT)
    for y, (t, n) in zip(ys, top):
        ax.text(n + 2, y, f"{n}", va="center", fontsize=7)
    ax.set_yticks(ys, [show(t) for t, _ in top], fontfamily="Menlo", fontsize=7.5)
    ax.set_xlim(0, 215)
    ax.set_xlabel("number of tasks (of 200)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREY, label="in SA's top 15"), Patch(color=ACCENT, label="also in the answer")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, borderaxespad=0.2)
    save(fig, "n_words.svg")


if __name__ == "__main__":
    dg_runs()
    sa_success()
    step_vs_length()
    top_words()
    print("wrote", sorted(p.name for p in OUT.glob("n_*.svg")))
