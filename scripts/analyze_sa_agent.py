"""Summarize SA task-success runs and the run-to-run noise check.

    python scripts/analyze_sa_agent.py --tag sa-agent-14b --noise dg-14b-noise

SA result per condition and task:
  - if the condition never changed a token in the tracked no-SA run → same outcome as no-SA (exact)
  - else → the outcome of that condition's own full run (<condition>.jsonl)
Writes results/sa_agent/<tag>/summary.md
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)} if path.exists() else {}


def paired(a, b, ids, n=10000, seed=0):
    d = [int(a[i]) - int(b[i]) for i in ids]
    rng = random.Random(seed)
    boot = sorted(100 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return 100 * sum(d) / len(d), boot[int(0.025 * n)], boot[int(0.975 * n)], sum(x > 0 for x in d), sum(x < 0 for x in d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sa-agent-14b")
    ap.add_argument("--noise", default="dg-14b-noise")
    args = ap.parse_args()
    d = ROOT / "results" / "sa_agent" / args.tag
    track = read(d / "track.jsonl")
    ids = sorted(track)
    none = {i: track[i]["valid"] for i in ids}
    conds = sorted({c for r in track.values() for c in r["diverged"]} | {":".join(p.stem.rsplit("_", 2))
                                                                        for p in d.glob("sa_*.jsonl")})
    lines = [f"# SA task success: {args.tag}", "",
             f"Tasks: {len(ids)} (BFCLv3 multi-turn base, val+test). No-SA (same PyTorch loop): "
             f"**{100 * sum(none.values()) / len(ids):.1f}%** ({sum(none.values())}/{len(ids)})", "",
             "| condition | tasks where SA changed ≥1 token | success | Δ vs no-SA (pts) | 95% CI | SA-only wins / losses | missing reruns |",
             "|---|---|---|---|---|---|---|"]
    out = {"n": len(ids), "none": sum(none.values())}
    for c in conds:
        full = read(d / f"{c.replace(':', '_')}.jsonl")
        changed = [i for i in ids if c in track[i]["diverged"]]
        res, missing = {}, 0
        for i in ids:
            if i in full:
                res[i] = full[i]["valid"]
            elif i not in changed:
                res[i] = none[i]
            else:
                missing += 1
        use = [i for i in ids if i in res]
        m, lo, hi, w, l = paired(res, none, use)
        acc = 100 * sum(res[i] for i in use) / len(use)
        lines.append(f"| {c} | {len(changed)}/{len(ids)} | {acc:.1f}% | {m:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {w} / {l} | {missing} |")
        out[c] = {"changed": len(changed), "success": sum(res[i] for i in use), "n": len(use), "delta": m, "ci": [lo, hi]}
        norms = [x["delta_norms"].get(c) for r in track.values() for x in r["calls"]
                 if c in x["delta_norms"]]
        if norms:
            norms.sort()
            out[c]["median_delta_norm_along_track"] = norms[len(norms) // 2]
    lines += ["", "Δ is paired over the same tasks; the CI is a bootstrap over tasks."]

    nz = ROOT / "results" / "dg" / args.noise
    if nz.exists():
        base = ROOT / "results" / "dg"
        runs = {"none": [read(base / "dg-14b" / "eval_none.jsonl"), read(base / "dg-14b-v1-strict-explore" / "eval_none.jsonl")],
                "dg": [read(base / "dg-14b" / "eval_dg.jsonl")]}
        for p in sorted(nz.glob("eval_*_rep*.jsonl")):
            runs[p.stem.split("_")[1]].append(read(p))
        lines += ["", "## Run-to-run noise (vLLM, greedy, identical prompts)", "",
                  "| condition | runs | success per run (%) | range |", "|---|---|---|---|"]
        for k, rs in runs.items():
            accs = [100 * sum(r["valid"] for r in x.values()) / len(x) for x in rs if x]
            lines.append(f"| {k} | {len(accs)} | {', '.join(f'{a:.1f}' for a in accs)} | {max(accs) - min(accs):.1f} pts |")
            out[f"noise_{k}"] = accs
        lines.append("")
        lines.append("none runs: dg-14b (first), dg-14b-v1-strict-explore (same eval prompts), then repeats. "
                     "dg runs use the same 133 rules.")
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    (d / "summary.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
