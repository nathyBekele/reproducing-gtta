"""Export the explorer's data (one small JS file) from saved results. No model needed.

    python scripts/export_explorer.py      → explorer/data.js
"""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VARS = {"paper": "sa_paper", "mask": "sa_code_mask", "exact": "sa_exact"}


def load(tag):
    return {r["id"]: r for r in (json.load(open(p)) for p in glob.glob(str(ROOT / "results" / "raw" / tag / "multi_turn_base_*.json")))}


def jsonl(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)} if Path(path).exists() else {}


def r3(x):
    return round(float(x), 3)


def main():
    base, dg, best = load("qwen2.5-14b-bf16"), load("qwen2.5-14b-bf16-dg"), load("qwen2.5-14b-bf16-lr1.0-s3")
    track = jsonl(ROOT / "results/sa_agent/sa-agent-14b/track.jsonl")
    sa01 = jsonl(ROOT / "results/sa_agent/sa-agent-14b/sa_paper_0.1_1.jsonl")
    sa10 = jsonl(ROOT / "results/sa_agent/sa-agent-14b/sa_paper_1.0_3.jsonl")
    v_none = jsonl(ROOT / "results/dg/dg-14b/eval_none.jsonl")
    v_dg = jsonl(ROOT / "results/dg/dg-14b/eval_dg.jsonl")
    tasks = []
    for tid in sorted(base, key=lambda s: int(s.rsplit("_", 1)[1])):
        r = base[tid]
        none_lp = r["conditions"]["none"]["logprobs"]
        t = {"id": tid, "split": r["split"], "classes": r["involved_classes"], "request": r["user_request"],
             "answer": r["answer"], "tokens": r["answer_tokens"], "T": r["n_prompt_tokens"],
             "T_dg": dg[tid]["n_prompt_tokens"] if tid in dg else None,
             "none_lp": [r3(x) for x in none_lp], "v": {}}
        for k, name in VARS.items():
            sa, cond = r["sa"][name], r["conditions"][name]
            t["v"][k] = {
                "norm": round(sa["delta_norm"], 4),
                "below_eps": r3(sa["frac_grad_below_eps"]),
                "loss": [round(sa["loss_before"], 5), round(sa["loss_after"], 5)],
                "d_gold": round(cond["mean"] - r["conditions"]["none"]["mean"], 5),
                "tok": [r3(a - b) for a, b in zip(cond["logprobs"], none_lp)],
                "top": [[x["token"], round(x["bias"], 5), int(x["in_prompt"]), int(x["in_answer"])]
                        for x in sa["top_tokens"] if x["rank"] == "top"][:15],
            }
        if tid in dg:
            t["norm_dg"] = {k: round(dg[tid]["sa"][n]["delta_norm"], 4) for k, n in VARS.items()}
        if tid in best:
            b = best[tid]
            t["best"] = {"norm": round(b["sa"]["sa_paper"]["delta_norm"], 3),
                         "d_gold": round(b["conditions"]["sa_paper"]["mean"] - b["conditions"]["none"]["mean"], 5)}
        t["copy"] = round(r["conditions"]["copy_alpha_0.25"]["mean"] - r["conditions"]["none"]["mean"], 5)
        tr = track.get(tid)
        if tr:
            ch = tr["diverged"]
            t["success"] = {
                "pt_none": tr["valid"],
                "sa01": sa01[tid]["valid"] if tid in sa01 else tr["valid"], "sa01_changed": "sa_paper:0.1:1" in ch,
                "sa10": sa10[tid]["valid"] if tid in sa10 else None, "sa10_changed": "sa_paper:1.0:3" in ch,
                "vllm_none": v_none[tid]["valid"] if tid in v_none else None,
                "vllm_dg": v_dg[tid]["valid"] if tid in v_dg else None,
            }
        tasks.append(t)

    summary = json.loads((ROOT / "results/sa_agent/sa-agent-14b/summary.json").read_text())
    data = {"model": "Qwen2.5-14B-Instruct (bf16)", "benchmark": "BFCLv3 multi-turn base (GTTA split, 200 tasks)",
            "source_commit": "r2llab/GTTA@24ba304", "tasks": tasks, "sa_agent_summary": summary}
    out = ROOT / "explorer" / "data.js"
    out.write_text("window.EXPLORER_DATA = " + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB, {len(tasks)} tasks)")


if __name__ == "__main__":
    main()
