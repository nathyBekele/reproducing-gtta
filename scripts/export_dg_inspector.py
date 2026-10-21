"""Export everything the DG inspector page needs into one small JS file. No model needed.

    python scripts/export_dg_inspector.py      → explorer/dg_data.js

Sources (all saved by scripts/modal_dg.py):
  results/dg/dg-14b/{dynamics.json, rules_raw.json, exploration.jsonl, eval_none.jsonl, eval_dg.jsonl}
  results/dg/dg-14b-noise/eval_{none,dg}_rep{1,2}.jsonl      (two repeats with identical prompts)
  data/bfcl_v3/  (tasks, correct calls, exploration goals; GTTA commit 24ba304)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sa import data  # noqa: E402
from sa.dg import parse_json_array  # noqa: E402

DG = ROOT / "results" / "dg"
CUT = 700   # characters kept per model output / tool output (enough to read; keeps the file small)


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def by_id(path):
    return {r["id"]: r for r in jl(path)}


def clip(s, n=CUT):
    s = str(s)
    return s if len(s) <= n else s[:n] + " …"


def rule(r):
    return {"s": r.get("initial_state", ""), "a": r.get("action_taken", ""), "d": r.get("environmental_dynamics", "")}


def main():
    # ---- runs: valid flags for 3 runs × {none, dg} ----
    runs = [("dg-14b/eval_none.jsonl", "dg-14b/eval_dg.jsonl"),
            ("dg-14b-noise/eval_none_rep1.jsonl", "dg-14b-noise/eval_dg_rep1.jsonl"),
            ("dg-14b-noise/eval_none_rep2.jsonl", "dg-14b-noise/eval_dg_rep2.jsonl")]
    loaded = [(by_id(DG / a), by_id(DG / b)) for a, b in runs]
    run_summary = [{"none": round(100 * sum(r["valid"] for r in n.values()) / len(n), 1),
                    "dg": round(100 * sum(r["valid"] for r in d.values()) / len(d), 1)} for n, d in loaded]

    # ---- exploration episodes and the rules written from each ----
    goals = {g["id"]: g for g in jl(data.DATA_DIR / "dg" / "multi_turn_exploration.json")}
    expl = jl(DG / "dg-14b" / "exploration.jsonl")
    raw = json.load(open(DG / "dg-14b" / "rules_raw.json"))
    episodes, n_steps = [], 0
    for i, (e, text) in enumerate(zip(expl, raw["texts"])):
        steps = e["log"][0] if e["log"] else []
        n_steps += len(steps)
        parsed = parse_json_array(text)
        rules = [rule(r) for r in (parsed or []) if isinstance(r, dict)]
        g = goals.get(e["id"], {})
        goal = g.get("question", [[{"content": ""}]])[0][0]["content"].replace("Goal:", "").strip()
        episodes.append({"id": e["id"], "classes": e["classes"], "goal": goal, "unparsed": parsed is None,
                         "steps": [{"a": clip(s.get("assistant", "")), "t": clip((s.get("tool") or [""])[0], 500),
                                    "err": "tool" not in s} for s in steps],
                         "rules": rules})

    # ---- kept rules, with the episode(s) whose raw rules contain the same action ----
    dyn = json.load(open(DG / "dg-14b" / "dynamics.json"))["dynamics"]
    kept = {}
    for cls, rules in dyn.items():
        out = []
        for r in rules:
            rr = rule(r)
            src = [k for k, ep in enumerate(episodes) if cls in ep["classes"] and any(x["a"].strip() == rr["a"].strip() for x in ep["rules"])]
            rr["src"] = src[:3]
            out.append(rr)
        kept[cls] = out
    raw_counts = {cls: len(v) for cls, v in raw["rules"].items()}

    # ---- tasks: user turns, correct calls, run-1 trajectories, and pass/fail in all 3 runs ----
    tasks = []
    for split in ("val", "test"):
        qs = jl(data.DATA_DIR / data.SPLIT_FILES[split])
        ans = by_id(data.DATA_DIR / "possible_answer" / data.SPLIT_FILES[split])
        for t in qs:
            tid = t["id"]
            n1, d1 = loaded[0][0][tid], loaded[0][1][tid]
            tasks.append({
                "id": tid, "split": split, "classes": t["involved_classes"],
                "turns": [(turn[0]["content"] if turn else "") for turn in t["question"]],
                "gold": ans[tid]["ground_truth"],
                "none": {"ok": [loaded[i][0][tid]["valid"] for i in range(3)], "err": n1.get("error_type"),
                         "steps": [[clip(x) for x in turn] for turn in n1["results"]]},
                "dg": {"ok": [loaded[i][1][tid]["valid"] for i in range(3)], "err": d1.get("error_type"),
                       "steps": [[clip(x) for x in turn] for turn in d1["results"]]},
            })
    tasks.sort(key=lambda x: int(x["id"].rsplit("_", 1)[1]))

    out = {
        "source": "r2llab/GTTA@24ba304", "model": "Qwen2.5-14B-Instruct",
        "runs": run_summary,
        "counts": {"episodes": len(episodes), "steps": n_steps, "raw": sum(raw_counts.values()),
                   "unparsed": raw["unparsed"], "kept": sum(len(v) for v in kept.values()),
                   "per_class": {c: {"raw": raw_counts.get(c, 0), "kept": len(kept[c])} for c in sorted(kept)}},
        "rules": kept, "episodes": episodes, "tasks": tasks,
    }
    dest = ROOT / "explorer" / "dg_data.js"
    dest.write_text("window.DG_DATA = " + json.dumps(out, separators=(",", ":"), ensure_ascii=False) + ";\n")
    print(f"wrote {dest} ({dest.stat().st_size / 1e3:.0f} KB): {len(tasks)} tasks, {out['counts']}")


if __name__ == "__main__":
    main()
