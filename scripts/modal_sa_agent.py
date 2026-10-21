"""SA measured by BFCLv3 multi-turn task success, on Modal GPUs (PyTorch, greedy).

Each condition is "name" or "variant:lr:steps", e.g.
    none                  no adaptation (same loop, no bias)
    sa_paper:0.1:1        their code, the paper's default setting
    sa_paper:1.0:3        their code, their best 14B setting
    sa_exact:0.1:1        true gradient
Every (condition, shard) runs in its own container in parallel; results stream back per task.

    modal run scripts/modal_sa_agent.py --model Qwen/Qwen2.5-1.5B-Instruct --tag sa-agent-1.5b-test --gpu L4 --limit 4 --timeout-min 15
    modal run scripts/modal_sa_agent.py --model Qwen/Qwen2.5-14B-Instruct --tag sa-agent-14b --gpu H100 --shards 2 --timeout-min 60

Outputs: results/sa_agent/<tag>/<condition>.jsonl (resumable: finished task ids are skipped).
Safety: hard timeout per container; the app stops when the local command ends.
"""
import json
import sys
import threading
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "transformers==5.20.2", "numpy", "safetensors", "huggingface_hub", "mpmath")
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "sa", "/root/sa")
    .add_local_dir(ROOT / "data" / "bfcl_v3", "/root/data/bfcl_v3")
    .add_local_dir(ROOT / "third_party", "/root/third_party")
)
app = modal.App("sa-agent", image=image)


def parse_condition(c):
    if c == "none":
        return "none", 0.1, 1
    v, lr, steps = c.split(":")
    return v, float(lr), int(steps)


@app.function(timeout=60 * 60)
def run(model: str, condition: str, shard: int, n_shards: int, splits: str, limit: int, skip: list, only: list = None):
    import time

    sys.path.insert(0, "/root")
    from sa import dg
    from sa.agent_sa import AgentSA, run_episode
    from sa.engine_torch import TorchEngine

    t0 = time.time()
    eng = TorchEngine(model)
    variant, lr, steps = parse_condition(condition)
    agent = AgentSA(eng, variant, lr, steps)
    yield {"log": f"[{condition} shard {shard}] model loaded in {time.time() - t0:.0f}s"}
    entries = []
    for split in splits.split(","):
        entries += dg.eval_entries(split)[: (limit or None)]
    entries = [e for i, e in enumerate(entries) if i % n_shards == shard and e[0]["id"] not in set(skip)]
    if only is not None:
        entries = [e for e in entries if e[0]["id"] in set(only)]
    ns = f"sa_agent_{shard}"
    for entry, gt in entries:
        t = time.time()
        ep = dg.eval_episodes([(entry, gt)], ns)[0]
        calls = run_episode(agent, ep)
        sc = dg.score(ep, gt, ns)
        yield {"row": {"id": entry["id"], **sc, "n_turns": len(ep.results),
                       "n_steps": sum(len(tu) for tu in ep.results), "results": ep.results,
                       "calls": calls, "seconds": round(time.time() - t, 1)}}


@app.function(timeout=60 * 60)
def track(model: str, conditions: str, shard: int, n_shards: int, splits: str, limit: int, skip: list):
    """No-SA run that records, per task, where each SA condition would first pick a different token."""
    import time

    sys.path.insert(0, "/root")
    from sa import dg
    from sa.agent_sa import DivergenceTracker, track_episode
    from sa.engine_torch import TorchEngine

    t0 = time.time()
    eng = TorchEngine(model)
    tracker = DivergenceTracker(eng, {c: parse_condition(c) for c in conditions.split(",")})
    yield {"log": f"[track shard {shard}] model loaded in {time.time() - t0:.0f}s"}
    entries = []
    for split in splits.split(","):
        entries += dg.eval_entries(split)[: (limit or None)]
    entries = [e for i, e in enumerate(entries) if i % n_shards == shard and e[0]["id"] not in set(skip)]
    ns = f"track_{shard}"
    for entry, gt in entries:
        t = time.time()
        ep = dg.eval_episodes([(entry, gt)], ns)[0]
        calls, diverged = track_episode(tracker, ep)
        sc = dg.score(ep, gt, ns)
        yield {"row": {"id": entry["id"], **sc, "n_turns": len(ep.results),
                       "n_steps": sum(len(tu) for tu in ep.results), "results": ep.results,
                       "diverged": diverged, "calls": calls, "seconds": round(time.time() - t, 1)}}


@app.local_entrypoint()
def main(model: str, tag: str, conditions: str = "sa_paper:0.1:1,sa_paper:1.0:3", gpu: str = "L4",
         shards: int = 1, timeout_min: int = 15, splits: str = "val,test", limit: int = 0, mode: str = "track"):
    """mode=track: one no-SA run per shard, recording where each SA condition would change a token
                   → track.jsonl.  mode=rerun: full SA runs only for tasks where that condition changed
                   a token → <condition>.jsonl.  mode=full: full SA runs for every task."""
    out = ROOT / "results" / "sa_agent" / tag
    out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    print(f"mode={mode} GPU={gpu} shards={shards}, hard limit {timeout_min} min per container")

    def done_ids(path):
        return [json.loads(l)["id"] for l in open(path)] if path.exists() else []

    def stream(gen, path, label):
        n = 0
        for msg in gen:
            with lock:
                if "log" in msg:
                    print(msg["log"], flush=True)
                    continue
                with open(path, "a") as f:
                    f.write(json.dumps(msg["row"]) + "\n")
                n += 1
                r = msg["row"]
                extra = f" diverged={list(r['diverged'])}" if "diverged" in r else ""
                print(f"[{label}] {n} {r['id']} valid={r['valid']} steps={r['n_steps']} {r['seconds']}s{extra}", flush=True)

    jobs = []
    if mode == "track":
        path = out / "track.jsonl"
        fn = track.with_options(gpu=gpu, timeout=timeout_min * 60)
        skip = done_ids(path)
        for s in range(shards):
            jobs.append((fn.remote_gen(model, conditions, s, shards, splits, limit, skip), path, f"track s{s}"))
    else:
        fn = run.with_options(gpu=gpu, timeout=timeout_min * 60)
        tracked = [json.loads(l) for l in open(out / "track.jsonl")] if mode == "rerun" else []
        for c in conditions.split(","):
            path = out / f"{c.replace(':', '_')}.jsonl"
            only = [r["id"] for r in tracked if c in r["diverged"]] if mode == "rerun" else None
            if only is not None:
                print(f"{c}: {len(only)} tasks changed by SA → rerun")
                if not only:
                    continue
            skip = done_ids(path)
            for s in range(shards):
                jobs.append((fn.remote_gen(model, c, s, shards, splits, limit, skip, only), path, f"{c} s{s}"))
    threads = [threading.Thread(target=stream, args=j) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("done")
