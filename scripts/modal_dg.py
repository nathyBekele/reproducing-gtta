"""Dynamics Grounding on BFCLv3 with vLLM on a Modal GPU: explore → extract → filter → evaluate.

    modal run scripts/modal_dg.py --model Qwen/Qwen2.5-1.5B-Instruct --tag dg-1.5b-test --gpu L4 --limit 3 --timeout-min 20
    modal run scripts/modal_dg.py --model Qwen/Qwen2.5-14B-Instruct --tag dg-14b --gpu H100 --timeout-min 60

Outputs (local): results/dg/<tag>/{exploration.jsonl, rules_raw.json, dynamics.json, eval_none.jsonl, eval_dg.jsonl, summary.json}
Safety: one container, a hard timeout, nothing left running. The model is the explorer, extractor and filter
(the paper used GPT-4.1 / GPT-4o / o3); see notes/decisions.md.
"""
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.32.0", "mpmath", "numpy")
    # The slim image has no CUDA compiler, so avoid FlashInfer's JIT kernels entirely:
    # FlashAttention backend for attention, plain PyTorch sampling.
    .env({"TOKENIZERS_PARALLELISM": "false", "VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
    .add_local_dir(ROOT / "sa", "/root/sa")
    .add_local_dir(ROOT / "data" / "bfcl_v3", "/root/data/bfcl_v3")
    .add_local_dir(ROOT / "third_party", "/root/third_party")
)
app = modal.App("sa-dg", image=image)


@app.function(timeout=60 * 60)
def pipeline(model: str, limit: int, splits: str):
    """Generator yielding (kind, payload) artifacts as each stage finishes."""
    import time

    sys.path.insert(0, "/root")
    import vllm
    from vllm import LLM, SamplingParams

    from sa import dg

    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=0.90,
              enable_prefix_caching=True)
    yield "log", f"vLLM {vllm.__version__}: model loaded in {time.time() - t0:.0f}s"

    def gen(max_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        def f(prompts):
            outs = llm.generate(prompts, sp, use_tqdm=False)
            return [o.outputs[0].text for o in outs]
        return f

    # 1) exploration
    t = time.time()
    eps = dg.exploration_episodes("explore")
    if limit:
        eps = eps[:: max(1, len(eps) // (limit * 8))][: limit * 8]   # a few per environment
    dg.run_episodes(gen(512), eps)
    exploration = [{"id": e.entry["id"], "classes": e.entry["involved_classes"], "log": e.log} for e in eps]
    yield "exploration", exploration
    yield "log", f"exploration: {len(eps)} goals, {sum(len(e.log[0]) for e in eps if e.log)} steps, {time.time() - t:.0f}s"

    # 2) extraction (one prompt per exploration log)
    t = time.time()
    raw_texts = gen(2048)([dg.extraction_prompt(e) for e in eps])
    rules_raw = {}
    n_fail = 0
    for e, text in zip(eps, raw_texts):
        cls = e.entry["involved_classes"][0]
        parsed = dg.parse_json_array(text)
        if parsed is None:
            n_fail += 1
            continue
        rules_raw.setdefault(cls, []).extend(r for r in parsed if isinstance(r, dict))
    yield "rules_raw", {"rules": rules_raw, "unparsed": n_fail, "texts": raw_texts}
    yield "log", f"extraction: {sum(len(v) for v in rules_raw.values())} rules, {n_fail} unparsed, {time.time() - t:.0f}s"

    # 3) filtering (one prompt per environment)
    t = time.time()
    classes = sorted(rules_raw)
    filt_texts = gen(4096)([dg.filter_prompt(dg.CLASS_TO_ENV[c], rules_raw[c]) for c in classes])
    dynamics = {}
    for c, text in zip(classes, filt_texts):
        parsed = dg.parse_json_array(text)
        dynamics[c] = [r for r in parsed if isinstance(r, dict)] if parsed is not None else rules_raw[c]
    yield "dynamics", {"dynamics": dynamics, "texts": filt_texts}
    yield "log", "filtering: " + ", ".join(f"{c} {len(rules_raw[c])}→{len(dynamics[c])}" for c in classes) + f" ({time.time() - t:.0f}s)"

    # 4) evaluation: none vs. DG on the multi-turn tasks
    entries = []
    for split in splits.split(","):
        entries += [(e, gt) for e, gt in dg.eval_entries(split)][: (limit or None)]
    gold = {e["id"]: gt for e, gt in entries}
    for name, dyn in (("none", None), ("dg", dynamics)):
        t = time.time()
        ns = f"eval_{name}"
        eps_eval = dg.eval_episodes(entries, ns, dyn)
        dg.run_episodes(gen(1024), eps_eval)
        rows = []
        for ep in eps_eval:
            sc = dg.score(ep, gold[ep.entry["id"]], ns)
            rows.append({"id": ep.entry["id"], **sc, "n_turns": len(ep.results),
                         "n_steps": sum(len(tu) for tu in ep.results), "results": ep.results,
                         "prompt_chars": len(ep.messages[0]["content"])})
        acc = sum(r["valid"] for r in rows) / len(rows)
        yield f"eval_{name}", rows
        yield "log", f"eval {name}: accuracy {acc:.3f} ({sum(r['valid'] for r in rows)}/{len(rows)}), {time.time() - t:.0f}s"


@app.function(timeout=60 * 60)
def evaluate(model: str, dynamics_json: str, repeats: int, limit: int, splits: str):
    """Noise check: re-run the none / DG evaluations `repeats` times with fixed rules (no exploration).
    Each repeat builds a fresh vLLM engine, as separate runs would."""
    import gc
    import time

    sys.path.insert(0, "/root")
    import torch
    from vllm import LLM, SamplingParams

    from sa import dg

    dynamics = json.loads(dynamics_json)["dynamics"]
    entries = []
    for split in splits.split(","):
        entries += [(e, gt) for e, gt in dg.eval_entries(split)][: (limit or None)]
    gold = {e["id"]: gt for e, gt in entries}
    sp = SamplingParams(temperature=0.0, max_tokens=1024)
    for r in range(1, repeats + 1):
        llm = LLM(model=model, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=0.90,
                  enable_prefix_caching=True)

        def gen(prompts):
            return [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)]

        for name, dyn in (("none", None), ("dg", dynamics)):
            t = time.time()
            ns = f"rep{r}_{name}"
            eps_eval = dg.eval_episodes(entries, ns, dyn)
            dg.run_episodes(gen, eps_eval)
            rows = [{"id": ep.entry["id"], **dg.score(ep, gold[ep.entry["id"]], ns),
                     "n_steps": sum(len(tu) for tu in ep.results), "results": ep.results} for ep in eps_eval]
            yield f"eval_{name}_rep{r}", rows
            yield "log", f"repeat {r} {name}: {sum(x['valid'] for x in rows)}/{len(rows)} ({time.time() - t:.0f}s)"
        del llm
        gc.collect()
        torch.cuda.empty_cache()


@app.local_entrypoint()
def main(model: str, tag: str, gpu: str = "L4", timeout_min: int = 20, limit: int = 0, splits: str = "val,test",
         rules: str = "", repeats: int = 0):
    """With --rules path/to/dynamics.json --repeats N: evaluation only (noise check), saved as eval_<name>_rep<k>.jsonl."""
    out = ROOT / "results" / "dg" / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"GPU={gpu}, hard limit {timeout_min} min, limit={limit or 'all'}")
    if rules:
        fn = evaluate.with_options(gpu=gpu, timeout=timeout_min * 60)
        summary = {"model": model, "gpu": gpu, "rules": rules, "repeats": repeats}
        for kind, payload in fn.remote_gen(model, Path(rules).read_text(), repeats, limit, splits):
            if kind == "log":
                print(payload, flush=True)
                summary.setdefault("log", []).append(payload)
            else:
                with open(out / f"{kind}.jsonl", "w") as f:
                    for row in payload:
                        f.write(json.dumps(row) + "\n")
                summary[kind] = sum(r["valid"] for r in payload) / len(payload)
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"saved to {out}")
        return
    fn = pipeline.with_options(gpu=gpu, timeout=timeout_min * 60)
    summary = {"model": model, "gpu": gpu, "limit": limit, "splits": splits}
    for kind, payload in fn.remote_gen(model, limit, splits):
        if kind == "log":
            print(payload, flush=True)
            summary.setdefault("log", []).append(payload)
        elif kind in ("exploration", "eval_none", "eval_dg"):
            with open(out / f"{kind}.jsonl", "w") as f:
                for row in payload:
                    f.write(json.dumps(row) + "\n")
            if kind.startswith("eval_"):
                summary[kind] = {"n": len(payload), "accuracy": sum(r["valid"] for r in payload) / len(payload)}
        else:
            (out / f"{kind}.json").write_text(json.dumps(payload, indent=1))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved to {out}")
