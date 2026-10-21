"""Run the experiment on a Modal GPU. Results stream back and are saved locally one by one.

One or more SA settings can share one container (the model loads once):
    --settings "0.1:1"            lr 0.1, 1 step (the paper's WebArena setting) → tag as given
    --settings "1.0:1,1.0:3"      several settings → tags get a suffix "-lr1.0-s1", "-lr1.0-s3"

Examples:
    modal run scripts/modal_run.py --model Qwen/Qwen2.5-1.5B-Instruct --tag qwen2.5-1.5b-gpu-test --gpu L4 --limit 2 --timeout-min 15
    modal run scripts/modal_run.py --model Qwen/Qwen2.5-14B-Instruct --tag qwen2.5-14b-bf16 --gpu H100 --timeout-min 60
    modal run scripts/modal_run.py --model Qwen/Qwen2.5-14B-Instruct --tag qwen2.5-14b-bf16 --settings "1.0:1,1.0:3" --gpu H100 --timeout-min 60

Safety: one container, a hard timeout, and the app stops when the run ends (nothing stays running).
Already-saved examples are skipped, so a stopped run can be resumed with the same command.
"""
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "transformers==5.20.2", "numpy", "safetensors", "huggingface_hub", "mpmath")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "sa", "/root/sa")
    .add_local_dir(ROOT / "data" / "bfcl_v3", "/root/data/bfcl_v3")
    .add_local_dir(ROOT / "third_party", "/root/third_party")
)

app = modal.App("sa-what-it-learns", image=image)


def parse_settings(spec):
    out = []
    for item in spec.split(","):
        lr, steps = item.split(":")
        out.append((float(lr), int(steps)))
    return out


def tag_for(base, lr, steps, n_settings):
    return base if n_settings == 1 else f"{base}-lr{lr}-s{steps}"


@app.function(timeout=60 * 60)
def run(model: str, settings_spec: str, base_tag: str, splits: str, limit: int, skip: dict, k: int,
        dynamics_json: str = ""):
    """Generator: yields ("config", tag, dict) per setting, then ("record", tag, dict) per example."""
    import time

    sys.path.insert(0, "/root")
    from sa import data, settings
    from sa.engine_torch import TorchEngine
    from sa.records import VARIANTS

    t0 = time.time()
    eng = TorchEngine(model)
    load_s = round(time.time() - t0, 1)
    dynamics = json.loads(dynamics_json)["dynamics"] if dynamics_json else None
    examples = []
    for split in splits.split(","):
        examples += data.load_split(split, dynamics=dynamics)[: (limit or None)]
    todo = parse_settings(settings_spec)
    for lr, steps in todo:
        tag = tag_for(base_tag, lr, steps, len(todo))
        eng.lr, eng.steps = lr, steps
        yield "config", tag, {
            **eng.info(), "variants": VARIANTS, "adam": settings.ADAM, "alpha_grid": settings.ALPHA_GRID,
            "random_seeds": settings.RANDOM_SEEDS, "k": k,
            "data_commit": (data.DATA_DIR / "SOURCE_COMMIT").read_text().strip(), "model_load_seconds": load_s,
            "dynamics": bool(dynamics),
        }
        done = set(skip.get(tag, []))
        for ex in examples:
            if ex.id not in done:
                yield "record", tag, eng.run_example(ex, k)


@app.local_entrypoint()
def main(model: str, tag: str, settings: str = "0.1:1", gpu: str = "L4", timeout_min: int = 15,
         splits: str = "val,test", limit: int = 0, k: int = 30, dynamics: str = ""):
    sys.path.insert(0, str(ROOT))
    from scripts.run_all import progress_line, write_record

    todo = parse_settings(settings)
    skip = {}
    for lr, steps in todo:
        t = tag_for(tag, lr, steps, len(todo))
        d = ROOT / "results" / "raw" / t
        d.mkdir(parents=True, exist_ok=True)
        skip[t] = [p.stem for p in d.glob("multi_turn_base_*.json")]
        print(f"{t}: {len(skip[t])} examples already saved (skipped)")
    print(f"GPU={gpu}, hard limit {timeout_min} min")

    dynamics_json = Path(dynamics).read_text() if dynamics else ""
    if dynamics:
        print(f"using DG rules from {dynamics}")
    fn = run.with_options(gpu=gpu, timeout=timeout_min * 60)
    n = 0
    for kind, t, payload in fn.remote_gen(model, settings, tag, splits, limit, skip, k, dynamics_json):
        out_dir = ROOT / "results" / "raw" / t
        if kind == "config":
            (out_dir / "config.json").write_text(json.dumps({**payload, "tag": t, "gpu": gpu}, indent=2))
            print(f"[{t}] lr={payload['lr']} steps={payload['steps']} | model loaded in "
                  f"{payload['model_load_seconds']}s on {payload['device']} ({payload['dtype']})", flush=True)
        else:
            write_record(out_dir, payload)
            n += 1
            print(f"[{t}] " + progress_line(n, "?", payload), flush=True)
    print(f"done: {n} new examples saved")
