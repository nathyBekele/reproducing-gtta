"""Run every condition on every example and save per-example JSON (resumable).

Examples:
    python scripts/run_all.py --engine mlx   --model Qwen/Qwen2.5-1.5B-Instruct --tag qwen2.5-1.5b
    python scripts/run_all.py --engine torch --model Qwen/Qwen2.5-14B-Instruct  --tag qwen2.5-14b-bf16

Output: results/raw/<tag>/<example id>.json and results/raw/<tag>/config.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sa import alignment, data, rivals  # noqa: E402
from sa.records import VARIANTS  # noqa: E402


def make_engine(engine, model):
    if engine == "mlx":
        from sa.engine_mlx import MLXEngine
        return MLXEngine(model)
    if engine == "torch":
        from sa.engine_torch import TorchEngine
        return TorchEngine(model)
    raise ValueError(engine)


def run_config(eng, tag, k):
    return {
        **eng.info(), "tag": tag, "variants": VARIANTS, "adam": alignment.ADAM,
        "lr": alignment.DEFAULT_LR, "steps": alignment.DEFAULT_STEPS,
        "alpha_grid": rivals.ALPHA_GRID, "random_seeds": rivals.RANDOM_SEEDS, "k": k,
        "data_commit": (data.DATA_DIR / "SOURCE_COMMIT").read_text().strip(),
    }


def select_examples(splits, limit=None):
    examples = []
    for split in splits.split(","):
        examples += data.load_split(split)[:limit]
    return examples


def write_record(out_dir, rec):
    path = Path(out_dir) / f"{rec['id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(path)


def progress_line(i, n, rec):
    c = rec["conditions"]
    return (f"[{i}/{n}] {rec['id']} T={rec['n_prompt_tokens']} {rec['seconds']}s | none {c['none']['mean']:.3f} | "
            f"paper {c['sa_paper']['mean']:.3f} | exact {c['sa_exact']['mean']:.3f} | copy@2 {c['copy_alpha_2.0']['mean']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--limit", type=int, default=None, help="max examples per split (for quick runs)")
    ap.add_argument("--k", type=int, default=30, help="top-k boosted tokens to store")
    args = ap.parse_args()

    out_dir = ROOT / "results" / "raw" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    eng = make_engine(args.engine, args.model)
    (out_dir / "config.json").write_text(json.dumps(run_config(eng, args.tag, args.k), indent=2))

    examples = select_examples(args.splits, args.limit)
    done = 0
    for i, ex in enumerate(examples, 1):
        if (out_dir / f"{ex.id}.json").exists():
            continue
        rec = eng.run_example(ex, args.k)
        write_record(out_dir, rec)
        done += 1
        print(progress_line(i, len(examples), rec), flush=True)
    print(f"done: {done} new examples in {out_dir}")


if __name__ == "__main__":
    main()
