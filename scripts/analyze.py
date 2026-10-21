"""Summarize a run: python scripts/analyze.py --tag qwen2.5-1.5b

Writes results/summary/<tag>/summary.json and summary.md.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sa import analysis  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    s = analysis.summarize(ROOT / "results" / "raw" / args.tag)
    out = ROOT / "results" / "summary" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    md = analysis.to_markdown(s)
    (out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
