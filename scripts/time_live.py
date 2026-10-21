"""Time the live GPU backend: wake-up (cold start) and one SA run on a BFCL task.

    SA_LIVE_URL=https://<you>--sa-live-gateway.modal.run python scripts/time_live.py --model small
    (reads the token from space/.secrets.local)
"""
import argparse
import json
import os
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
URL = os.environ.get("SA_LIVE_URL", "")   # the gateway URL printed by `modal deploy scripts/modal_live.py`


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--run", action="store_true", help="also time one SA run on task #0")
    a = ap.parse_args()
    if not URL:
        raise SystemExit("set SA_LIVE_URL to your gateway URL")
    token = dict(l.split("=", 1) for l in (ROOT / "space/.secrets.local").read_text().split())["SA_LIVE_TOKEN"]
    c = httpx.Client(timeout=60, follow_redirects=True)

    def post(path, body):
        r = c.post(f"{URL}/{path}", json={**body, "token": token})
        r.raise_for_status()
        return r.json()

    def wait(call_id):
        while True:
            j = post("result", {"call_id": call_id})
            if j["status"] != "running":
                return j
            time.sleep(1)

    t = time.time()
    j = wait(post("warm", {"model": a.model})["call_id"])
    print(f"wake ({a.model}): {time.time() - t:.1f}s → {j['status']}", flush=True)
    if a.run:
        task = json.loads((ROOT / "space/data/tasks.json").read_text())[0]
        for i in range(2):
            t = time.time()
            j = wait(post("submit", {"model": a.model, "prompt": task["prompt"], "answer": task["answer"],
                                     "variant": "paper", "lr": 0.1, "steps": 1, "generate": i == 1})["call_id"])
            r = j.get("result", {})
            print(f"run {i + 1} ({'with' if i else 'no'} generation): {time.time() - t:.1f}s → {j['status']}, "
                  f"‖δ‖={r.get('delta_norm')}, server {r.get('seconds')}", flush=True)


if __name__ == "__main__":
    main()
