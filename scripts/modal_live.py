"""GPU backend for the live demo (Hugging Face Space → gateway → GPU worker running sa/live.py).

    modal run scripts/modal_live.py::warm_cache    # one-off: download model weights into a Modal Volume
    modal deploy scripts/modal_live.py             # deploy (costs nothing while idle)
    modal app stop sa-live                          # switch it off completely

Speed:
- memory snapshots: each worker loads its model on CPU once, Modal saves that memory, and later cold
  starts restore it and only copy the weights to the GPU (much faster than importing + loading again)
- the Space calls /warm when a visitor opens the Try page, so the GPU boots while they read
- jobs are submit-and-poll (/submit → call id, /result → result or "running"), never one long request

Cost safety:
- scales to zero after SCALEDOWN_S idle seconds; at most one container per model; hard timeouts
- the only public URL is the gateway, a tiny CPU function that checks the shared token (Modal secret
  "sa-live-token", key SA_LIVE_TOKEN) before any GPU is touched; only the Space knows the token
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
SCALEDOWN_S = 180
# GPU lists are fallbacks in order: if the first type has no free capacity, Modal uses the next one
# (a single type can mean minutes in a queue; the snapshot restore itself takes only a few seconds).
MODELS = {"small": ("Qwen/Qwen2.5-1.5B-Instruct", ["L4", "A10G", "L40S", "A100-40GB"], 16000),
          "large": ("Qwen/Qwen2.5-14B-Instruct", ["L40S", "A100-80GB", "H100"], 12000)}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "transformers==5.20.2", "numpy", "safetensors", "huggingface_hub", "fastapi[standard]")
    .env({"HF_HOME": "/models/hf", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"})
    .add_local_file(ROOT / "sa" / "live.py", "/root/sa_live.py")
)
# The gateway gets its own tiny image so it boots in seconds (the big image with torch is for the GPUs).
gateway_image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]")
vol = modal.Volume.from_name("sa-live-models", create_if_missing=True)
secret = modal.Secret.from_name("sa-live-token")
app = modal.App("sa-live", image=image)


def _run(sa, body):
    try:
        return sa.run(prompt=body.get("prompt", ""), answer=body.get("answer", ""),
                      variant=body.get("variant", "paper"), lr=min(float(body.get("lr", 0.1)), 10.0),
                      steps=max(1, min(int(body.get("steps", 1)), 5)), generate=bool(body.get("generate", False)),
                      max_new=min(int(body.get("max_new", 96)), 160))
    except ValueError as e:
        return {"error": str(e)}


def _load_cpu(key):
    import sys
    import time

    import torch
    sys.path.insert(0, "/root")
    from sa_live import LiveSA
    t = time.time()
    name, _, max_tokens = MODELS[key]
    sa = LiveSA(name, device="cpu", dtype=torch.bfloat16, max_tokens=max_tokens)
    print(f"[{key}] loaded on CPU in {time.time() - t:.1f}s (this state is snapshotted)", flush=True)
    return sa


def _to_gpu(sa, key):
    import time
    t = time.time()
    sa.to("cuda")
    print(f"[{key}] moved to GPU in {time.time() - t:.1f}s", flush=True)


# GPU workers: no public URL; only the gateway can call them.
@app.cls(gpu=MODELS["small"][1], volumes={"/models": vol}, scaledown_window=SCALEDOWN_S, max_containers=1,
         timeout=300, enable_memory_snapshot=True, memory=16384)
class Small:
    @modal.enter(snap=True)
    def load_cpu(self):
        self.sa = _load_cpu("small")

    @modal.enter(snap=False)
    def to_gpu(self):
        _to_gpu(self.sa, "small")

    @modal.method()
    def ping(self):
        return {"ready": True, "model": MODELS["small"][0]}

    @modal.method()
    def run(self, body: dict):
        return _run(self.sa, body)


@app.cls(gpu=MODELS["large"][1], volumes={"/models": vol}, scaledown_window=SCALEDOWN_S, max_containers=1,
         timeout=420, enable_memory_snapshot=True, memory=49152)
class Large:
    @modal.enter(snap=True)
    def load_cpu(self):
        self.sa = _load_cpu("large")

    @modal.enter(snap=False)
    def to_gpu(self):
        _to_gpu(self.sa, "large")

    @modal.method()
    def ping(self):
        return {"ready": True, "model": MODELS["large"][0]}

    @modal.method()
    def run(self, body: dict):
        return _run(self.sa, body)


WORKERS = {"small": Small, "large": Large}


# Public entry point: a tiny CPU web app that checks the token BEFORE any GPU is touched.
@app.function(image=gateway_image, cpu=0.5, memory=512, secrets=[secret], scaledown_window=300, max_containers=2,
              timeout=120)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def gateway():
    import hmac
    import os

    from fastapi import FastAPI, HTTPException

    web = FastAPI(docs_url=None, redoc_url=None)

    def auth(body):
        token = str(body.pop("token", "") or "")
        if not token or not hmac.compare_digest(token, os.environ["SA_LIVE_TOKEN"]):
            raise HTTPException(status_code=401, detail="unauthorized")

    def worker(body):
        model = body.pop("model", "small")
        if model not in WORKERS:
            raise HTTPException(status_code=400, detail="unknown model")
        return WORKERS[model]()

    @web.post("/warm")
    def warm(body: dict):
        auth(body)
        return {"call_id": worker(body).ping.spawn().object_id}

    @web.post("/submit")
    def submit(body: dict):
        auth(body)
        return {"call_id": worker(body).run.spawn(body).object_id}

    @web.post("/result")
    def result(body: dict):
        auth(body)
        call_id = str(body.get("call_id", ""))
        if not call_id.startswith("fc-"):
            raise HTTPException(status_code=400, detail="bad call id")
        try:
            out = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            return {"status": "running"}
        except modal.exception.OutputExpiredError:
            raise HTTPException(status_code=410, detail="result expired")
        except Exception as e:  # noqa: BLE001  (worker crashed, cancelled, …)
            return {"status": "error", "detail": f"the GPU job failed ({type(e).__name__})"}
        if isinstance(out, dict) and "error" in out:
            return {"status": "error", "detail": out["error"]}
        return {"status": "done", "result": out}

    return web


@app.function(volumes={"/models": vol}, timeout=30 * 60)
def warm_cache():
    """Download both models into the Volume once, so cold starts read from disk instead of the Hub."""
    import os
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download
    for name, _, _ in MODELS.values():
        snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
        print("cached", name, flush=True)
    vol.commit()
