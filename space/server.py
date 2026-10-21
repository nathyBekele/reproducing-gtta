"""FastAPI server for the Space (pages + API). app.py wraps it for the Gradio SDK.

Environment variables (Space → Settings → Variables and secrets):
  SA_BACKEND        "modal" (default): runs go to the Modal GPU gateway
                    "local": runs execute on this Space's own CPU (free, slow; small model only)
  MODAL_URL         the Modal gateway URL (secret)
  MODAL_MODELS      which GPU models to offer: "small,large" (default), or "small" to hide the 14B option
  SA_LIVE_TOKEN     shared token the Modal gateway checks (secret)
  LOCAL_MODEL       model for SA_BACKEND=local (default Qwen/Qwen2.5-0.5B-Instruct)
  CLARITY_ID        Microsoft Clarity project ID (empty = no Clarity)
  FORMSUBMIT_URL    FormSubmit AJAX endpoint for visitor emails (empty = no emails)
  RUNS_PER_HOUR     per-visitor limit (default 20)
  RUNS_PER_DAY      total limit across all visitors (default 300)

Flow: the page calls /api/warm when it opens (the GPU boots while the visitor reads), /api/run submits a
job and returns its id, and the page polls /api/result/<id> every couple of seconds.
"""
import json
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
TASKS = {t["id"]: t for t in json.loads((HERE / "data" / "tasks.json").read_text())}

BACKEND = os.environ.get("SA_BACKEND", "modal").strip().lower()
MODAL_URL = os.environ.get("MODAL_URL", "").strip().rstrip("/")
MODAL_MODELS = [m.strip() for m in os.environ.get("MODAL_MODELS", "small,large").split(",") if m.strip()]
TOKEN = os.environ.get("SA_LIVE_TOKEN", "")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
RUNS_PER_HOUR = int(os.environ.get("RUNS_PER_HOUR", "20"))
RUNS_PER_DAY = int(os.environ.get("RUNS_PER_DAY", "300"))
WARM_EVERY_S = 60          # at most one wake-up call per model per minute, whatever the traffic
MODEL_LABELS = {"small": "Qwen2.5-1.5B-Instruct (GPU)", "large": "Qwen2.5-14B-Instruct (GPU, the paper's model)"}

app = FastAPI(docs_url=None, redoc_url=None)

# ---------------- limits ----------------
_hits = defaultdict(deque)
_day = {"date": "", "n": 0}
_lock = threading.Lock()


def _client_ip(req: Request):
    fwd = req.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?")


def _allow(ip):
    now = time.time()
    with _lock:
        today = time.strftime("%Y-%m-%d", time.gmtime(now))
        if _day["date"] != today:
            _day.update(date=today, n=0)
        if _day["n"] >= RUNS_PER_DAY:
            return "The demo has reached today's run limit. Please try again tomorrow."
        q = _hits[ip]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= RUNS_PER_HOUR:
            return f"Limit of {RUNS_PER_HOUR} runs per hour reached. Please wait a bit."
        q.append(now)
        _day["n"] += 1
    return None


# ---------------- local backend (jobs run in a background thread) ----------------
_local = {"sa": None}
_local_lock = threading.Lock()
_jobs = {}


def _local_sa():
    from sa_live import LiveSA
    with _local_lock:
        if _local["sa"] is None:
            _local["sa"] = LiveSA(LOCAL_MODEL, device="cpu", max_tokens=9000)
        return _local["sa"]


def _local_job(job, args):
    try:
        sa = _local_sa()
        with _local_lock:
            out = sa.run(**args)
        _jobs[job] = {"status": "done", "result": out, "t": time.time()}
    except ValueError as e:
        _jobs[job] = {"status": "error", "detail": str(e), "t": time.time()}
    except Exception as e:  # noqa: BLE001
        _jobs[job] = {"status": "error", "detail": f"run failed ({type(e).__name__})", "t": time.time()}


def _local_submit(args):
    job = "local-" + uuid.uuid4().hex
    for k in [k for k, v in list(_jobs.items()) if time.time() - v["t"] > 3600 and k != "local-warm"]:
        _jobs.pop(k, None)
    _jobs[job] = {"status": "running", "t": time.time()}
    threading.Thread(target=_local_job, args=(job, args), daemon=True).start()
    return job


# ---------------- Modal backend ----------------
_last_warm = {}


async def _gateway(path, payload, attempts=3):
    """POST to the Modal gateway; retries briefly if it is still booting or hiccups (5xx / connection error)."""
    import asyncio
    last = "The GPU backend didn't respond. Please try again in a minute."
    for i in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=15.0), follow_redirects=True) as client:
                r = await client.post(f"{MODAL_URL}/{path}", json={**payload, "token": TOKEN})
        except httpx.HTTPError:
            await asyncio.sleep(2)
            continue
        if r.status_code == 200:
            return r.json()
        if r.headers.get("content-type", "").startswith("application/json"):
            last = r.json().get("detail", last)
        if r.status_code < 500:
            raise HTTPException(r.status_code, last)
        await asyncio.sleep(2)
    raise HTTPException(502, last)


# ---------------- API ----------------
def _models():
    if BACKEND == "local":
        return [{"id": "local", "label": f"{LOCAL_MODEL.split('/')[-1]} (free CPU, slower)"}]
    if not (MODAL_URL and TOKEN):
        return []
    return [{"id": k, "label": MODEL_LABELS[k]} for k in MODAL_MODELS if k in MODEL_LABELS]


@app.get("/config.js")
def config_js():
    cfg = {"clarityId": os.environ.get("CLARITY_ID", ""), "formsubmitUrl": os.environ.get("FORMSUBMIT_URL", ""),
           "backend": BACKEND, "models": _models()}
    return Response("window.SITE_CONFIG = " + json.dumps(cfg) + ";\n", media_type="application/javascript",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/tasks")
def tasks():
    return [{"id": t["id"], "split": t["split"], "classes": t["classes"], "request": t["request"]} for t in TASKS.values()]


@app.get("/api/task/{task_id}")
def task(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(404, "unknown task")
    return TASKS[task_id]


@app.post("/api/warm")
async def warm(req: Request):
    """Start the model in the background. Returns a job id the page can poll to show 'ready'."""
    body = await req.json()
    if BACKEND == "local":
        job = "local-warm"
        if job not in _jobs:
            _jobs[job] = {"status": "running", "t": time.time()}

            def load():
                try:
                    _local_sa()
                    _jobs[job] = {"status": "done", "result": {"ready": True}, "t": time.time()}
                except Exception as e:  # noqa: BLE001
                    _jobs[job] = {"status": "error", "detail": f"model failed to load ({type(e).__name__})", "t": time.time()}
            threading.Thread(target=load, daemon=True).start()
        return {"job": job}
    model = body.get("model", "small")
    if model not in MODAL_MODELS or not MODAL_URL:
        raise HTTPException(400, "unknown model")
    now = time.time()
    last = _last_warm.get(model)
    if last and now - last["t"] < WARM_EVERY_S:
        return {"job": last["job"]}
    out = await _gateway("warm", {"model": model})
    _last_warm[model] = {"t": now, "job": out["call_id"]}
    return {"job": out["call_id"]}


@app.post("/api/run")
async def run(req: Request):
    body = await req.json()
    prompt, answer = str(body.get("prompt", "")), str(body.get("answer", ""))
    if len(prompt) > 60000 or len(answer) > 4000:
        raise HTTPException(400, "Input too long for this demo (prompt ≤ 60k characters, answer ≤ 4k).")
    msg = _allow(_client_ip(req))
    if msg:
        return JSONResponse({"detail": msg}, status_code=429)
    args = {"prompt": prompt, "answer": answer, "variant": body.get("variant", "paper"),
            "lr": min(max(float(body.get("lr", 0.1)), 1e-4), 10.0), "steps": max(1, min(int(body.get("steps", 1)), 5)),
            "generate": bool(body.get("generate", False)), "max_new": 96}
    if BACKEND == "local":
        return {"job": _local_submit(args)}
    model = body.get("model", "small")
    if not MODAL_URL or model not in MODAL_MODELS:
        raise HTTPException(400, "That model isn't available right now.")
    out = await _gateway("submit", {**args, "model": model})
    return {"job": out["call_id"]}


@app.get("/api/result/{job}")
async def result(job: str):
    if job.startswith("local-"):
        j = _jobs.get(job)
        if not j:
            raise HTTPException(404, "unknown job")
        return {k: v for k, v in j.items() if k != "t"}
    if BACKEND == "local" or not job.startswith("fc-"):
        raise HTTPException(404, "unknown job")
    return await _gateway("result", {"call_id": job})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/try")
def try_page():
    return FileResponse(STATIC / "try.html")


@app.get("/dg")
def dg_page():
    return FileResponse(STATIC / "dg.html")


app.mount("/", StaticFiles(directory=STATIC), name="static")
