"""Entry point for the Hugging Face Space (Gradio SDK, ZeroGPU hardware), same pattern as NathyB/ir-studio.

The Space runs `python app.py`. We mount a minimal Gradio page at /gradio (the Gradio SDK expects one) on our
FastAPI server (server.py: the pages at / and /try, plus the API), then serve everything with uvicorn on 7860.
The live demo's heavy work runs on Modal GPUs (SA_BACKEND=modal) or this Space's CPU (SA_BACKEND=local).
"""
import os
import socket
import sys

os.environ["GRADIO_SSR_MODE"] = "false"
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

# ZeroGPU hardware checks at startup that the app has a @spaces.GPU function and reports readiness.
# We don't use the ZeroGPU GPU itself (Modal does the GPU work), so this probe is never called.
try:
    import spaces
except ImportError:
    spaces = None

if spaces is not None:
    @spaces.GPU
    def _zerogpu_probe(*args, **kwargs):
        """Never called. Exists only to satisfy ZeroGPU's startup detection."""
        return None


def _zerogpu_startup_report():
    """ZeroGPU expects a startup report before the server binds; we serve with uvicorn directly, so send it."""
    if spaces is None:
        return
    try:
        from spaces.zero import startup as zero_startup
    except ImportError:
        return
    try:
        zero_startup()
        print("ZeroGPU startup report sent")
    except Exception as exc:  # noqa: BLE001
        print(f"ZeroGPU startup report skipped: {exc!r}")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402

from server import app as fastapi_app  # noqa: E402

with gr.Blocks(title="GTTA SA Explorer") as demo:
    gr.Markdown("# GTTA Syntactic Alignment Explorer\nThe site is served at the root path `/` (results) and `/try` (live demo).")
    if spaces is not None:
        _probe_btn = gr.Button("zerogpu_probe", visible=False)
        _probe_btn.click(fn=_zerogpu_probe)

try:
    app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio", ssr_mode=False)
except TypeError:
    app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

# server.py serves the static pages with a catch-all mount at "/"; keep it last so /gradio stays reachable.
_routes = app.router.routes
_static = [r for r in _routes if getattr(r, "name", None) == "static"]
app.router.routes = [r for r in _routes if r not in _static] + _static


def _port_in_use(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        return probe.connect_ex((host, port)) == 0


def _choose_port():
    candidates = [7860] if os.getenv("SPACE_ID") else []
    port_env = os.getenv("PORT")
    if port_env and port_env.isdigit():
        candidates.append(int(port_env))
    candidates.append(7860)
    for c in candidates:
        if not _port_in_use(c):
            return c
    return candidates[0]


if __name__ == "__main__":
    host = os.getenv("HOST") or "0.0.0.0"
    port = _choose_port()
    print(f"[GTTA SA Explorer] serving on {host}:{port}")
    _zerogpu_startup_report()
    uvicorn.run(app, host=host, port=port, log_level="info")
