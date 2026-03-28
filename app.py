"""
app.py — Entry point for HuggingFace Spaces
Starts FastAPI backend (port 8000) as a subprocess,
then launches Gradio frontend (port 7860) in the main thread.
HF Spaces exposes port 7860 by default.

"""

import atexit
import os
import signal
import subprocess
import sys
import time
import urllib.request

# Fix gradio_client bug: get_type() crashes when schema is bool (additionalProperties: false)
try:
    from gradio_client import utils as _gc_utils
    _orig_get_type = _gc_utils.get_type
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "any"
        return _orig_get_type(schema)
    _gc_utils.get_type = _safe_get_type
except Exception:
    pass

from frontend import build_ui

BACKEND_PORT  = 8000
FRONTEND_PORT = 7860


def free_port(port: int):
    """Kill any process holding the given port."""
    try:
        result = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True).strip()
        for pid in result.splitlines():
            os.kill(int(pid), signal.SIGKILL)
    except Exception:
        pass


def wait_for_backend(port: int, timeout: int = 30):
    """Poll until the backend is accepting connections."""
    url = f"http://localhost:{port}/system/info"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            time.sleep(0.3)
    print("⚠️  Backend did not start in time — status bar may show Unknown")


if __name__ == "__main__":
    os.system("clear")
    free_port(BACKEND_PORT)
    free_port(FRONTEND_PORT)

    # Start backend as a subprocess (avoids asyncio event-loop conflict with Gradio)
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend:app",
         "--host", "0.0.0.0", "--port", str(BACKEND_PORT), "--log-level", "warning"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    atexit.register(backend_proc.terminate)

    wait_for_backend(BACKEND_PORT)
    print(f"✅ Backend running on port {BACKEND_PORT}")

    # Start Gradio in main thread (blocks)
    in_hf = bool(os.getenv("SPACE_ID"))          # HF Spaces sets SPACE_ID
    ui = build_ui()
    ui.queue().launch(server_name="0.0.0.0", server_port=FRONTEND_PORT, show_error=True, inbrowser=not in_hf)
