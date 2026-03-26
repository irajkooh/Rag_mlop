"""
frontend.py — Gradio UI
────────────────────────────────────────────────────────────────────
Status bar (top, blue): Environment | Device | LLM — pulled live
                        from backend /system/info at startup.

Two tabs:
  💬 Chat    (default) — memory, per-answer copy, Read/Stop TTS, Copy All, Clear
  📂 Upload             — drag-and-drop PDFs, KB status panel

Vector store : ChromaDB  (via backend)
LLM          : Ollama primary · Groq fallback  (via backend)
Design       : deep navy + electric blue + teal accent, Space Mono / DM Sans
"""

import asyncio
import gradio as gr
import requests
import uuid
import os
import re

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# ══════════════════════════════════════════════════════════════════════════════
# Backend helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_system_info() -> dict:
    """Fetch /system/info from backend — drives the blue status bar."""
    try:
        r = requests.get(f"{BACKEND_URL}/system/info", timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {
            "environment":   "Unknown",
            "device":        "Unknown",
            "llm":           "Unknown",
            "chroma_chunks": 0,
        }


def ask_backend(question: str, session_id: str) -> dict:
    try:
        r = requests.post(
            f"{BACKEND_URL}/ask",
            json={"question": question, "session_id": session_id},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"answer": f"⚠️ Backend error: {e}", "sources": [], "latency_ms": 0}


def clear_backend_session(session_id: str):
    try:
        requests.post(
            f"{BACKEND_URL}/session/clear",
            json={"session_id": session_id},
            timeout=10,
        )
    except Exception:
        pass


def upload_to_backend(file_path: str, filename: str | None = None) -> str:
    try:
        name = filename or os.path.basename(file_path)
        with open(file_path, "rb") as f:
            r = requests.post(
                f"{BACKEND_URL}/upload",
                files={"file": (name, f, "application/pdf")},
                timeout=300,
            )
        r.raise_for_status()
        d = r.json()
        return f"✅ {d['message']}  —  {d['chunks_added']} chunks added  |  total: {d['total_chunks']}"
    except Exception as e:
        return f"❌ Upload failed: {e}"


def upload_url_to_backend(url: str) -> str:
    url = url.strip()
    if not url:
        return "⚠️  Please enter a URL."
    try:
        r = requests.post(
            f"{BACKEND_URL}/upload_url",
            json={"url": url},
            timeout=60,
        )
        r.raise_for_status()
        d = r.json()
        return f"✅ {d['message']}  —  {d['chunks_added']} chunks added  |  total: {d['total_chunks']}"
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return f"❌ {detail}"
    except Exception as e:
        return f"❌ URL indexing failed: {e}"


def get_file_list() -> list:
    try:
        r = requests.get(f"{BACKEND_URL}/kb/status", timeout=10)
        return list(r.json().get("files", {}).keys())
    except Exception:
        return []


def file_selector_update():
    files = get_file_list()
    return gr.update(choices=files, value=[])


def btn_state(files: list):
    enabled = bool(files)
    return gr.update(interactive=enabled), gr.update(interactive=enabled)


def delete_selected(selected: list):
    if not selected:
        return "⚠️  No files selected.", gr.update(), get_kb_status()
    results = []
    for fname in selected:
        try:
            r = requests.delete(f"{BACKEND_URL}/files/{fname}", timeout=30)
            r.raise_for_status()
            d = r.json()
            results.append(f"✅ Removed {fname}  ({d['chunks_removed']} chunks)")
        except Exception as e:
            results.append(f"❌ {fname}: {e}")
    files = get_file_list()
    return "\n".join(results), gr.update(choices=files, value=[]), get_kb_status()


def delete_all():
    try:
        r = requests.delete(f"{BACKEND_URL}/files", timeout=30)
        r.raise_for_status()
        return "✅ All files removed from index.", gr.update(choices=[], value=[]), get_kb_status()
    except Exception as e:
        files = get_file_list()
        return f"❌ {e}", gr.update(choices=files, value=[]), get_kb_status()


def get_kb_status() -> str:
    try:
        r = requests.get(f"{BACKEND_URL}/kb/status", timeout=10)
        d = r.json()
        if d["total_chunks"] == 0:
            return "📂  No documents indexed yet.\n    Upload a PDF to get started."
        lines = [f"📚  {d['total_chunks']} total chunks across {len(d['files'])} file(s)\n"]
        for fname, count in sorted(d["files"].items()):
            lines.append(f"  • {fname}  ({count} chunks)")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️  Cannot reach backend: {e}"


def build_status_bar_html() -> str:
    """
    Returns the HTML for the blue status bar.
    Called once at startup; the bar is static for the session lifetime.
    Users can click 🔄 to refresh it.
    """
    info = get_system_info()

    env     = info.get("environment",   "Unknown")
    device  = info.get("device",        "Unknown")
    llm     = info.get("llm",           "Unknown")
    chunks  = info.get("chroma_chunks", 0)

    # Pick icon for device
    dev_icon = "🍎" if "mps" in device.lower() else ("🎮" if "cuda" in device.lower() else "💻")

    return f"""
    <div class="status-bar">
        <span class="status-pill env">
            <span class="status-dot"></span>
            {'🌐 HuggingFace Space' if 'hugging' in env.lower() else '🏠 Local'}
        </span>
        <span class="status-sep">·</span>
        <span class="status-pill device">
            {dev_icon} Device: <strong>{device.upper()}</strong>
        </span>
        <span class="status-sep">·</span>
        <span class="status-pill llm">
            🤖 LLM: <strong>{llm}</strong>
        </span>
        <span class="status-sep">·</span>
        <span class="status-pill chunks">
            🗄️ ChromaDB: <strong>{chunks}</strong> chunks
        </span>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# Chat logic
# ══════════════════════════════════════════════════════════════════════════════

def _inline_copy_btn(plain_text: str) -> str:
    """Tiny inline HTML copy button injected into each assistant bubble."""
    escaped = (plain_text
               .replace("\\", "\\\\")
               .replace("`", "\\`")
               .replace("\n", "\\n")
               .replace("'", "\\'"))
    return (
        '<span style="float:right;font-size:10px;color:#4fc3f7;cursor:pointer;'
        'font-family:monospace;padding:2px 6px;border:1px solid #1565c0;'
        'border-radius:4px;margin-left:8px;user-select:none;" '
        f'onclick="navigator.clipboard.writeText(`{escaped}`).then('
        "()=>{this.textContent='✓';setTimeout(()=>this.textContent='⧉ copy',1400)})"
        '">⧉ copy</span>'
    )


def chat(user_message: str, history: list, session_id: str):
    if not user_message.strip():
        return history, session_id, "", ""

    result   = ask_backend(user_message, session_id)
    answer   = result.get("answer", "I Don't Know")
    sources  = result.get("sources", [])
    latency  = result.get("latency_ms", 0)

    # Strip trailing empty "Key Points:" heading the LLM sometimes emits on IDK
    import re as _re
    answer = _re.sub(r'\n+\*{0,2}Key Points:\*{0,2}\s*$', '', answer).rstrip()

    src_md   = ("\n\n📎 *Sources: " + ", ".join(sources) + "*") if sources else ""
    lat_md   = f"\n⏱️ *{latency:.0f} ms*"

    content = answer + src_md + lat_md

    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": content})
    return history, session_id, "", answer


def clear_chat(session_id: str):
    clear_backend_session(session_id)
    return [], str(uuid.uuid4())


def copy_all_text(history: list) -> str:
    if not history:
        return "The conversation is empty."
    lines = []
    for m in history:
        role    = "You" if m["role"] == "user" else "Assistant"
        content = re.sub(r"<[^>]+>", "", m["content"])   # strip HTML tags
        lines.append(f"{role}:\n{content.strip()}")
    sep = "\n\n" + "─" * 44 + "\n\n"
    return "\n\n" + sep.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Space+Mono:wght@400;700&display=swap');

/* ── Tokens ─────────────────────────────────────────────────────────── */
:root {
    --bg:         #060c18;
    --surface:    #0d1526;
    --surface2:   #162036;
    --surface3:   #1e2d48;
    --blue:       #1e88e5;
    --blue-light: #4fc3f7;
    --blue-glow:  rgba(30,136,229,0.18);
    --blue-dark:  #1565c0;
    --teal:       #00bcd4;
    --teal-dim:   #00838f;
    --amber:      #ffb300;
    --amber-dim:  #e65100;
    --red:        #ef5350;
    --green:      #66bb6a;
    --text:       #cfd8e8;
    --muted:      #6b7d96;
    --user-bg:    #0a1f3a;
    --bot-bg:     #071624;
    --radius:     12px;
    --radius-sm:  7px;
}

/* ── Base ───────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Blue Status Bar ─────────────────────────────────────────────────── */
.status-bar {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
    background: linear-gradient(90deg, #0a1a3a 0%, #0d2060 50%, #0a1a3a 100%);
    border: 1px solid var(--blue-dark);
    border-radius: var(--radius);
    padding: 10px 18px;
    margin-bottom: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    color: var(--blue-light);
    white-space: nowrap;
}
.status-pill strong { color: #fff; }
.status-dot {
    width: 7px; height: 7px;
    background: var(--green);
    border-radius: 50%;
    box-shadow: 0 0 5px var(--green);
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity:1; } 50% { opacity:.4; }
}
.status-sep { color: var(--blue-dark); font-size: 16px; }

/* ── Header ──────────────────────────────────────────────────────────── */
.rag-header {
    text-align: center;
    padding: 20px 0 8px;
}
.rag-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    background: linear-gradient(110deg, var(--blue-light) 0%, var(--teal) 55%, var(--amber) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 5px;
    letter-spacing: -0.5px;
}
.rag-header p {
    color: var(--muted);
    font-size: 12px;
    margin: 0 0 8px;
    font-style: italic;
}
.rag-header .badge {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--blue-dark);
    color: var(--blue-light);
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    padding: 3px 12px;
    border-radius: 20px;
    letter-spacing: 1.2px;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
.tab-nav { border-bottom: 1px solid var(--surface2) !important; }
.tab-nav button {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 12px 24px !important;
    transition: all 0.2s !important;
    letter-spacing: 0.5px;
}
.tab-nav button.selected,
.tab-nav button:hover {
    color: var(--blue-light) !important;
    border-bottom-color: var(--blue) !important;
}

/* ── Chatbot ─────────────────────────────────────────────────────────── */
.chatbot {
    background: var(--surface) !important;
    border: 1px solid var(--surface2) !important;
    border-radius: var(--radius) !important;
}
.message-bubble-border.user {
    background: var(--user-bg) !important;
    border: 1px solid rgba(30,136,229,0.25) !important;
    border-radius: var(--radius) !important;
}
.message-bubble-border.bot {
    background: var(--bot-bg) !important;
    border: 1px solid rgba(30,136,229,0.10) !important;
    border-radius: var(--radius) !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────── */
.gr-textbox textarea, textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--surface3) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px var(--blue-glow) !important;
    outline: none !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
button {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.3px !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.18s ease !important;
    cursor: pointer !important;
}

/* Send — blue gradient */
button.primary, .btn-send button {
    background: linear-gradient(135deg, var(--blue), var(--blue-dark)) !important;
    color: #fff !important;
    border: none !important;
    padding: 10px 18px !important;
    height: 40px !important;
    min-height: 40px !important;
    width: 100% !important;
}
button.primary:hover, .btn-send button:hover {
    filter: brightness(1.15) !important;
    box-shadow: 0 4px 16px var(--blue-glow) !important;
}

/* Read Last — same size as Send */
.btn-read button {
    height: 40px !important;
    min-height: 40px !important;
    width: 100% !important;
}

/* Action — surface */
button.secondary, .btn-action button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--surface3) !important;
    padding: 9px 14px !important;
}
button.secondary:hover, .btn-action button:hover {
    border-color: var(--blue) !important;
    color: var(--blue-light) !important;
}

/* Read — amber */
.btn-read button {
    background: linear-gradient(135deg, var(--amber), var(--amber-dim)) !important;
    color: #111 !important;
    border: none !important;
    font-weight: 700 !important;
    padding: 9px 14px !important;
}
.btn-read button:hover { filter: brightness(1.1) !important; }

/* Danger — red-tint */
.btn-danger button {
    background: var(--surface2) !important;
    color: var(--red) !important;
    border: 1px solid var(--surface3) !important;
    padding: 9px 14px !important;
}
.btn-danger button:hover {
    border-color: var(--red) !important;
    background: rgba(239,83,80,0.07) !important;
}

/* Refresh — teal outline */
.btn-refresh button {
    background: transparent !important;
    color: var(--teal) !important;
    border: 1px solid var(--teal-dim) !important;
    padding: 7px 12px !important;
    font-size: 10px !important;
}
.btn-refresh button:hover {
    background: rgba(0,188,212,0.08) !important;
    border-color: var(--teal) !important;
}

/* ── Sample question buttons ─────────────────────────────────────────── */
.sample-q {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
.sample-q button {
    background: var(--surface2) !important;
    color: #ffffff !important;
    border: 1px solid var(--surface3) !important;
    font-size: 10.5px !important;
    padding: 4px 8px !important;
    height: 44px !important;
    min-height: 44px !important;
    max-height: 44px !important;
    white-space: normal !important;
    line-height: 1.25 !important;
    text-align: center !important;
    width: 100% !important;
    overflow: hidden !important;
}
.sample-q button:hover {
    border-color: var(--blue) !important;
    color: #ffffff !important;
    background: rgba(41,182,246,0.1) !important;
}

/* ── Upload status / KB textbox — yellow text ───────────────────────── */
.upload-status textarea,
.upload-status textarea:disabled,
.upload-status textarea:read-only,
.copy-box textarea,
.copy-box textarea:disabled,
.copy-box textarea:read-only {
    color: var(--amber) !important;
    -webkit-text-fill-color: var(--amber) !important;
    opacity: 1 !important;
}

/* ── Copy-all / KB boxes ─────────────────────────────────────────────── */
.copy-box textarea,
.copy-box textarea:disabled,
.copy-box textarea:read-only {
    background: var(--surface2) !important;
    color: var(--amber) !important;
    -webkit-text-fill-color: var(--amber) !important;
    opacity: 1 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
}

/* ── Info panel ──────────────────────────────────────────────────────── */
.info-panel {
    background: var(--surface);
    border: 1px solid var(--surface2);
    border-left: 3px solid var(--blue);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    font-size: 12px;
    color: var(--amber) !important;
    line-height: 1.7;
}
.info-panel * {
    color: var(--amber) !important;
}

/* ── Delete-all confirmation banner ─────────────────────────────────── */
.confirm-banner {
    border: 1px solid var(--red) !important;
    border-radius: var(--radius-sm) !important;
    background: rgba(239,83,80,0.08) !important;
    padding: 6px 10px !important;
    align-items: center !important;
}
.confirm-alert {
    font-size: 11.5px;
    color: var(--red);
    flex: 1;
    padding: 4px 8px;
}
.btn-confirm-yes button {
    background: var(--red) !important;
    color: #fff !important;
    border: none !important;
    font-size: 11px !important;
    padding: 5px 12px !important;
}
.btn-confirm-no button {
    font-size: 11px !important;
    padding: 5px 12px !important;
}

/* ── Upload area ─────────────────────────────────────────────────────── */
.upload-zone {
    border: 2px dashed var(--blue-dark) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
    transition: border-color 0.2s !important;
}
.upload-zone:hover { border-color: var(--blue) !important; }

/* ── Thinking state: amber text in question box while disabled ───────── */
#question-input textarea:disabled,
#question-input textarea[disabled],
#question-input textarea:read-only {
    color: var(--amber) !important;
    -webkit-text-fill-color: var(--amber) !important;
    opacity: 1 !important;
}

/* ── Divider ─────────────────────────────────────────────────────────── */
.divider { height:1px; background: var(--surface2); margin:4px 0 8px; border:none; }

/* ── Footer ──────────────────────────────────────────────────────────── */
.rag-footer {
    text-align: center;
    padding: 12px 0 2px;
    color: #ffffff;
    font-size: 10px;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.5px;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# UI builder
# ══════════════════════════════════════════════════════════════════════════════

def build_ui() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="RAG Knowledge Assistant", theme=gr.themes.Base()) as demo:

        # ── Shared state ───────────────────────────────────────────────────
        session_id    = gr.State(str(uuid.uuid4()))
        last_answer   = gr.State("")        # plain text kept for TTS
        hist_snapshot = gr.State([])        # clean history snapshot for Stop restore

        # ══════════════════════════════════════════════════════════════════
        # BLUE STATUS BAR  (top of page)
        # ══════════════════════════════════════════════════════════════════
        status_html = gr.HTML(value=build_status_bar_html())

        with gr.Row(equal_height=True):
            gr.HTML("")          # spacer
            with gr.Column(scale=0, min_width=110, elem_classes="btn-refresh"):
                refresh_status_btn = gr.Button("🔄 Refresh Status")

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="rag-header">
            <h1>⚡ RAG Knowledge Assistant</h1>
            <p>Answers strictly from your uploaded PDFs · ChromaDB · Ollama / Groq</p>
            <span class="badge">CHROMADB · OLLAMA · SENTENCE-TRANSFORMERS · FASTAPI · GRADIO</span>
        </div>
        """)

        refresh_status_btn.click(
            fn=build_status_bar_html,
            inputs=[],
            outputs=[status_html],
        )

        # ══════════════════════════════════════════════════════════════════
        # TABS
        # ══════════════════════════════════════════════════════════════════
        with gr.Tabs(selected=0):

            # ── TAB 1 — CHAT ────────────────────────────────────────────
            with gr.TabItem("💬  Chat", id=0):

                chatbot = gr.Chatbot(
                    value=[],
                    type="messages",
                    height=490,
                    show_label=False,
                    bubble_full_width=False,
                    avatar_images=(None, "🤖"),
                    render_markdown=True,
                    sanitize_html=False,
                    elem_id="chatbot",
                )

                # Input row
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7):
                        user_input = gr.Textbox(
                            placeholder="Ask anything from your uploaded PDF documents…",
                            show_label=False,
                            lines=1,
                            max_lines=5,
                            elem_id="question-input",
                        )
                    with gr.Column(scale=2, min_width=120):
                        with gr.Column(elem_classes="btn-send"):
                            send_btn = gr.Button("Send ➤", variant="primary")

                with gr.Row():
                    with gr.Column(scale=1, elem_classes="btn-danger"):
                        stop_btn = gr.Button("⏹ Stop")
                    with gr.Column(scale=1, elem_classes="btn-read"):
                        read_btn = gr.Button("🔊 Read Last")

                gr.HTML("<hr class='divider'>")

                # Sample questions
                SAMPLE_QUESTIONS = [
                    "How you can help me?",
                    "How many documents are there?",
                    "What is the first document about?",
                    "Summarize each document in max 10 bullet points.",
                    "List top keywords or concepts mentioned.",
                    "What are the names mentioned in documents?",
                    "List all the links in the documents?",
                    "What problems or challenges are discussed?",
                ]

                sq_btns = []
                for row_qs in [SAMPLE_QUESTIONS[:4], SAMPLE_QUESTIONS[4:]]:
                    with gr.Row():
                        for q in row_qs:
                            with gr.Column(scale=1, elem_classes="sample-q"):
                                sq_btns.append((q, gr.Button(q)))

                gr.HTML("<hr class='divider'>")

                # Action row
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="btn-danger"):
                        clear_btn = gr.Button("🗑️ Clear Chat")
                    with gr.Column(scale=1, elem_classes="btn-action btn-copy-all"):
                        copy_all_btn = gr.Button("📋 Copy All")

                gr.HTML("""
                <div class="info-panel" style="margin-top:10px;">
                    💡 <strong>Tips:</strong>
                    Press <kbd>Enter</kbd> to send ·
                    Each answer has an <strong>⧉ copy</strong> button ·
                    <strong>Read Last</strong> toggles TTS on/off ·
                    If the answer isn't in your documents, the assistant says
                    <em>I Don't Know</em>.
                </div>
                """)

                # ── Event wiring ─────────────────────────────────────────
                async def on_send(msg, hist, sid):
                    if not msg.strip():
                        yield hist, sid, gr.update(value=msg, interactive=True), "", hist
                        return
                    with_question = hist + [{"role": "user", "content": msg}]
                    yield with_question, sid, gr.update(value="Thinking…", interactive=False), "", hist
                    await asyncio.sleep(0)  # flush first yield so "Thinking…" appears before we block
                    # run_in_executor keeps the event loop free so Stop's /cancel
                    # request can be received and processed during the backend call
                    loop = asyncio.get_event_loop()
                    new_hist, new_sid, _, plain = await loop.run_in_executor(
                        None, lambda: chat(msg, hist, sid)
                    )
                    yield new_hist, new_sid, gr.update(value="", interactive=True), plain, new_hist

                _SCROLL_JS = """() => {
                    const c = document.querySelector('#chatbot');
                    if (!c) return;
                    // Walk every child and scroll any overflowing container
                    [c, ...c.querySelectorAll('*')].forEach(el => {
                        const s = window.getComputedStyle(el);
                        if ((s.overflowY === 'auto' || s.overflowY === 'scroll') && el.scrollHeight > el.clientHeight)
                            el.scrollTop = el.scrollHeight;
                    });
                }"""

                send_event = send_btn.click(
                    on_send,
                    inputs=[user_input, chatbot, session_id],
                    outputs=[chatbot, session_id, user_input, last_answer, hist_snapshot],
                )
                send_event.then(None, inputs=[], outputs=[], js=_SCROLL_JS)
                submit_event = user_input.submit(
                    on_send,
                    inputs=[user_input, chatbot, session_id],
                    outputs=[chatbot, session_id, user_input, last_answer, hist_snapshot],
                )
                submit_event.then(None, inputs=[], outputs=[], js=_SCROLL_JS)

                def make_sq_handler(question):
                    async def handler(hist, sid):
                        async for val in on_send(question, hist, sid):
                            yield val
                    return handler

                sq_events = []
                for q, btn in sq_btns:
                    sq_ev = btn.click(
                        make_sq_handler(q),
                        inputs=[chatbot, session_id],
                        outputs=[chatbot, session_id, user_input, last_answer, hist_snapshot],
                        cancels=[send_event, submit_event],
                    )
                    sq_ev.then(None, inputs=[], outputs=[], js=_SCROLL_JS)
                    sq_events.append(sq_ev)

                # Two separate click events on stop_btn:
                # 1. Cancel ALL running chat generators
                stop_btn.click(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    cancels=[send_event, submit_event] + sq_events,
                )
                # 2. Restore chatbot from snapshot + reset textbox + fresh session_id.
                #    Fresh session_id abandons the backend session that may have received
                #    the cancelled question (the executor thread runs on after cancel).
                def on_stop_cleanup(hist):
                    return hist, gr.update(value="", interactive=True), str(uuid.uuid4())

                stop_btn.click(
                    fn=on_stop_cleanup,
                    inputs=[hist_snapshot],
                    outputs=[chatbot, user_input, session_id],
                )

                clear_btn.click(
                    clear_chat,
                    inputs=[session_id],
                    outputs=[chatbot, session_id],
                )

                # Copy All — JS clipboard
                copy_all_btn.click(
                    None, inputs=[], outputs=[],
                    js="""() => {
                        const rows = [...document.querySelectorAll('.message-wrap > div, .wrap > .message')];
                        const nodes = [...document.querySelectorAll('[data-testid="user"], [data-testid="bot"]')];
                        const allMsgs = nodes.length ? nodes
                            : [...document.querySelectorAll('.message.user, .message.bot')];
                        if (!allMsgs.length) return;
                        let text = '';
                        allMsgs.forEach(el => {
                            const isUser = el.dataset.testid === 'user' || el.classList.contains('user');
                            text += (isUser ? 'You: ' : 'Assistant: ') + (el.innerText || '').trim() + '\\n\\n';
                        });
                        navigator.clipboard.writeText(text.trim()).then(() => {
                            const btn = document.querySelector('.btn-copy-all button');
                            if (btn) { const orig = btn.textContent; btn.textContent = '✅ Copied!'; setTimeout(() => btn.textContent = orig, 1500); }
                        });
                    }""",
                )

                # TTS toggle — Read Last / Stop Reading (DOM-based, no state dependency)
                read_btn.click(
                    None, inputs=[], outputs=[],
                    js="""() => {
                        const btn = document.querySelector('.btn-read button');
                        if (window.speechSynthesis.speaking) {
                            window.speechSynthesis.cancel();
                            if (btn) btn.textContent = '🔊 Read Last';
                            return;
                        }
                        const botNodes = [...document.querySelectorAll('[data-testid="bot"]')];
                        const fallback = [...document.querySelectorAll('.message.bot')];
                        const nodes = botNodes.length ? botNodes : fallback;
                        if (!nodes.length) return;
                        const last = nodes[nodes.length - 1];
                        const clone = last.cloneNode(true);
                        clone.querySelectorAll('span[onclick], button').forEach(el => el.remove());
                        const text = (clone.innerText || '').replace(/[*_`#>~|]/g,'').replace(/\\n/g,' ').trim();
                        if (!text) return;
                        const u = new SpeechSynthesisUtterance(text);
                        u.rate = 0.93; u.pitch = 1.0;
                        u.onend = () => { if (btn) btn.textContent = '🔊 Read Last'; };
                        u.onerror = () => { if (btn) btn.textContent = '🔊 Read Last'; };
                        if (btn) btn.textContent = '⏹ Stop Reading';
                        window.speechSynthesis.speak(u);
                    }""",
                )

            # ── TAB 2 — UPLOAD ──────────────────────────────────────────
            with gr.TabItem("📂  Upload Documents", id=1):

                gr.HTML("""
                <div class="info-panel" style="margin-bottom:14px;">
                    📄 Upload <strong>PDF files</strong> — they are extracted, chunked,
                    embedded, and stored in <strong>ChromaDB</strong>.
                    Switch to <strong>Chat</strong> to query them.
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        pdf_upload = gr.File(
                            label="Drop PDFs here (or click to browse)",
                            file_types=[".pdf"],
                            file_count="multiple",
                            elem_classes="upload-zone",
                            height=200,
                        )
                        with gr.Row():
                            upload_btn = gr.Button("⬆️ Upload & Index", variant="primary")
                            refresh_kb  = gr.Button("🔄 Refresh",       variant="secondary")

                        upload_status = gr.Textbox(
                            label="Upload Status",
                            lines=3,
                            interactive=False,
                            value=get_kb_status(),
                            elem_classes="upload-status",
                        )

                        gr.HTML("<hr class='divider'>")
                        gr.HTML("<p style='font-family:monospace;font-size:11px;color:#6b7d96;margin-bottom:4px;'>🌐 Index a web page URL</p>")
                        with gr.Row():
                            with gr.Column(scale=5):
                                url_input = gr.Textbox(
                                    placeholder="https://example.com/article",
                                    show_label=False,
                                    lines=1,
                                )
                            with gr.Column(scale=2, min_width=130, elem_classes="btn-send"):
                                url_btn = gr.Button("🌐 Add URL", variant="primary")

                        gr.HTML("<hr class='divider'>")
                        gr.HTML("<p style='font-family:monospace;font-size:11px;color:#6b7d96;margin-bottom:4px;'>🗂️ Select files to delete</p>")

                        file_selector = gr.CheckboxGroup(
                            label="",
                            choices=get_file_list(),
                            value=[],
                            elem_classes="file-selector",
                        )

                        # Inline confirmation banner (hidden until Delete All clicked)
                        with gr.Row(visible=False, elem_classes="confirm-banner") as confirm_row:
                            gr.HTML("""
                            <div class="confirm-alert">
                                ⚠️ This will remove <strong style="color:var(--amber)">all files</strong> from the index. Continue?
                            </div>""")
                            confirm_yes_btn = gr.Button("✅ Yes, delete all", variant="primary",   elem_classes="btn-confirm-yes")
                            confirm_no_btn  = gr.Button("❌ Cancel",          variant="secondary", elem_classes="btn-confirm-no")

                        _has_files = bool(get_file_list())
                        with gr.Row():
                            delete_sel_btn = gr.Button("🗑️ Delete Selected", variant="secondary", elem_classes="btn-danger", interactive=_has_files)
                            delete_all_btn = gr.Button("💥 Delete All",      variant="secondary", elem_classes="btn-danger", interactive=_has_files)

                    with gr.Column(scale=1):
                        gr.HTML(
                            "<p style='font-family:monospace;font-size:11px;"
                            "color:#6b7d96;margin-bottom:6px;'>📚 ChromaDB Knowledge Base</p>"
                        )
                        kb_status_out = gr.Textbox(
                            label="",
                            lines=11,
                            interactive=False,
                            value=get_kb_status(),
                            elem_classes="copy-box",
                        )

                def handle_upload(files):
                    if not files:
                        return "⚠️  No files selected.", gr.update(), get_kb_status()
                    paths = files if isinstance(files, list) else [files]
                    results = []
                    for f in paths:
                        path = f.path if hasattr(f, "path") else str(f)
                        name = f.orig_name if hasattr(f, "orig_name") and f.orig_name else os.path.basename(path)
                        results.append(upload_to_backend(path, name))
                    files = get_file_list()
                    s, d = btn_state(files)
                    return "\n".join(results), gr.update(choices=files, value=[]), get_kb_status(), s, d

                upload_btn.click(
                    handle_upload,
                    inputs=[pdf_upload],
                    outputs=[upload_status, file_selector, kb_status_out, delete_sel_btn, delete_all_btn],
                )

                def handle_url(url):
                    msg   = upload_url_to_backend(url)
                    files = get_file_list()
                    s, d  = btn_state(files)
                    return msg, "", gr.update(choices=files, value=[]), get_kb_status(), s, d

                url_btn.click(
                    handle_url,
                    inputs=[url_input],
                    outputs=[upload_status, url_input, file_selector, kb_status_out, delete_sel_btn, delete_all_btn],
                )
                url_input.submit(
                    handle_url,
                    inputs=[url_input],
                    outputs=[upload_status, url_input, file_selector, kb_status_out, delete_sel_btn, delete_all_btn],
                )

                refresh_kb.click(
                    lambda: (file_selector_update(), get_kb_status()),
                    inputs=[],
                    outputs=[file_selector, kb_status_out],
                )
                delete_sel_btn.click(
                    lambda sel: delete_selected(sel) + btn_state(get_file_list()),
                    inputs=[file_selector],
                    outputs=[upload_status, file_selector, kb_status_out, delete_sel_btn, delete_all_btn],
                )
                delete_all_btn.click(
                    lambda: gr.update(visible=True),
                    inputs=[],
                    outputs=[confirm_row],
                )
                confirm_no_btn.click(
                    lambda: gr.update(visible=False),
                    inputs=[],
                    outputs=[confirm_row],
                )
                def do_delete_all():
                    status, fs, kb = delete_all()
                    s, d = btn_state([])
                    return status, fs, kb, gr.update(visible=False), s, d

                confirm_yes_btn.click(
                    do_delete_all,
                    inputs=[],
                    outputs=[upload_status, file_selector, kb_status_out, confirm_row, delete_sel_btn, delete_all_btn],
                )

        # ── Footer ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="rag-footer">
            RAG·MLOps &nbsp;·&nbsp; irajkoohi/rag_mlops &nbsp;·&nbsp;
            ChromaDB · Ollama · Sentence-Transformers · FastAPI · Gradio
        </div>
        """)

    demo.queue()
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7861, show_error=True)
