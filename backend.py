

# ...existing code...

"""
FastAPI backend for the Agentic Multimodal RAG system.
Exposes endpoints for document management and querying.
"""
import os

# On local dev (no SPACE_ID env var) default the embedding model to CPU.
# MPS (Apple Silicon GPU) can segfault after a previous crash leaves the Metal
# driver in a bad state. CPU is fast enough for local dev.
if not os.environ.get("SPACE_ID"):
    os.environ.setdefault("TORCH_DEVICE", "cpu")

import asyncio
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import threading
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import json
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from utils.document_processor import process_document_chunked, SUPPORTED_EXTENSIONS, extract_images, images_to_chunks
from utils.vector_store import VectorStoreManager
from utils.rag_engine import RAGEngine
from utils.memory import ConversationMemory, estimate_tokens
from utils.device import device_info
from utils.table_store import TableStore
from utils.image_store import ImageStore

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "./data")
VECTORSTORE_DIR = os.environ.get("VECTORSTORE_DIR", "./vectorstore")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


# HF Hub Dataset used for persistent user-uploaded file storage.
# Set MultiModalRag_dataset ("irajkoohi/AgenticMultiModalRag_dataset") and AgenticMultiModalRag_Token as Space secrets.
# Files uploaded via the app are pushed here and re-downloaded on every cold start,
# so they survive container restarts and redeployments.
HF_DATASET_REPO = os.environ.get("MultiModalRag_dataset", "irajkoohi/AgenticMultiModalRag_dataset")
HF_TOKEN = os.environ.get("AgenticMultiModalRag_Token", "")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# ─── HF Hub Persistent Storage Helpers ───────────────────────────────────────

def _hf_api():
    """Return a configured HfApi instance, or None if not set up."""
    if HF_DATASET_REPO and HF_TOKEN:
        from huggingface_hub import HfApi
        return HfApi(token=HF_TOKEN)
    return None



# ─── Sync Progress Tracker ─────────────────────────────────────────────
_sync_progress = {"status": "idle", "current": 0, "total": 0, "message": ""}
_sync_lock = threading.Lock()

def sync_from_hf_hub_with_progress():
    """Download user-uploaded files from HF Hub dataset to data dir, with progress tracking."""
    api = _hf_api()
    if not api:
        with _sync_lock:
            _sync_progress.update({"status": "error", "message": "HF_DATASET_REPO or HF_TOKEN not set"})
        return
    try:
        import huggingface_hub
        with _sync_lock:
            _sync_progress.update({"status": "listing", "message": "Listing files...", "current": 0, "total": 0})
        files = list(api.list_repo_files(HF_DATASET_REPO, repo_type="dataset"))
        data_files = [f for f in files if f.startswith("data/") and Path(f).suffix.lower() in SUPPORTED_EXTENSIONS and Path(f).name]
        total = len(data_files)
        with _sync_lock:
            _sync_progress.update({"status": "downloading", "message": f"Downloading {total} files...", "current": 0, "total": total})
        downloaded_count = 0
        for idx, path_in_repo in enumerate(data_files, 1):
            basename = Path(path_in_repo).name
            local_path = Path(DATA_DIR) / basename
            if local_path.exists():
                continue
            dl = huggingface_hub.hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=path_in_repo,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            shutil.copy2(dl, str(local_path))
            downloaded_count += 1
            with _sync_lock:
                _sync_progress.update({"status": "downloading", "message": f"Downloaded {idx}/{total}: {basename}", "current": idx, "total": total})
        with _sync_lock:
            _sync_progress.update({"status": "done", "message": f"Downloaded {downloaded_count} new file(s)", "current": total, "total": total})
    except Exception as e:
        with _sync_lock:
            _sync_progress.update({"status": "error", "message": str(e)})


def get_sync_progress():
    with _sync_lock:
        return dict(_sync_progress)


def push_to_hf_hub(filename: str):
    """Push a single file from data dir to the HF Hub dataset repo."""
    if not (HF_DATASET_REPO and HF_TOKEN):
        return
    api = _hf_api()
    if not api:
        return
    try:
        api.upload_file(
            path_or_fileobj=str(Path(DATA_DIR) / filename),
            path_in_repo=f"data/{filename}",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Upload {filename}",
        )
        logger.info(f"HF Hub: pushed '{filename}'")
    except Exception as e:
        logger.warning(f"HF Hub push failed for '{filename}': {e}")


def delete_from_hf_hub(filename: str):
    """Delete a single file from the HF Hub dataset repo."""
    api = _hf_api()
    if not api:
        return
    try:
        api.delete_file(
            path_in_repo=f"data/{filename}",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Delete {filename}",
        )
        logger.info(f"HF Hub: deleted '{filename}'")
    except Exception as e:
        logger.warning(f"HF Hub delete failed for '{filename}': {e}")


def sync_vectorstore_from_hf_hub():
    """Download persisted vectorstore from HF Hub dataset.
    Must be called BEFORE VectorStoreManager is initialized so ChromaDB can
    load existing embeddings and avoid re-indexing on cold start.
    """
    if not (HF_DATASET_REPO and HF_TOKEN):
        # print("[STARTUP] sync_vectorstore: SKIPPED — HF_DATASET_REPO or HF_TOKEN not set", flush=True)
        return
    try:
        import huggingface_hub
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        files = list(api.list_repo_files(HF_DATASET_REPO, repo_type="dataset"))
        vs_files = [f for f in files if f.startswith("vectorstore/")]
        if not vs_files:
            # print("[STARTUP] sync_vectorstore: no vectorstore in HF Hub — will build from scratch", flush=True)
            return
        # print(f"[STARTUP] sync_vectorstore: downloading {len(vs_files)} file(s)...", flush=True)
        for path_in_repo in vs_files:
            rel = path_in_repo[len("vectorstore/"):]
            if not rel:
                continue
            local = Path(VECTORSTORE_DIR) / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            dl = huggingface_hub.hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=path_in_repo,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            shutil.copy2(dl, str(local))
        # print(f"[STARTUP] sync_vectorstore: restored {len(vs_files)} file(s) OK", flush=True)
    except Exception as e:
        # print(f"[STARTUP] sync_vectorstore: FAILED — {e}", flush=True)
        logger.warning(f"HF Hub vectorstore sync failed: {e}")


def push_vectorstore_to_hf_hub():
    """Push the entire vectorstore directory to HF Hub dataset.
    Called after every index or delete operation so embeddings survive restarts.
    """
    if not (HF_DATASET_REPO and HF_TOKEN):
        return
    api = _hf_api()
    if not api:
        return
    try:
        # Compact SQLite before uploading to keep file sizes small.
        # ChromaDB accumulates free pages over time; VACUUM reclaims them.
        _sqlite_path = Path(VECTORSTORE_DIR) / "chroma.sqlite3"
        if _sqlite_path.exists():
            try:
                import sqlite3 as _sqlite3
                _conn = _sqlite3.connect(str(_sqlite_path), timeout=5)
                _conn.execute("VACUUM")
                _conn.close()
                logger.info("Vectorstore SQLite compacted before push")
            except Exception as _ve:
                logger.warning(f"SQLite VACUUM skipped: {_ve}")
        api.upload_folder(
            folder_path=VECTORSTORE_DIR,
            path_in_repo="vectorstore",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message="Update vectorstore",
            ignore_patterns=["*.lock", ".DS_Store", "*.wal", "*.shm"],
        )
        logger.info("HF Hub: pushed vectorstore")
    except Exception as e:
        logger.warning(f"HF Hub vectorstore push failed: {e}")


def push_tables_to_hf_hub():
    if not (HF_DATASET_REPO and HF_TOKEN):
        return
    api = _hf_api()
    if not api:
        return
    try:
        api.upload_folder(
            folder_path=str(Path(DATA_DIR) / "tables"),
            path_in_repo="tables",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message="Update tables",
            ignore_patterns=["*.lock", ".DS_Store"],
            delete_patterns="*",
        )
        logger.info("HF Hub: pushed tables")
    except Exception as e:
        logger.warning(f"HF Hub tables push failed: {e}")


def sync_tables_from_hf_hub():
    if not (HF_DATASET_REPO and HF_TOKEN):
        # print("[STARTUP] sync_tables: SKIPPED — HF_DATASET_REPO or HF_TOKEN not set", flush=True)
        return
    try:
        import huggingface_hub
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        files = list(api.list_repo_files(HF_DATASET_REPO, repo_type="dataset"))
        table_files = [f for f in files if f.startswith("tables/")]
        if not table_files:
            # print("[STARTUP] sync_tables: no tables found on HF Hub — will rely on on-demand extraction", flush=True)
            return
        # print(f"[STARTUP] sync_tables: downloading {len(table_files)} file(s)...", flush=True)
        tables_dir = Path(DATA_DIR) / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        for path_in_repo in table_files:
            rel = path_in_repo[len("tables/"):]
            if not rel:
                continue
            local = tables_dir / rel
            dl = huggingface_hub.hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=path_in_repo,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            shutil.copy2(dl, str(local))
            logger.info(f"HF Hub tables: restored '{rel}'")
        # print(f"[STARTUP] sync_tables: restored {len(table_files)} file(s) OK", flush=True)
    except Exception as e:
        # print(f"[STARTUP] sync_tables: FAILED — {e}", flush=True)
        logger.warning(f"HF Hub tables sync failed: {e}")


def push_images_to_hf_hub():
    if not (HF_DATASET_REPO and HF_TOKEN):
        return
    api = _hf_api()
    if not api:
        return
    try:
        api.upload_folder(
            folder_path=str(Path(DATA_DIR) / "images"),
            path_in_repo="images",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message="Update images",
            ignore_patterns=["*.lock", ".DS_Store"],
            delete_patterns="*",
        )
        logger.info("HF Hub: pushed images")
    except Exception as e:
        logger.warning(f"HF Hub images push failed: {e}")


def sync_images_from_hf_hub():
    if not (HF_DATASET_REPO and HF_TOKEN):
        return
    try:
        import huggingface_hub
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        files = list(api.list_repo_files(HF_DATASET_REPO, repo_type="dataset"))
        image_files = [f for f in files if f.startswith("images/")]
        if not image_files:
            return
        images_dir = Path(DATA_DIR) / "images"
        for path_in_repo in image_files:
            rel = path_in_repo[len("images/"):]
            if not rel:
                continue
            local = images_dir / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            dl = huggingface_hub.hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=path_in_repo,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            shutil.copy2(dl, str(local))
            logger.info(f"HF Hub images: restored '{rel}'")
    except Exception as e:
        logger.warning(f"HF Hub images sync failed: {e}")


def _copy_committed_files():
    """Copy baseline PDFs committed to the Space repo under _secrets/data/ into DATA_DIR.
    These are always available in the Space even without HF Hub Dataset configured.
    """
    secrets_data = Path("_secrets/data")
    if not secrets_data.exists():
        return
    for fp in secrets_data.iterdir():
        if fp.suffix.lower() in SUPPORTED_EXTENSIONS:
            dest = Path(DATA_DIR) / fp.name
            if not dest.exists():
                shutil.copy2(str(fp), str(dest))
                logger.info(f"Copied committed file '{fp.name}' → data/")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Pre-init: restore persisted data so ChromaDB loads existing embeddings ──
# This runs BEFORE VectorStoreManager is created. On HF Spaces (SPACE_ID is set)
# it downloads the latest data + vectorstore from HF Hub. Locally, files are
# already on disk so we skip the network calls for a fast startup.
_IS_HF_SPACE = bool(os.environ.get("SPACE_ID"))

print(
    f"[STARTUP] IS_HF_SPACE={_IS_HF_SPACE} | "
    f"HF_DATASET_REPO={'SET' if HF_DATASET_REPO else 'NOT SET'} | "
    f"HF_TOKEN={'SET' if HF_TOKEN else 'NOT SET'}",
    flush=True,
)
_copy_committed_files()
if _IS_HF_SPACE:
    sync_vectorstore_from_hf_hub()
    sync_from_hf_hub_with_progress()
    sync_tables_from_hf_hub()
    sync_images_from_hf_hub()
else:
    logger.info("Pre-init: local run — skipping HF Hub sync (files already on disk).")

# ─── Singletons ───────────────────────────────────────────────────────────────
vs = VectorStoreManager(persist_dir=VECTORSTORE_DIR)
_vs_chunks = vs.total_chunks()
_vs_sources = vs.list_sources()
_data_files = [f.name for f in Path(DATA_DIR).iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
# print(
#     f"[STARTUP] VS loaded: {_vs_chunks} chunks, {len(_vs_sources)} source(s): {_vs_sources}",
#     flush=True,
# )
# print(f"[STARTUP] DATA_DIR files: {_data_files}", flush=True)
rag = RAGEngine(vector_store=vs, model=OLLAMA_MODEL)
memory = ConversationMemory()
ts = TableStore()
img_store = ImageStore()

# ─── Agents / Tools ───────────────────────────────────────────────────────────
from Tools.llm_tool import LLMTool
from Tools.vector_search_tool import VectorSearchTool
from Tools.table_extraction_tool import TableExtractionTool
from Agents.router_agent import RouterAgent
from Agents.sql_gen_agent import SQLGenAgent
from Agents.table_agent import TableAgent
from Agents.doc_image_agent import DocImageAgent
from Agents.grading_agent import GradingAgent
from Agents.hallucination_agent import HallucinationAgent
from Agents.supervisor_agent import SupervisorAgent
llm_tool = LLMTool(rag)
vs_tool = VectorSearchTool(vs)
table_extractor = TableExtractionTool(llm_tool)
router = RouterAgent()
sql_gen = SQLGenAgent(llm_tool)
table_agent_inst = TableAgent(llm_tool, ts, table_extractor, DATA_DIR, SUPPORTED_EXTENSIONS)
doc_image_agent_inst = DocImageAgent(rag, vs_tool)
grading_agent_inst = GradingAgent(llm_tool)
hallucination_agent_inst = HallucinationAgent(llm_tool)
supervisor = SupervisorAgent(
    router, sql_gen, table_agent_inst, doc_image_agent_inst,
    grading_agent_inst, hallucination_agent_inst, vs_tool,
)


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Agentic Multimodal RAG API", version="1.0.0")

# CORS is a browser security mechanism that blocks web pages from making requests to a different domain than the one that served the page. For example, if your Gradio frontend runs on localhost:7860 and tries to call your FastAPI backend on localhost:8000, the browser would normally block that request.
app.add_middleware( 
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    n_results: int = 8
    temperature: float = 0.0
    source_filter: List[str] = []


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    tokens_user: int = 0
    tokens_assistant: int = 0
    chunks_used: int = 0
    sql_query: str = ""
    answer_method: str = "rag"  # "rag" | "table_query"


class StatusResponse(BaseModel):
    documents: List[str]
    total_chunks: int
    data_dir_files: List[str]
    model: str
    device: str


class URLIndexRequest(BaseModel):
    url: str
    max_depth: int = 2
    max_pages: int = 50


# ─── Helper ───────────────────────────────────────────────────────────────────
def index_file(filepath: str) -> int:
    """Process and index a file into the vector store, and extract tables into TableStore."""
    chunks = process_document_chunked(filepath)
    source_name = Path(filepath).name
    vs.remove_document(source_name)
    n = vs.add_documents(chunks, source_name)
    try:
        dfs = table_extractor.extract(filepath)
        if dfs:
            ts.save(source_name, dfs)
    except Exception as e:
        logger.warning(f"Table extraction failed for '{source_name}': {e}")
    return n


def index_all_data_dir():
    """Index all supported files in DATA_DIR on startup."""
    indexed_sources = set(vs.list_sources())
    for fp in Path(DATA_DIR).iterdir():
        if fp.suffix.lower() in SUPPORTED_EXTENSIONS and fp.name not in indexed_sources:
            try:
                n = index_file(str(fp))
                logger.info(f"Indexed '{fp.name}': {n} chunks")
            except Exception as e:
                logger.error(f"Failed to index '{fp.name}': {e}")


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    # ChromaDB PersistentClient already restores the vectorstore from disk on init.
    # Do NOT auto-index data/ here — that would re-index files the user deleted
    # from the index, making them reappear after every restart.
    # On HF Spaces, sync_from_hf_hub() + sync_vectorstore_from_hf_hub() run at
    # module load (before this), so the vectorstore is already fully restored.

    def _backfill_tables():
        for fp in Path(DATA_DIR).iterdir():
            if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            src = fp.name
            if ts.was_attempted(src):
                continue
            try:
                dfs = table_extractor.extract(str(fp))
                ts.save(src, dfs)
                if dfs:
                    logger.info(f"Startup backfill: {len(dfs)} table(s) for '{src}'")
            except Exception as e:
                logger.warning(f"Startup backfill failed for '{src}': {e}")
                ts.save(src, [])

    def _backfill_images():
        for fp in Path(DATA_DIR).iterdir():
            if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            src = fp.name
            if img_store.was_attempted(src):
                continue
            try:
                images = extract_images(str(fp))
                img_store.save(src, images)
                if images:
                    logger.info(f"Startup backfill: {len(images)} image(s) for '{src}'")
            except Exception as e:
                logger.warning(f"Startup image backfill failed for '{src}': {e}")
                img_store.save(src, [])

    def _index_missing_files():
        """Index any file in data/ that is not yet in the vectorstore.
        Runs on every startup to self-heal after a stale or failed vectorstore restore.
        Safe to run because deleted files are now removed from disk too.
        """
        indexed = set(vs.list_sources())
        missing = [
            fp for fp in Path(DATA_DIR).iterdir()
            if fp.suffix.lower() in SUPPORTED_EXTENSIONS and fp.name not in indexed
        ]
        if not missing:
            return
        logger.warning(
            "%d file(s) in data/ not in vectorstore — indexing: %s",
            len(missing), [f.name for f in missing],
        )
        failed = []
        for fp in missing:
            try:
                n = index_file(str(fp))
                logger.warning(f"Startup index OK: '{fp.name}' — {n} chunks")
            except Exception as e:
                logger.error(f"Startup index FAILED: '{fp.name}': {e}", exc_info=True)
                failed.append(fp.name)
        logger.warning(
            "Startup index complete: %d OK, %d failed. VS now has %d chunks.",
            len(missing) - len(failed), len(failed), vs.total_chunks(),
        )
        if _IS_HF_SPACE:
            if failed:
                logger.warning(f"Skipping vectorstore push — {len(failed)} file(s) failed: {failed}")
            else:
                push_vectorstore_to_hf_hub()

    loop = asyncio.get_event_loop()
    # loop.run_in_executor(None, _index_missing_files)  # Disabled: do not auto-index data/ on startup
    loop.run_in_executor(None, _backfill_tables)
    loop.run_in_executor(None, _backfill_images)
    logger.info("HTTP server ready.")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/status", response_model=StatusResponse)
async def get_status():
    data_files = [
        f.name for f in Path(DATA_DIR).iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return StatusResponse(
        documents=vs.list_sources(),
        total_chunks=vs.total_chunks(),
        data_dir_files=data_files,
        model=rag.model,
        device=device_info()["label"],
    )


@app.get("/debug")
async def get_debug():
    """Diagnostic endpoint — shows startup config and current VS/data state."""
    data_files = [
        f.name for f in Path(DATA_DIR).iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    vs_files = []
    try:
        vs_files = list(Path(VECTORSTORE_DIR).rglob("*"))
        vs_files = [str(p.relative_to(VECTORSTORE_DIR)) for p in vs_files if p.is_file()]
    except Exception:
        pass
    return {
        "is_hf_space": _IS_HF_SPACE,
        "hf_dataset_repo_set": bool(HF_DATASET_REPO),
        "hf_token_set": bool(HF_TOKEN),
        "vs_chunks": vs.total_chunks(),
        "vs_sources": vs.list_sources(),
        "data_dir_files": data_files,
        "vectorstore_files": vs_files,
    }


@app.post("/documents/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Save a document to disk and start background indexing. Returns immediately."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Supported: {SUPPORTED_EXTENSIONS}")

    save_path = Path(DATA_DIR) / file.filename
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    with _upload_lock:
        _upload_jobs[file.filename] = {"status": "processing"}

    background_tasks.add_task(_index_background, file.filename, str(save_path))
    return {
        "message": f"⏳ Indexing started for '{file.filename}' — polling for status.",
        "status": "processing",
        "filename": file.filename,
    }


@app.get("/documents/upload/status")
async def upload_status(filename: str):
    """Poll the indexing status of a background file upload."""
    with _upload_lock:
        job = _upload_jobs.get(filename)
    if job is None:
        raise HTTPException(404, f"No upload job found for '{filename}'")
    return job


@app.delete("/documents/{filename:path}")
async def delete_document(filename: str):
    """Remove a document's embeddings and delete the file from disk and HF Hub."""
    removed_chunks = vs.remove_document(filename)
    ts.remove(filename)
    img_store.remove(filename)
    file_path = Path(DATA_DIR) / filename
    file_path.unlink(missing_ok=True)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, delete_from_hf_hub, filename)
    await loop.run_in_executor(None, push_vectorstore_to_hf_hub)
    await loop.run_in_executor(None, push_tables_to_hf_hub)
    await loop.run_in_executor(None, push_images_to_hf_hub)
    if removed_chunks > 0:
        return {"message": f"Removed '{filename}' ({removed_chunks} chunks)."}
    else:
        raise HTTPException(404, f"No indexed chunks found for '{filename}'.")


@app.post("/reextract")
async def reextract_all():
    """Re-run table and image extraction on all files in DATA_DIR. Streams per-file progress as SSE."""
    files = [f for f in Path(DATA_DIR).iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    async def _generate():
        results = {}
        total = len(files)
        loop = asyncio.get_running_loop()

        async def _keepalives(fut, interval: float = 20.0):
            """Async generator: yield SSE keepalive comments every `interval` seconds until fut is done."""
            while not fut.done():
                try:
                    await asyncio.wait_for(asyncio.shield(fut), timeout=interval)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"

        for i, f in enumerate(files):
            source_name = f.name
            tables_saved = 0
            images_saved = 0
            yield f"data: {json.dumps({'type': 'progress', 'file': source_name, 'index': i + 1, 'total': total})}\n\n"
            try:
                fut = loop.run_in_executor(None, table_extractor.extract, str(f))
                async for ka in _keepalives(fut):
                    yield ka
                dfs = fut.result()
                await loop.run_in_executor(None, ts.save, source_name, dfs)
                tables_saved = await loop.run_in_executor(None, ts.merged_count, source_name)
            except Exception as e:
                logger.warning(f"Table reextract failed for '{source_name}': {e}")
            try:
                fut = loop.run_in_executor(None, extract_images, str(f))
                async for ka in _keepalives(fut):
                    yield ka
                images = fut.result()
                await loop.run_in_executor(None, img_store.save, source_name, images)
                images_saved = len(images)
            except Exception as e:
                logger.warning(f"Image reextract failed for '{source_name}': {e}")
            results[source_name] = {"tables": tables_saved, "images": images_saved}
        loop.run_in_executor(None, push_tables_to_hf_hub)
        loop.run_in_executor(None, push_images_to_hf_hub)
        yield f"data: {json.dumps({'type': 'complete', 'results': results})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.delete("/documents")
async def delete_all_documents():
    """Remove ALL embeddings and delete all user files from disk and HF Hub."""
    filenames = [f.name for f in Path(DATA_DIR).iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    removed = vs.clear_all()
    ts.clear_all()
    img_store.clear_all()
    for name in filenames:
        (Path(DATA_DIR) / name).unlink(missing_ok=True)
    loop = asyncio.get_running_loop()
    for name in filenames:
        await loop.run_in_executor(None, delete_from_hf_hub, name)
    await loop.run_in_executor(None, push_vectorstore_to_hf_hub)
    await loop.run_in_executor(None, push_tables_to_hf_hub)
    await loop.run_in_executor(None, push_images_to_hf_hub)
    return {"message": f"Removed {removed} indexed chunks and {len(filenames)} file(s).", "chunks_removed": removed}


# ─── Upload job tracker ──────────────────────────────────────────────────────
_upload_jobs: dict = {}   # filename → {"status": "processing"|"done"|"error", ...}
_upload_lock = threading.Lock()


def _index_background(filename: str, save_path: str):
    """Runs in a background thread: index file, persist to HF Hub, update job status."""
    def _set_phase(msg: str):
        with _upload_lock:
            _upload_jobs[filename]["phase"] = msg
    try:
        _set_phase("parsing document…")
        chunks = process_document_chunked(save_path)
        total = len(chunks)
        source_name = Path(save_path).name
        vs.remove_document(source_name)

        # Embed and upsert in batches of 150 so we can report live progress and
        # avoid one massive blocking encode() call (critical on HF Space CPU).
        EMBED_BATCH = 150
        done = 0
        for i in range(0, max(total, 1), EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            end = min(i + EMBED_BATCH, total)
            _set_phase(f"embedding chunks {i + 1}–{end} of {total}…")
            vs.add_documents(batch, source_name, chunk_offset=i)
            done += len(batch)

        n_chunks = done

        _set_phase("extracting tables…")
        try:
            dfs = table_extractor.extract(save_path)
            ts.save(source_name, dfs)
            if dfs:
                logger.info(f"TableStore: saved {len(dfs)} table(s) for '{source_name}'")
        except Exception as e:
            logger.warning(f"Table extraction failed for '{source_name}': {e}")

        _set_phase("extracting images…")
        try:
            images = extract_images(save_path)
            img_store.save(source_name, images)
            if images:
                logger.info(f"ImageStore: saved {len(images)} image(s) for '{source_name}'")
                _set_phase("indexing image text (OCR)…")
                img_chunks = images_to_chunks(save_path, images)
                if img_chunks:
                    vs.add_documents(img_chunks, source_name)
                    logger.info(f"Image OCR: {len(img_chunks)} chunk(s) indexed for '{source_name}'")
        except Exception as e:
            logger.warning(f"Image extraction failed for '{source_name}': {e}")

        # Mark done IMMEDIATELY so the frontend poll resolves without waiting for
        # the (potentially slow) HF Hub push that follows.
        with _upload_lock:
            _upload_jobs[filename] = {
                "status": "done",
                "message": f"Uploaded and indexed '{filename}' ({n_chunks} chunks).",
                "chunks": n_chunks,
            }
        logger.info(f"Background index done: '{filename}' — {n_chunks} chunks")
        # Push to HF Hub after marking done so a slow upload doesn't block status.
        push_to_hf_hub(filename)
        push_vectorstore_to_hf_hub()
        push_tables_to_hf_hub()
        push_images_to_hf_hub()
    except Exception as e:
        logger.error(f"Background index failed for '{filename}': {e}", exc_info=True)
        # File is kept on disk even if indexing fails — user can retry
        with _upload_lock:
            _upload_jobs[filename] = {"status": "error", "message": str(e)}


# ─── URL crawl job tracker ────────────────────────────────────────────────────
_crawl_jobs: dict = {}   # url → {"status": "crawling"|"done"|"error", ...}
_crawl_lock = threading.Lock()


def _crawl_background(url: str, max_depth: int, max_pages: int):
    """Runs in a background thread: crawl + index, then update job status."""
    from utils.url_processor import crawl_url
    try:
        vs.remove_document(url)
        chunks, crawled_urls = crawl_url(url, max_depth=max_depth, max_pages=max_pages)
        if not chunks:
            with _crawl_lock:
                _crawl_jobs[url] = {"status": "error", "message": "No content extracted."}
            return
        n_chunks = vs.add_documents(chunks, url)
        with _crawl_lock:
            _crawl_jobs[url] = {
                "status": "done",
                "message": (
                    f"Indexed {len(crawled_urls)} page(s) and file(s) "
                    f"({n_chunks} chunks) from {url}"
                ),
                "pages": len(crawled_urls),
                "chunks": n_chunks,
            }
        logger.info(f"Crawl done: {url} — {len(crawled_urls)} pages, {n_chunks} chunks")
    except Exception as e:
        logger.error(f"Crawl failed for {url}: {e}", exc_info=True)
        with _crawl_lock:
            _crawl_jobs[url] = {"status": "error", "message": str(e)}


@app.post("/documents/url")
async def index_url(req: URLIndexRequest, background_tasks: BackgroundTasks):
    """Start a background crawl of a URL (2 levels deep). Returns immediately."""
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "URL must start with http:// or https://")

    with _crawl_lock:
        _crawl_jobs[url] = {"status": "crawling"}

    background_tasks.add_task(_crawl_background, url, req.max_depth, req.max_pages)
    return {
        "message": f"⏳ Crawling started for {url} — refresh the document list in ~30 s.",
        "status": "crawling",
        "url": url,
    }


@app.get("/documents/url/status")
async def url_crawl_status(url: str):
    """Poll the status of a background URL crawl."""
    with _crawl_lock:
        job = _crawl_jobs.get(url)
    if job is None:
        raise HTTPException(404, f"No crawl job found for {url}")
    return job






@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """Query the RAG system."""
    try:
        def _run_query():
            result = supervisor.handle(
                req.question,
                memory,
                n_results=req.n_results,
                temperature=req.temperature,
                source_filter=req.source_filter or None,
            )
            return result

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _run_query)

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            tokens_user=estimate_tokens(req.question),
            tokens_assistant=estimate_tokens(result["answer"]),
            chunks_used=result["chunks_used"],
            sql_query=result["sql_query"],
            answer_method=result["answer_method"],
        )
    except Exception as e:
        logger.error(f"Query endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/clear")
async def clear_memory():
    memory.clear()
    return {"message": "Conversation memory cleared."}


@app.get("/memory/stats")
async def memory_stats():
    from utils.memory import estimate_tokens
    total_tokens = sum(estimate_tokens(m.content) for m in memory.messages)
    summary_tokens = estimate_tokens(memory.summary) if memory.summary else 0
    return {
        "message_count": len(memory.messages),
        "total_tokens": total_tokens + summary_tokens,
        "has_summary": memory.summary is not None,
        "max_tokens": memory.max_tokens,
    }


@app.get("/models")
async def list_models():
    return {"models": rag.list_available_models(), "current": OLLAMA_MODEL}
