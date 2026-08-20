"""
⚡ Llama.cpp Turbo Desktop - Unified Python Backend Server
Exposes high-performance REST and SSE APIs for Electron Desktop frontend:
- OpenAI-compatible Chat & Completion endpoints
- Model scanning and asynchronous loading with baseline fallback
- GGUF Inspector with binary metadata parsing
- Google TurboQuant KV Cache configuration & diagnostics
- Session & history manager
- Hugging Face model downloader & search
- System hardware monitor (CPU / RAM)
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid

# Windows UTF-8 stdout fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import psutil
import uvicorn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.downloader import CURATED_MODELS, DownloadWorker, list_repo_gguf_files, search_hf_models
from src.core.gguf_reader import GGUFInspector
from src.core.gpu_detector import GPUDetector, get_hardware_info
from src.core.llama_engine import LlamaEngine
from src.core.model_scanner import scan_for_gguf_models
from src.core.session_manager import SessionManager
from src.core.turbo_quant import TurboQuantManager

# Initialize Core Engine & Managers
app = FastAPI(title="Llama.cpp Turbo Desktop Backend", version="1.0.0")

# Enable CORS for local Electron origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = LlamaEngine()
session_mgr = SessionManager()
turbo_mgr = TurboQuantManager()
gpu_detector = GPUDetector.get_instance()

# Global state for active downloads and model scan caches
active_downloads: Dict[str, DownloadWorker] = {}
download_progress_state: Dict[str, Dict[str, Any]] = {}
scanned_models_cache: List[Dict[str, Any]] = []


# ----------------------------------------------------
# Request Schemas
# ----------------------------------------------------
class ModelLoadRequest(BaseModel):
    model_path: str
    n_gpu_layers: Optional[int] = -1
    compute_mode: Optional[str] = "auto"
    n_ctx: Optional[int] = 4096
    n_threads: Optional[int] = max(1, (os.cpu_count() or 4) - 1)
    n_batch: Optional[int] = 512
    flash_attn: Optional[bool] = True
    turbo_bits: Optional[int] = 4
    turbo_hadamard: Optional[bool] = True
    turbo_sparsity: Optional[float] = 0.20


class ChatMessageSchema(BaseModel):
    role: str
    content: str


class ChatCompletionRequestSchema(BaseModel):
    messages: List[ChatMessageSchema]
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    min_p: Optional[float] = 0.05
    repeat_penalty: Optional[float] = 1.1
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    max_tokens: Optional[int] = 2048
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = True
    grammar_type: Optional[str] = "none"
    custom_grammar: Optional[str] = ""


class CompletionRequestSchema(BaseModel):
    prompt: Union[str, List[str]]
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    min_p: Optional[float] = 0.05
    repeat_penalty: Optional[float] = 1.1
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    max_tokens: Optional[int] = 2048
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = True
    grammar_type: Optional[str] = "none"
    custom_grammar: Optional[str] = ""


class TurboConfigRequest(BaseModel):
    enabled: bool = True
    bits: int = 4
    hadamard: bool = True
    sparsity: float = 0.20


class InspectRequest(BaseModel):
    file_path: str


class SessionCreateRequest(BaseModel):
    title: Optional[str] = "New Conversation"
    system_prompt: Optional[str] = "You are a helpful, expert AI assistant."


class SessionUpdateRequest(BaseModel):
    title: Optional[str] = None
    system_prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None


class DownloadStartRequest(BaseModel):
    repo_id: str
    filename: str


# ----------------------------------------------------
# 1. Health & System Diagnostics
# ----------------------------------------------------
@app.get("/health")
def health_check():
    mem = psutil.virtual_memory()
    compute_info = engine.get_computation_status()
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": engine.is_loaded,
        "model_name": engine.model_name,
        "model_path": engine.model_path,
        "active_compute_backend": getattr(engine, "active_compute_backend", compute_info["active_backend"]),
        "turbo_quant": {
            "enabled": engine.config.get("turbo_enabled", True),
            "bits": engine.config.get("turbo_bits", 4),
            "ratio": f"{16 / engine.config.get('turbo_bits', 4):.1f}x",
            "hadamard": engine.config.get("turbo_hadamard", True),
            "sparsity": engine.config.get("turbo_sparsity", 0.20),
        },
        "system": {
            "cpu_percent": round(psutil.cpu_percent(), 1),
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_percent": round(mem.percent, 1),
        },
        "gpu_computation": compute_info,
    }


@app.get("/v1/system/stats")
def get_system_stats():
    mem = psutil.virtual_memory()
    compute_info = engine.get_computation_status()
    return {
        "cpu_percent": round(psutil.cpu_percent(), 1),
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_percent": round(mem.percent, 1),
        "gpu": compute_info,
    }


@app.get("/v1/system/hardware")
def get_system_hardware():
    """Returns complete GPU, CUDA, Vulkan, and CPU hardware diagnostic capabilities."""
    return get_hardware_info()


# ----------------------------------------------------
# 2. Model Discovery, Scanning, Loading & Inspection
# ----------------------------------------------------
@app.get("/v1/models")
def list_models():
    """Lists currently discovered GGUF models and current active model."""
    global scanned_models_cache
    if not scanned_models_cache:
        scanned_models_cache = scan_for_gguf_models()

    return {
        "object": "list",
        "count": len(scanned_models_cache),
        "active_model": {
            "name": engine.model_name,
            "path": engine.model_path,
            "is_loaded": engine.is_loaded,
            "compute_backend": getattr(engine, "active_compute_backend", "CPU"),
            "layers_offloaded": getattr(engine, "active_layers_offloaded", 0),
        },
        "data": scanned_models_cache,
        "models": scanned_models_cache,
    }


class ScanRequest(BaseModel):
    custom_path: Optional[str] = None
    deep: Optional[bool] = False


@app.get("/v1/models/scan")
@app.post("/v1/models/scan")
def trigger_scan(
    custom_path: Optional[str] = None,
    deep: Optional[bool] = False,
    req: Optional[ScanRequest] = None,
):
    """Forces a fresh scan across all system drives and LLM storage folders or a custom target folder."""
    global scanned_models_cache
    target_path = custom_path
    target_deep = deep
    if req:
        if req.custom_path:
            target_path = req.custom_path
        if req.deep is not None:
            target_deep = req.deep

    new_models = scan_for_gguf_models(custom_path=target_path, deep=bool(target_deep))
    if target_path:
        existing_paths = {m["path"] for m in scanned_models_cache}
        for nm in new_models:
            if nm["path"] not in existing_paths:
                scanned_models_cache.append(nm)
                existing_paths.add(nm["path"])
    else:
        scanned_models_cache = new_models

    return {
        "status": "success",
        "object": "list",
        "count": len(scanned_models_cache),
        "data": scanned_models_cache,
        "models": scanned_models_cache,
    }


@app.post("/v1/models/load")
def load_model(req: ModelLoadRequest):
    """Loads a GGUF model into memory with specified hardware (CUDA/Vulkan/CPU) and TurboQuant parameters."""
    if not os.path.exists(req.model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {req.model_path}")

    # Vision adapter filter
    base_name = os.path.basename(req.model_path).lower()
    if base_name.startswith("mmproj-") or "mmproj" in base_name:
        raise HTTPException(
            status_code=400,
            detail="The selected file is a multimodal vision projector (mmproj), not a standalone language model."
        )

    config = {
        "n_gpu_layers": req.n_gpu_layers if req.n_gpu_layers is not None else -1,
        "compute_mode": req.compute_mode or "auto",
        "n_ctx": req.n_ctx or 4096,
        "n_threads": req.n_threads or max(1, (os.cpu_count() or 4) - 1),
        "n_batch": req.n_batch or 512,
        "flash_attn": req.flash_attn if req.flash_attn is not None else True,
        "turbo_enabled": True,
        "turbo_bits": req.turbo_bits or 4,
        "turbo_hadamard": req.turbo_hadamard if req.turbo_hadamard is not None else True,
        "turbo_sparsity": req.turbo_sparsity if req.turbo_sparsity is not None else 0.20,
    }

    success = engine.load_model(req.model_path, config=config)
    if not success:
        err_msg = getattr(engine, "last_error", "Unknown initialization error.")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {err_msg}")

    return {
        "status": "success",
        "model_name": engine.model_name,
        "model_path": engine.model_path,
        "active_compute_backend": getattr(engine, "active_compute_backend", "CPU"),
        "layers_offloaded": getattr(engine, "active_layers_offloaded", 0),
        "config": engine.config,
    }


@app.post("/v1/models/unload")
def unload_model():
    """Unloads the active model from RAM and releases all buffers."""
    engine.unload_model()
    import gc
    gc.collect()
    return {"status": "success", "message": "Model unloaded successfully."}


@app.post("/v1/models/inspect")
def inspect_model(req: InspectRequest):
    """Fast binary inspection of GGUF architecture, tensors, metadata, and TurboQuant savings."""
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="File does not exist.")

    inspector = GGUFInspector(req.file_path)
    meta = inspector.inspect()

    # Calculate TurboQuant estimates
    tq = TurboQuantManager()
    tq_estimates = tq.estimate_savings(
        raw_kv_mb=max(10.0, (meta.get("context_length", 4096) * 2 * 32 * 128 * 2) / (1024 * 1024)),
        model_weights_mb=meta.get("file_size_gb", 1.0) * 1024.0,
    )
    meta["turboquant_estimates"] = tq_estimates
    return meta


# ----------------------------------------------------
# 3. Google TurboQuant Configuration
# ----------------------------------------------------
@app.post("/v1/turboquant/config")
def update_turboquant_config(req: TurboConfigRequest):
    """Updates global TurboQuant quantization bit-depth, Hadamard transform, and sparsity ratio."""
    engine.update_turbo_quant(req.enabled, req.bits, req.hadamard, req.sparsity)
    turbo_mgr.update_config(req.enabled, req.bits, req.hadamard, req.sparsity)
    return {
        "status": "success",
        "turbo_enabled": req.enabled,
        "turbo_bits": req.bits,
        "compression_ratio": f"{16 / req.bits:.1f}x" if req.enabled else "1.0x",
        "hadamard": req.hadamard,
        "sparsity": req.sparsity,
    }


# ----------------------------------------------------
# 4. OpenAI-Compatible Chat & Text Completions (Streaming SSE)
# ----------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequestSchema):
    if not engine.is_loaded:
        raise HTTPException(
            status_code=400,
            detail="No model is currently loaded in the desktop application. Please load a GGUF model first."
        )

    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    stop_seqs = req.stop if isinstance(req.stop, list) else ([req.stop] if req.stop else [])

    params = {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "min_p": req.min_p,
        "repeat_penalty": req.repeat_penalty,
        "presence_penalty": req.presence_penalty,
        "frequency_penalty": req.frequency_penalty,
        "max_tokens": req.max_tokens,
        "stop": stop_seqs,
        "grammar_type": req.grammar_type,
        "custom_grammar": req.custom_grammar,
    }

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    model_name = req.model or engine.model_name

    if req.stream:
        async def event_generator():
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()

            def worker():
                try:
                    for chunk in engine.generate_chat_stream(msgs, params):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            threading.Thread(target=worker, daemon=True).start()

            token_count = 0
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    err_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": f"\n\n[Error: {str(item)}]"}, "finish_reason": "error"}],
                    }
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    break

                token_count += 1
                data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": item},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"

            # Finish chunk
            done_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    else:
        full_text = ""
        for chunk in engine.generate_chat_stream(msgs, params):
            full_text += chunk

        prompt_tokens = len(" ".join([m["content"] for m in msgs]).split())
        completion_tokens = len(full_text.split())

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created_time,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


@app.post("/v1/completions")
async def text_completions(req: CompletionRequestSchema):
    if not engine.is_loaded:
        raise HTTPException(
            status_code=400,
            detail="No model is currently loaded in the desktop application."
        )

    prompt_text = req.prompt if isinstance(req.prompt, str) else "\n".join(req.prompt)
    stop_seqs = req.stop if isinstance(req.stop, list) else ([req.stop] if req.stop else [])

    params = {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "min_p": req.min_p,
        "repeat_penalty": req.repeat_penalty,
        "presence_penalty": req.presence_penalty,
        "frequency_penalty": req.frequency_penalty,
        "max_tokens": req.max_tokens,
        "stop": stop_seqs,
        "grammar_type": req.grammar_type,
        "custom_grammar": req.custom_grammar,
    }

    req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    model_name = req.model or engine.model_name

    if req.stream:
        async def stream_gen():
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()

            def worker():
                try:
                    for chunk in engine.generate_text_stream(prompt_text, params):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            threading.Thread(target=worker, daemon=True).start()

            while True:
                item = await queue.get()
                if item is None or isinstance(item, Exception):
                    break
                data = {
                    "id": req_id,
                    "object": "text_completion",
                    "created": created_time,
                    "model": model_name,
                    "choices": [{"text": item, "index": 0, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data)}\n\n"

            yield f"data: {json.dumps({'id': req_id, 'choices': [{'text': '', 'index': 0, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    else:
        full_text = ""
        for chunk in engine.generate_text_stream(prompt_text, params):
            full_text += chunk

        return {
            "id": req_id,
            "object": "text_completion",
            "created": created_time,
            "model": model_name,
            "choices": [{"text": full_text, "index": 0, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(prompt_text.split()),
                "completion_tokens": len(full_text.split()),
                "total_tokens": len(prompt_text.split()) + len(full_text.split()),
            },
        }


@app.post("/v1/generation/stop")
def stop_generation():
    """Requests immediate cancellation of active token generation."""
    engine.stop_generation()
    return {"status": "success", "message": "Stop requested."}


# ----------------------------------------------------
# 5. Chat Sessions Management
# ----------------------------------------------------
@app.get("/v1/sessions")
def list_sessions():
    sessions = session_mgr.list_sessions()
    return {
        "active_id": session_mgr.active_session_id,
        "sessions": [s.to_dict() for s in sessions],
    }


@app.post("/v1/sessions")
def create_session(req: SessionCreateRequest):
    new_s = session_mgr.create_session(title=req.title or "New Conversation")
    if req.system_prompt:
        new_s.system_prompt = req.system_prompt
        session_mgr.save_session(new_s)
    return new_s.to_dict()


@app.get("/v1/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in session_mgr.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session_mgr.sessions[session_id].to_dict()


@app.put("/v1/sessions/{session_id}")
def update_session(session_id: str, req: SessionUpdateRequest):
    if session_id not in session_mgr.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    s = session_mgr.sessions[session_id]
    if req.title is not None:
        s.title = req.title
    if req.system_prompt is not None:
        s.system_prompt = req.system_prompt
    if req.messages is not None:
        s.messages = req.messages
    s.updated_at = time.time()
    session_mgr.save_session(s)
    return s.to_dict()


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str):
    session_mgr.delete_session(session_id)
    return {"status": "success", "active_id": session_mgr.active_session_id}


# ----------------------------------------------------
# 6. Hugging Face Downloader & Store
# ----------------------------------------------------
@app.get("/v1/downloader/presets")
def get_presets():
    return CURATED_MODELS


@app.get("/v1/downloader/search")
def search_models(query: str = Query(..., min_length=2)):
    return search_hf_models(query, limit=15)


@app.get("/v1/downloader/repo_files")
def get_repo_files(repo_id: str = Query(...)):
    """Fetches all .gguf files inside a specific Hugging Face repository."""
    return list_repo_gguf_files(repo_id)


api_server_enabled = True


@app.get("/v1/server/status")
def get_server_status():
    return {
        "enabled": api_server_enabled,
        "port": 8008,
        "model": engine.model_name if engine.is_loaded else None,
        "turboquant": engine.config.get("turbo_enabled", True),
    }


@app.post("/v1/server/toggle")
def toggle_server_status():
    global api_server_enabled
    api_server_enabled = not api_server_enabled
    return {"enabled": api_server_enabled}


@app.post("/v1/downloader/start")
def start_download(req: DownloadStartRequest):
    key = f"{req.repo_id}/{req.filename}"
    if key in active_downloads and active_downloads[key].isRunning():
        return {"status": "already_running", "key": key}

    download_progress_state[key] = {
        "status": "downloading",
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "speed_mb_s": 0.0,
        "eta_s": 0.0,
        "percent": 0.0,
        "file_path": None,
        "error": None,
    }

    def on_prog(d, t, sp, eta):
        pct = (d / t * 100.0) if t > 0 else 0.0
        download_progress_state[key].update({
            "status": "downloading",
            "downloaded_bytes": d,
            "total_bytes": t,
            "speed_mb_s": round(sp, 2),
            "eta_s": round(eta, 1),
            "percent": round(pct, 1),
        })

    def on_fin(dest):
        download_progress_state[key].update({
            "status": "completed",
            "percent": 100.0,
            "file_path": dest,
        })
        global scanned_models_cache
        scanned_models_cache = scan_for_gguf_models()

    def on_err(err):
        download_progress_state[key].update({
            "status": "error",
            "error": str(err),
        })

    worker = DownloadWorker(
        req.repo_id,
        req.filename,
        output_dir="models",
        on_progress=on_prog,
        on_finished=on_fin,
        on_error=on_err,
    )
    active_downloads[key] = worker
    worker.start()

    return {"status": "started", "key": key}


@app.get("/v1/downloader/progress")
def get_download_progress(key: str = Query(...)):
    if key not in download_progress_state:
        return {"status": "not_found"}
    return download_progress_state[key]


@app.post("/v1/downloader/pause")
def pause_download(req: DownloadStartRequest):
    key = f"{req.repo_id}/{req.filename}"
    if key in active_downloads and active_downloads[key].isRunning():
        active_downloads[key].pause()
        if key in download_progress_state:
            download_progress_state[key]["status"] = "paused"
        return {"status": "paused", "key": key}
    return {"status": "not_active", "key": key}


@app.post("/v1/downloader/resume")
def resume_download(req: DownloadStartRequest):
    key = f"{req.repo_id}/{req.filename}"
    if key in active_downloads and active_downloads[key].isRunning():
        active_downloads[key].resume()
        if key in download_progress_state:
            download_progress_state[key]["status"] = "downloading"
        return {"status": "resumed", "key": key}
    return start_download(req)


@app.post("/v1/downloader/stop")
def stop_download(req: DownloadStartRequest):
    key = f"{req.repo_id}/{req.filename}"
    if key in active_downloads:
        active_downloads[key].stop()
        if key in download_progress_state:
            download_progress_state[key]["status"] = "stopped"
            download_progress_state[key]["error"] = "Download stopped by user."
        return {"status": "stopped", "key": key}
    return {"status": "not_active", "key": key}


@app.get("/v1/project/contributors")
def get_project_contributors():
    """
    Returns dynamic repository contributors extracted from git log,
    preserving the immutable Primary Contributor / Original Creator.
    """
    primary_contributor = {
        "name": "Protik Das",
        "role": "Primary Contributor & Lead Architect",
        "is_primary": True,
        "badge": "⭐ Original Creator",
    }

    git_contributors = []
    try:
        proc = subprocess.run(
            ["git", "log", "--format=%aN|%aE"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            author_counts = {}
            for line in proc.stdout.strip().splitlines():
                line = line.strip()
                if not line or "|" not in line:
                    continue
                name, email = line.split("|", 1)
                name = name.strip()
                email = email.strip()
                key = f"{name}|{email}"
                author_counts[key] = author_counts.get(key, 0) + 1

            for key, count in author_counts.items():
                name, email = key.split("|", 1)
                is_primary = name.lower() in ["protik das", "protik", "author"]
                git_contributors.append({
                    "name": name,
                    "email": email,
                    "commits": count,
                    "is_primary": is_primary,
                    "badge": "⭐ Lead" if is_primary else "⚡ Contributor",
                })
    except Exception:
        pass

    has_primary = any(c.get("is_primary") for c in git_contributors)
    if not has_primary:
        git_contributors.insert(0, primary_contributor)

    return {
        "primary": primary_contributor,
        "contributors": git_contributors,
    }


# ----------------------------------------------------
# Main Entrypoint
# ----------------------------------------------------
def start_server(port: int = 8008):
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8008))
    print(f"[Server] Starting Llama.cpp Turbo Desktop Backend on http://127.0.0.1:{port}")
    start_server(port)
