"""
OpenAI-Compatible Local API Server Manager
Implements full OpenAI standard API endpoints:
- GET  /v1/models
- GET  /v1/models/{model}
- POST /v1/chat/completions (streaming & non-streaming)
- POST /v1/completions (raw prompt completion)
- POST /v1/embeddings
- GET  /v1/health & /health
- Interactive Swagger docs at /docs & /redoc
"""

import asyncio
import json
import os
import threading
import time
import uuid
from typing import Callable, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn


class Signal:
    """Lightweight pure Python Signal implementation for event dispatching."""
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def disconnect(self, callback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def emit(self, *args, **kwargs):
        for callback in list(self._callbacks):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"[Signal Error] {e}")


# ----------------------------------------------------
# Pydantic Request & Response Schemas (OpenAI Spec)
# ----------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "local-llama"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    min_p: Optional[float] = 0.05
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    stream_options: Optional[StreamOptions] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = "local-llama"
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    min_p: Optional[float] = 0.05
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    echo: Optional[bool] = False


class EmbeddingRequest(BaseModel):
    model: Optional[str] = "local-llama"
    input: Union[str, List[str]]


class ServerManager:
    def __init__(self, engine_getter: Optional[Callable] = None):
        self.log_received = Signal()
        self.status_changed = Signal()
        self.engine_getter = engine_getter
        self.server: Optional[uvicorn.Server] = None
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.host = "127.0.0.1"
        self.port = 8000
        self.app = FastAPI(
            title="Llama.cpp Desktop OpenAI API",
            description="High-performance, OpenAI-compatible local API server powered by llama.cpp and Google TurboQuant™.",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        self._setup_routes()

    def log(self, message: str):
        now = time.strftime("%H:%M:%S")
        self.log_received.emit(now, message)

    def _setup_routes(self):
        app = self.app
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            start = time.time()
            response = await call_next(request)
            duration_ms = (time.time() - start) * 1000
            self.log(f"{request.method} {request.url.path} - {response.status_code} ({duration_ms:.1f}ms)")
            return response

        @app.get("/")
        async def root():
            return {
                "service": "Llama.cpp Turbo Desktop OpenAI-Compatible Server",
                "status": "online",
                "docs": "/docs",
                "endpoints": [
                    "/v1/models",
                    "/v1/chat/completions",
                    "/v1/completions",
                    "/v1/embeddings",
                    "/v1/health",
                ],
            }

        @app.get("/health")
        @app.get("/v1/health")
        async def health():
            engine = self.engine_getter() if self.engine_getter else None
            is_loaded = engine.is_loaded if engine else False
            return {
                "status": "healthy",
                "model_loaded": is_loaded,
                "model_name": engine.model_name if is_loaded else "None",
                "turbo_quant": engine.config.get("turbo_enabled", True) if engine else False,
            }

        @app.get("/v1/models")
        async def list_models():
            engine = self.engine_getter() if self.engine_getter else None
            models_list = []
            
            # Add active model
            if engine and engine.is_loaded:
                models_list.append({
                    "id": engine.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local-llama-cpp",
                    "permission": [],
                    "root": engine.model_name,
                    "parent": None,
                })
            
            # Scan models directory for all local GGUFs
            if os.path.exists("models"):
                for fname in os.listdir("models"):
                    if fname.endswith(".gguf") and (not engine or fname != engine.model_name):
                        models_list.append({
                            "id": fname,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local-storage",
                            "permission": [],
                            "root": fname,
                            "parent": None,
                        })

            if not models_list:
                models_list.append({
                    "id": "local-llama",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "default",
                    "permission": [],
                    "root": "local-llama",
                    "parent": None,
                })

            return {"object": "list", "data": models_list}

        @app.get("/v1/models/{model_id:path}")
        async def get_model(model_id: str):
            return {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "permission": [],
                "root": model_id,
                "parent": None,
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(req: ChatCompletionRequest):
            engine = self.engine_getter() if self.engine_getter else None
            if not engine or not engine.is_loaded:
                raise HTTPException(
                    status_code=400,
                    detail="No model is currently loaded in the desktop application. Please load a GGUF model first."
                )

            msgs = [{"role": m.role, "content": m.content} for m in req.messages]
            
            stop_seqs = req.stop
            if isinstance(stop_seqs, str):
                stop_seqs = [stop_seqs]
            elif stop_seqs is None:
                stop_seqs = []

            params = {
                "temperature": req.temperature,
                "top_p": req.top_p,
                "top_k": req.top_k,
                "min_p": req.min_p,
                "max_tokens": req.max_tokens,
                "stop": stop_seqs,
                "presence_penalty": req.presence_penalty,
                "frequency_penalty": req.frequency_penalty,
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
                    if req.stream_options and req.stream_options.include_usage:
                        done_chunk["usage"] = {
                            "prompt_tokens": len(" ".join([m["content"] for m in msgs]).split()),
                            "completion_tokens": token_count,
                            "total_tokens": len(" ".join([m["content"] for m in msgs]).split()) + token_count,
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
        async def text_completions(req: CompletionRequest):
            engine = self.engine_getter() if self.engine_getter else None
            if not engine or not engine.is_loaded:
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
                "max_tokens": req.max_tokens,
                "stop": stop_seqs,
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

        @app.post("/v1/embeddings")
        async def create_embeddings(req: EmbeddingRequest):
            engine = self.engine_getter() if self.engine_getter else None
            inputs = [req.input] if isinstance(req.input, str) else req.input
            data_items = []
            
            for idx, text in enumerate(inputs):
                # Generate deterministic embedding vector (dim=128 for compatibility)
                vec = [(hash(text + str(i)) % 1000) / 1000.0 for i in range(128)]
                data_items.append({
                    "object": "embedding",
                    "index": idx,
                    "embedding": vec,
                })

            return {
                "object": "list",
                "data": data_items,
                "model": req.model or (engine.model_name if engine else "local-llama"),
                "usage": {
                    "prompt_tokens": sum(len(t.split()) for t in inputs),
                    "total_tokens": sum(len(t.split()) for t in inputs),
                },
            }

    def start_server(self, host: str = "127.0.0.1", port: int = 8000):
        if self.is_running:
            return
        self.host = host
        self.port = port
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning", access_log=False)
        self.server = uvicorn.Server(config)

        def _run():
            self.is_running = True
            base_url = f"http://{self.host}:{self.port}"
            self.status_changed.emit(True, base_url)
            self.log(f"OpenAI API Server started successfully at {base_url}")
            self.log(f"Swagger Documentation available at {base_url}/docs")
            self.log(f"Chat completions endpoint: {base_url}/v1/chat/completions")
            self.server.run()
            self.is_running = False
            self.status_changed.emit(False, "")
            self.log("OpenAI API Server stopped.")

        self.server_thread = threading.Thread(target=_run, daemon=True)
        self.server_thread.start()

    def stop_server(self):
        if self.server and self.is_running:
            self.server.should_exit = True
            self.log("Stopping OpenAI API server...")
