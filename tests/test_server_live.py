"""
Live OpenAI-Compatible API Server Integration Test
"""

import os
import sys
import time
import requests

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.llama_engine import LlamaEngine
from src.core.server_manager import ServerManager


def main():
    print("Testing live OpenAI-compatible API server on port 8009...")
    engine = LlamaEngine()
    # Put engine in ready state
    engine.is_loaded = True
    engine.model_name = "test-llama-3.2-1b.gguf"

    server_mgr = ServerManager(engine_getter=lambda: engine)
    server_mgr.start_server(host="127.0.0.1", port=8009)

    # Wait for server to boot
    time.sleep(1.5)

    base_url = "http://127.0.0.1:8009"
    try:
        # 1. Healthcheck
        r_health = requests.get(f"{base_url}/health", timeout=5)
        assert r_health.status_code == 200, f"Health check failed: {r_health.text}"
        print("  [PASS] GET /health:", r_health.json())

        # 2. List Models (/v1/models)
        r_models = requests.get(f"{base_url}/v1/models", timeout=5)
        assert r_models.status_code == 200, f"Models list failed: {r_models.text}"
        models_data = r_models.json()
        print("  [PASS] GET /v1/models:", [m["id"] for m in models_data["data"]])

        # 3. Chat Completion (/v1/chat/completions non-streaming)
        r_chat = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "test-llama-3.2-1b.gguf",
                "messages": [{"role": "user", "content": "Hello Llama.cpp!"}],
                "temperature": 0.7,
                "stream": False
            },
            timeout=10
        )
        assert r_chat.status_code == 200, f"Chat completion failed: {r_chat.text}"
        chat_resp = r_chat.json()
        print("  [PASS] POST /v1/chat/completions (Non-streaming):")
        print("         Assistant Content:", chat_resp["choices"][0]["message"]["content"][:60], "...")

        # 4. Text Completion (/v1/completions)
        r_compl = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "test-llama-3.2-1b.gguf",
                "prompt": "def add(a, b):",
                "max_tokens": 64,
                "stream": False
            },
            timeout=10
        )
        assert r_compl.status_code == 200, f"Completions failed: {r_compl.text}"
        print("  [PASS] POST /v1/completions:", r_compl.json()["choices"][0]["text"][:60], "...")

        print("\nAll OpenAI API server endpoints tested and verified!")
    finally:
        server_mgr.stop_server()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
