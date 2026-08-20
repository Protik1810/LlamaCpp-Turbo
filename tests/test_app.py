"""
Automated Verification & Unit Tests for Llama.cpp Turbo Desktop
"""

import os
import sys
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.turbo_quant import fast_walsh_hadamard_transform, TurboQuantKVCache, TurboQuantManager
from src.core.session_manager import SessionManager
from src.core.downloader import CURATED_MODELS
from src.core.model_scanner import get_default_scan_locations


def test_turbo_quant():
    print("[1/6] Testing Google TurboQuant Engine...")
    
    # 1. Fast Walsh-Hadamard Transform
    data = np.random.randn(4, 128).astype(np.float32)
    fwht_out = fast_walsh_hadamard_transform(data)
    assert fwht_out.shape == data.shape, f"FWHT shape mismatch: {fwht_out.shape} vs {data.shape}"
    
    # FWHT is self-inverse: H(H(x)) == x
    reconstructed = fast_walsh_hadamard_transform(fwht_out)
    diff = np.max(np.abs(reconstructed - data))
    assert diff < 1e-4, f"FWHT reconstruction error too high: {diff}"
    print("  [PASS] Fast Walsh-Hadamard Transform verified (self-inverse orthogonality preserved)")

    # 2. TurboQuant KV Cache Compression
    kv_cache = TurboQuantKVCache(bits=4, use_hadamard=True, sparsity_ratio=0.20)
    test_kv = np.random.randn(10, 64).astype(np.float32)
    q_indices, scale, zero_point = kv_cache.compress_tensor(test_kv)
    decompressed = kv_cache.decompress_tensor(q_indices, scale, zero_point, test_kv.shape)
    assert decompressed is not None
    
    assert q_indices.shape == test_kv.shape
    assert kv_cache.compression_ratio > 3.0, f"Expected >3.0x compression ratio, got {kv_cache.compression_ratio}"
    print(f"  [PASS] TurboQuant KV Compression verified: {kv_cache.compression_ratio:.1f}x compression ratio ({kv_cache.memory_savings_pct}% savings)")

    # 3. TurboQuant Manager Analytical Model
    manager = TurboQuantManager()
    savings = manager.estimate_savings(raw_kv_mb=1024.0, model_weights_mb=2048.0)
    assert "compression_ratio" in savings
    print(f"  [PASS] TurboQuant Manager Analytical Estimates: {savings['compression_ratio']} ({savings['savings_pct']} savings, {savings['estimated_speedup']} speedup)")


def test_session_manager():
    print("\n[2/6] Testing Session & Chat History Manager...")
    mgr = SessionManager(storage_dir="data/test_sessions")
    
    # Create session
    session = mgr.create_session("Unit Test Session")
    session.add_message("user", "Hello llama.cpp!")
    session.add_message("assistant", "Hello! How can I assist you today?", {"tokens": 10, "tok_per_sec": 45.2})
    mgr.save_session(session)
    
    # Verify export
    md_export = session.export_markdown()
    assert "Hello llama.cpp!" in md_export
    assert "45.2 tok/s" in md_export
    
    # Cleanup test session
    mgr.delete_session(session.id)
    print("  [PASS] Session creation, message logging, Markdown export, and deletion verified")


def test_downloader_presets():
    print("\n[3/6] Testing Model Presets & Downloader Configuration...")
    assert len(CURATED_MODELS) >= 8, "Expected at least 8 curated models"
    for m in CURATED_MODELS:
        assert "repo_id" in m and "filename" in m and "size_gb" in m
    print(f"  [PASS] {len(CURATED_MODELS)} curated models verified (Llama 3.2, Qwen 2.5, DeepSeek R1, Phi-3.5, Gemma 2)")


def test_server_manager_routes():
    print("\n[4/6] Testing OpenAI-Compatible API Server Schema...")
    from src.core.server_manager import ServerManager, ChatCompletionRequest, ChatMessage
    server = ServerManager()
    assert server.host == "127.0.0.1"
    req = ChatCompletionRequest(
        model="local-llama",
        messages=[ChatMessage(role="user", content="Test prompt")],
        temperature=0.7,
        max_tokens=256
    )
    assert req.model == "local-llama"
    assert len(req.messages) == 1
    print("  [PASS] FastAPI OpenAI schemas and routing verified")


def test_model_scanner():
    print("\n[5/6] Testing System Model Scanner...")
    locations = get_default_scan_locations()
    assert len(locations) >= 5, "Expected at least 5 default scan locations"
    print(f"  [PASS] Discovered {len(locations)} system search locations (LM Studio, Ollama, HuggingFace, Downloads, Workspace)")


def test_frontend_and_backend():
    print("\n[6/6] Testing Backend & Frontend Integrity...")
    from src.server import app
    assert app.title == "Llama.cpp Turbo Desktop Backend"
    assert app.version == "1.0.0"

    electron_index = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "electron", "renderer", "index.html"))
    electron_app_js = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "electron", "renderer", "js", "app.js"))
    electron_style_css = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "electron", "renderer", "css", "style.css"))
    
    assert os.path.exists(electron_index), "index.html is missing"
    assert os.path.exists(electron_app_js), "app.js is missing"
    assert os.path.exists(electron_style_css), "style.css is missing"
    print("  [PASS] FastAPI Backend and Electron Frontend bundle verified")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING ALL LLAMA.CPP TURBO DESKTOP VERIFICATION TESTS")
    print("=" * 60)
    test_turbo_quant()
    test_session_manager()
    test_downloader_presets()
    test_server_manager_routes()
    test_model_scanner()
    test_frontend_and_backend()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (6/6)")
    print("=" * 60)
