"""
Integration test suite for the Electron Python Backend Server (FastAPI + TurboQuant + Hugging Face)
"""

import os
import sys
import time
import requests

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.server import app
import threading
import uvicorn


def run_electron_backend_tests():
    print("============================================================")
    print("RUNNING ELECTRON PYTHON BACKEND & HUGGING FACE TEST SUITE")
    print("============================================================")

    PORT = 8011
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="error"),
        daemon=True
    )
    server_thread.start()
    time.sleep(1.5)

    base_url = f"http://127.0.0.1:{PORT}"

    # 1. Health Endpoint
    print("[1/6] Testing Health & System Diagnostics...")
    r = requests.get(f"{base_url}/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    health = r.json()
    assert health["status"] == "ok"
    assert "system" in health
    print(f"  [PASS] Health check verified: CPU={health['system']['cpu_percent']}%, RAM={health['system']['ram_used_gb']}/{health['system']['ram_total_gb']} GB")

    # 2. Hugging Face Curated Presets & Search
    print("[2/6] Testing Hugging Face Presets & Live Search...")
    r = requests.get(f"{base_url}/v1/downloader/presets")
    assert r.status_code == 200
    presets = r.json()
    assert len(presets) >= 8
    print(f"  [PASS] {len(presets)} curated models loaded successfully (Llama 3.2, Qwen 2.5, DeepSeek R1, etc.)")

    r_search = requests.get(f"{base_url}/v1/downloader/search?query=qwen2.5-coder")
    assert r_search.status_code == 200
    search_res = r_search.json()
    assert len(search_res) > 0
    print(f"  [PASS] Hugging Face live search verified: Found {len(search_res)} repositories matching 'qwen2.5-coder'")

    # 3. Hugging Face Repo GGUF Files Explorer
    print("[3/6] Testing Hugging Face Repo File Explorer...")
    test_repo = presets[0]["repo_id"]
    r_files = requests.get(f"{base_url}/v1/downloader/repo_files?repo_id={test_repo}")
    assert r_files.status_code == 200
    files_list = r_files.json()
    assert len(files_list) > 0
    print(f"  [PASS] Found {len(files_list)} GGUF quantizations in {test_repo} (e.g. {files_list[0]['filename']} - {files_list[0]['size_gb']} GB)")

    # 4. System Model Scanner Endpoint
    print("[4/6] Testing System Model Scanner...")
    r_scan = requests.get(f"{base_url}/v1/models/scan")
    assert r_scan.status_code == 200
    scan_data = r_scan.json()
    print(f"  [PASS] Discovered {scan_data['count']} GGUF models on host system")

    # 5. TurboQuant Configuration Update
    print("[5/6] Testing Google TurboQuant Dynamic Configuration...")
    r_tq = requests.post(f"{base_url}/v1/turboquant/config", json={"enabled": True, "bits": 3, "hadamard": True, "sparsity": 0.25})
    assert r_tq.status_code == 200
    tq_resp = r_tq.json()
    assert tq_resp["turbo_bits"] == 3
    assert tq_resp["compression_ratio"] == "5.3x"
    print(f"  [PASS] TurboQuant config dynamically updated to {tq_resp['compression_ratio']} compression")

    # 6. Session Manager REST API
    print("[6/7] Testing Chat Session REST Endpoints...")
    r_sess = requests.post(f"{base_url}/v1/sessions", json={"title": "Test Electron Session"})
    assert r_sess.status_code == 200
    sess = r_sess.json()
    sess_id = sess["id"]
    print(f"  [PASS] Created test conversation: {sess_id}")

    r_del = requests.delete(f"{base_url}/v1/sessions/{sess_id}")
    assert r_del.status_code == 200
    print("  [PASS] Deleted test conversation cleanly")

    # 7. OpenAI API Server Status & Toggle
    print("[7/7] Testing OpenAI API Server Status & Toggle...")
    r_stat = requests.get(f"{base_url}/v1/server/status")
    assert r_stat.status_code == 200
    assert r_stat.json()["enabled"] is True
    print(f"  [PASS] Server status reported: {r_stat.json()}")

    r_tog = requests.post(f"{base_url}/v1/server/toggle")
    assert r_tog.status_code == 200
    assert r_tog.json()["enabled"] is False

    r_tog_on = requests.post(f"{base_url}/v1/server/toggle")
    assert r_tog_on.status_code == 200
    # 8. Test Download Controls (Pause / Resume / Stop)
    print("[8/8] Testing Download Manager Controls (Pause / Resume / Stop)...")
    dl_payload = {
        "repo_id": "test/model",
        "filename": "test.gguf"
    }
    r_pause = requests.post(f"{base_url}/v1/downloader/pause", json=dl_payload)
    assert r_pause.status_code == 200

    r_stop = requests.post(f"{base_url}/v1/downloader/stop", json=dl_payload)
    assert r_stop.status_code == 200
    print("  [PASS] Download controls verified: pause and stop endpoints responsive")
    # 9. Test Project Contributors Endpoint
    print("[9/9] Testing Project Contributors Endpoint...")
    r_contrib = requests.get(f"{base_url}/v1/project/contributors")
    assert r_contrib.status_code == 200
    c_data = r_contrib.json()
    assert "primary" in c_data
    assert c_data["primary"]["name"] == "Protik Das"
    assert len(c_data["contributors"]) >= 1
    print("  [PASS] Project contributors verified: dynamic registry with immutable primary creator")

    print("============================================================")
    print("ALL ELECTRON PYTHON BACKEND TESTS PASSED SUCCESSFULLY! (9/9)")
    print("============================================================")
    os._exit(0)


if __name__ == "__main__":
    run_electron_backend_tests()
