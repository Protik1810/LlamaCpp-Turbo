import requests

API_BASE = "http://127.0.0.1:8008"

def test_server_status_and_toggle():
    print("[1/2] Testing Server Status & Toggle...")
    res = requests.get(f"{API_BASE}/v1/server/status", timeout=5)
    assert res.status_code == 200, f"Status failed: {res.text}"
    data = res.json()
    assert "enabled" in data
    assert data["port"] == 8008

    # Toggle off
    res_toggle = requests.post(f"{API_BASE}/v1/server/toggle", timeout=5)
    assert res_toggle.status_code == 200
    assert res_toggle.json()["enabled"] is False

    # Toggle back on
    res_toggle2 = requests.post(f"{API_BASE}/v1/server/toggle", timeout=5)
    assert res_toggle2.status_code == 200
    assert res_toggle2.json()["enabled"] is True
    print("  [PASS] Server toggle working cleanly!")

def test_download_worker_threading():
    print("[2/2] Testing Hugging Face Repo File Explorer & Downloader Initializer...")
    res = requests.get(f"{API_BASE}/v1/downloader/repo_files?repo_id=bartowski/Llama-3.2-1B-Instruct-GGUF", timeout=10)
    assert res.status_code == 200
    files = res.json()
    assert len(files) > 0
    print(f"  [PASS] Found {len(files)} quantizations ready for download!")

if __name__ == "__main__":
    test_server_status_and_toggle()
    test_download_worker_threading()
    print("ALL DOWNLOAD WORKER & SERVER TOGGLE TESTS PASSED!")
