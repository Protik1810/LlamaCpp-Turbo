"""
HuggingFace GGUF Model Downloader
Provides curated popular model presets, HuggingFace GGUF search, and chunked streaming downloads with progress reporting.
"""

import os
import time
import threading
from typing import Any, Callable, Dict, List, Optional
import requests

CURATED_MODELS = [
    {
        "name": "Llama 3.2 1B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_gb": 0.81,
        "description": "Meta's ultra-compact, ultra-fast 1B model. Ideal for low-RAM systems and quick tasks.",
        "params": "1.23B",
        "context": 131072,
    },
    {
        "name": "Llama 3.2 3B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 2.02,
        "description": "Meta's highly capable 3B model. Outstanding balance of reasoning, speed, and memory.",
        "params": "3.21B",
        "context": 131072,
    },
    {
        "name": "Qwen 2.5 0.5B Instruct (Q4_K_M)",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size_gb": 0.40,
        "description": "Alibaba's featherweight 0.5B model. Instant response times and tiny RAM footprint.",
        "params": "0.49B",
        "context": 32768,
    },
    {
        "name": "Qwen 2.5 1.5B Instruct (Q4_K_M)",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_gb": 1.10,
        "description": "Superb efficiency and coding capabilities in a lightweight 1.5B package.",
        "params": "1.54B",
        "context": 32768,
    },
    {
        "name": "Qwen 2.5 3B Instruct (Q4_K_M)",
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_gb": 2.20,
        "description": "Excellent multilingual and coding capabilities rivaling much larger models.",
        "params": "3.09B",
        "context": 32768,
    },
    {
        "name": "DeepSeek R1 Distill Qwen 1.5B (Q4_K_M)",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "size_gb": 1.12,
        "description": "DeepSeek's state-of-the-art chain-of-thought reasoning distilled into 1.5B.",
        "params": "1.78B",
        "context": 131072,
    },
    {
        "name": "Phi-3.5 Mini Instruct (Q4_K_M)",
        "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size_gb": 2.39,
        "description": "Microsoft's high-reasoning 3.8B model with 128k context and robust logic.",
        "params": "3.82B",
        "context": 131072,
    },
    {
        "name": "Gemma 2 2B IT (Q4_K_M)",
        "repo_id": "bartowski/gemma-2-2b-it-GGUF",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "size_gb": 1.63,
        "description": "Google's lightweight Gemma 2 architecture with strong conversational safety.",
        "params": "2.61B",
        "context": 8192,
    },
]


import threading
from typing import Any, Callable, Dict, List, Optional
import requests


class DownloadWorker(threading.Thread):
    def __init__(
        self,
        repo_id: str,
        filename: str,
        output_dir: str = "models",
        on_progress: Optional[Callable[[int, int, float, float], None]] = None,
        on_finished: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(daemon=True)
        self.repo_id = repo_id
        self.filename = filename
        self.output_dir = output_dir
        self.on_progress_cb = on_progress
        self.on_finished_cb = on_finished
        self.on_error_cb = on_error
        self.is_cancelled = False
        self.is_paused = False
        self._is_running = False

    def isRunning(self) -> bool:
        return self._is_running

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def cancel(self):
        self.is_cancelled = True
        self.is_paused = False

    def stop(self):
        self.cancel()

    def run(self):
        self._is_running = True
        try:
            # Ensure absolute output dir in project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            target_dir = os.path.join(base_dir, self.output_dir) if not os.path.isabs(self.output_dir) else self.output_dir
            os.makedirs(target_dir, exist_ok=True)

            dest_path = os.path.join(target_dir, self.filename)
            temp_path = dest_path + ".part"

            url = f"https://huggingface.co/{self.repo_id}/resolve/main/{self.filename}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) LlamaCppTurboDesktop/1.0"
            }
            downloaded = 0

            # Support resuming from partial file
            if os.path.exists(temp_path):
                downloaded = os.path.getsize(temp_path)
                headers["Range"] = f"bytes={downloaded}-"

            response = requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True)

            if response.status_code == 416:
                if downloaded > 0:
                    os.replace(temp_path, dest_path)
                    if self.on_finished_cb:
                        self.on_finished_cb(dest_path)
                    return
                headers.pop("Range", None)
                downloaded = 0
                response = requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True)

            if response.status_code not in [200, 206]:
                err_msg = f"HTTP {response.status_code}: {response.reason or 'Failed to download from Hugging Face'}"
                if self.on_error_cb:
                    self.on_error_cb(err_msg)
                return

            total_size = int(response.headers.get("content-length", 0)) + downloaded

            start_time = time.time()
            last_time = start_time
            bytes_since_last = 0
            speed_mb = 0.0

            mode = "ab" if downloaded > 0 and response.status_code == 206 else "wb"
            if mode == "wb":
                downloaded = 0

            chunk_size = 1024 * 512  # 512 KB chunks
            with open(temp_path, mode) as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    while self.is_paused and not self.is_cancelled:
                        time.sleep(0.2)

                    if self.is_cancelled:
                        if self.on_error_cb:
                            self.on_error_cb("Download stopped by user.")
                        return

                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        bytes_since_last += len(chunk)

                        curr_time = time.time()
                        time_diff = curr_time - last_time
                        if time_diff >= 0.3:
                            speed_mb = (bytes_since_last / (1024 * 1024)) / time_diff
                            rem_bytes = max(0, total_size - downloaded)
                            eta_s = (rem_bytes / (speed_mb * 1024 * 1024)) if speed_mb > 0.01 else 0
                            if self.on_progress_cb:
                                self.on_progress_cb(downloaded, total_size, speed_mb, eta_s)
                            last_time = curr_time
                            bytes_since_last = 0

            # Final progress emit
            if self.on_progress_cb and total_size > 0:
                self.on_progress_cb(downloaded, total_size, speed_mb, 0.0)

            # Move .part to final .gguf
            if os.path.exists(temp_path):
                os.replace(temp_path, dest_path)
                if self.on_finished_cb:
                    self.on_finished_cb(dest_path)
            else:
                if self.on_error_cb:
                    self.on_error_cb("File missing after download completion.")
        except Exception as e:
            if self.on_error_cb:
                self.on_error_cb(f"Download failed: {str(e)}")
        finally:
            self._is_running = False


def search_hf_models(query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Searches Hugging Face for GGUF models matching query."""
    try:
        url = f"https://huggingface.co/api/models?search={query}&filter=gguf&limit={limit}&sort=downloads&direction=-1"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            results = []
            for m in res.json():
                repo_id = m.get("id", "")
                author = repo_id.split("/")[0] if "/" in repo_id else repo_id
                results.append({
                    "id": repo_id,
                    "author": author,
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "last_modified": m.get("lastModified", ""),
                    "tags": m.get("tags", []),
                })
            return results
    except Exception as e:
        print(f"HF Search error: {e}")
    return []


def list_repo_gguf_files(repo_id: str) -> List[Dict[str, Any]]:
    """Lists all available .gguf model files and sizes inside a Hugging Face repository."""
    try:
        url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            files = []
            for item in res.json():
                path = item.get("path", "")
                if path.lower().endswith(".gguf") and not path.lower().startswith("mmproj-"):
                    size_bytes = item.get("size", 0)
                    size_gb = round(size_bytes / (1024**3), 2)
                    files.append({
                        "filename": path,
                        "size_bytes": size_bytes,
                        "size_gb": size_gb,
                        "quant": path.replace(".gguf", "").split("-")[-1].upper(),
                    })
            return sorted(files, key=lambda x: x["size_bytes"])
    except Exception as e:
        print(f"Error listing repo files for {repo_id}: {e}")
    return []
