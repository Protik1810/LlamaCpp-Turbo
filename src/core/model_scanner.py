"""
System Model Scanner Module
Performs fast, multi-threaded scanning across user directories, external drives, and popular LLM tool paths
(LM Studio, HuggingFace Hub, Ollama manifests & blobs, GPT4All, Downloads, Documents) to discover and validate all GGUF models.
"""

import json
import os
import struct
import sys
import threading
from typing import Any, Dict, List, Optional

from .gguf_reader import GGUF_MAGIC, GGUFInspector


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


def get_ollama_manifest_map() -> Dict[str, str]:
    """
    Parses Ollama manifest files to map sha256 blob filenames to clean model names (e.g. 'gemma4:12b', 'qwen2.5-coder:7b').
    """
    home = os.path.expanduser("~")
    manifests_root = os.path.join(home, ".ollama", "models", "manifests")
    mapping = {}

    if not os.path.exists(manifests_root):
        return mapping

    for root, _, files in os.walk(manifests_root):
        for f in files:
            manifest_file = os.path.join(root, f)
            try:
                with open(manifest_file, "r", encoding="utf-8", errors="ignore") as mf:
                    data = json.load(mf)
                    layers = data.get("layers", [])
                    # Compute model tag from relative path
                    rel_path = os.path.relpath(manifest_file, manifests_root)
                    parts = rel_path.replace("\\", "/").split("/")
                    # e.g. registry.ollama.ai/library/qwen2.5/3b-instruct -> qwen2.5:3b-instruct
                    if len(parts) >= 3:
                        model_name = f"{parts[-2]}:{parts[-1]}"
                    else:
                        model_name = parts[-1]

                    for layer in layers:
                        if layer.get("mediaType") == "application/vnd.ollama.image.model":
                            digest = layer.get("digest", "")
                            if digest.startswith("sha256:"):
                                blob_name = "sha256-" + digest[7:]
                                mapping[blob_name] = model_name
            except Exception:
                pass
    return mapping


def get_all_system_drives() -> List[str]:
    """Dynamically enumerates all available logical drives on Windows, or root on POSIX."""
    drives = []
    if sys.platform == "win32":
        try:
            import ctypes
            import string
            bitmask = ctypes.windll.kernel32.GetLogicalDrives()
            for letter in string.ascii_uppercase:
                if bitmask & 1:
                    drive_str = f"{letter}:\\"
                    if os.path.exists(drive_str):
                        drives.append(drive_str)
                bitmask >>= 1
        except Exception:
            for letter in ["C:\\", "D:\\", "E:\\", "F:\\", "G:\\", "H:\\", "Z:\\"]:
                if os.path.exists(letter):
                    drives.append(letter)
    else:
        drives.append("/")
    return drives


def get_default_scan_locations(deep: bool = False) -> List[Dict[str, Any]]:
    """Returns comprehensive system paths where LLM tools and users store GGUF models."""
    home = os.path.expanduser("~")
    locations = [
        {"name": "Local Workspace Models", "path": os.path.abspath("models")},
        {"name": "Current Directory Models", "path": os.path.join(os.getcwd(), "models")},
    ]

    # Application installation directories
    exe_dir = os.path.dirname(sys.executable)
    locations.append({"name": "Executable Models", "path": os.path.join(exe_dir, "models")})
    parent_app_dir = os.path.abspath(os.path.join(exe_dir, "..", ".."))
    locations.append({"name": "App Root Models", "path": os.path.join(parent_app_dir, "models")})

    # Environment variables
    for env_var, name in [
        ("OLLAMA_MODELS", "Custom Ollama Models Path"),
        ("HF_HOME", "Custom HF Home"),
        ("HUGGINGFACE_HUB_CACHE", "Custom HF Hub Cache"),
        ("MODELS_PATH", "Custom Models Path"),
        ("LLAMA_MODELS_PATH", "Custom Llama Models Path"),
    ]:
        val = os.environ.get(env_var)
        if val and os.path.exists(val):
            locations.append({"name": name, "path": os.path.abspath(val)})

    # Standard User & Application directories
    locations.extend([
        {"name": "LM Studio Models", "path": os.path.join(home, ".lmstudio", "models")},
        {"name": "LM Studio Bundled", "path": os.path.join(home, ".lmstudio", ".internal", "bundled-models")},
        {"name": "LM Studio Cache", "path": os.path.join(home, ".cache", "lm-studio", "models")},
        {"name": "LM Studio AppData", "path": os.path.join(home, "AppData", "Local", "Programs", "LM Studio")},
        {"name": "Ollama Models (Blobs)", "path": os.path.join(home, ".ollama", "models", "blobs")},
        {"name": "Ollama AppData", "path": os.path.join(home, "AppData", "Local", "Ollama", "models", "blobs")},
        {"name": "Hugging Face Hub Cache", "path": os.path.join(home, ".cache", "huggingface", "hub")},
        {"name": "GPT4All Models", "path": os.path.join(home, "AppData", "Local", "nomic.ai", "GPT4All")},
        {"name": "Jan.ai Models", "path": os.path.join(home, "jan", "models")},
        {"name": "Jan.ai AppData", "path": os.path.join(home, "AppData", "Roaming", "jan", "models")},
        {"name": "User Downloads", "path": os.path.join(home, "Downloads")},
        {"name": "User Documents", "path": os.path.join(home, "Documents")},
        {"name": "User Desktop", "path": os.path.join(home, "Desktop")},
        {"name": "User Home Models", "path": os.path.join(home, "models")},
        {"name": "User Home GGUF", "path": os.path.join(home, "gguf")},
        {"name": "User Home LLM", "path": os.path.join(home, "llm")},
        {"name": "User Home AI", "path": os.path.join(home, "ai")},
    ])

    # All Logical Drive Roots and common AI folders across all drives
    common_subdirs = [
        "models", "Models", "MODELS",
        "gguf", "GGUF", "Gguf",
        "llm", "LLM", "llms", "LLMs",
        "ai", "AI", "ai_models", "AI_Models",
        "Downloads", "downloads",
        "text-generation-webui/models",
        "oobabooga/models",
        "koboldcpp",
        "LM-Studio/models", "LMStudio/models", "LM Studio/models",
        "ollama/models/blobs",
        "jan/models",
        "gpt4all",
    ]

    drives = get_all_system_drives()
    for drive in drives:
        # Check standard user mirror on other drives (e.g. D:\Users\...)
        drive_user = os.path.join(drive, "Users", os.path.basename(home))
        if os.path.exists(drive_user):
            locations.append({"name": f"{drive[0]}: User Downloads", "path": os.path.join(drive_user, "Downloads")})
            locations.append({"name": f"{drive[0]}: User LM Studio", "path": os.path.join(drive_user, ".lmstudio", "models")})
            locations.append({"name": f"{drive[0]}: User Ollama", "path": os.path.join(drive_user, ".ollama", "models", "blobs")})
            locations.append({"name": f"{drive[0]}: User Models", "path": os.path.join(drive_user, "models")})

        # Check common LLM folders directly on drive root
        for sub in common_subdirs:
            p = os.path.join(drive, sub)
            if os.path.exists(p):
                locations.append({"name": f"Drive ({drive[0]}:) {sub}", "path": p})

        if deep:
            locations.append({"name": f"Drive ({drive[0]}:) Root", "path": drive})

    # Filter and deduplicate
    existing = []
    seen = set()
    for loc in locations:
        try:
            p = os.path.abspath(loc["path"])
            if p not in seen and os.path.exists(p):
                seen.add(p)
                existing.append({"name": loc["name"], "path": p, "exists": True})
        except Exception:
            continue
    return existing


def is_valid_gguf_file(file_path: str) -> bool:
    """Quickly checks if a file begins with valid GGUF magic bytes without full parsing."""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 1024 * 1024:  # At least 1MB
            return False
        with open(file_path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            return magic == GGUF_MAGIC
    except Exception:
        return False


class ModelScannerWorker(threading.Thread):
    def __init__(self, target_paths: List[Dict[str, Any]], max_depth: int = 10):
        super().__init__(daemon=True)
        self.file_scanned = Signal()
        self.model_found = Signal()
        self.progress_updated = Signal()
        self.scan_finished = Signal()
        self.target_paths = target_paths
        self.max_depth = max_depth
        self.is_cancelled = False
        self.found_models: List[Dict[str, Any]] = []
        self.ollama_map: Dict[str, str] = {}

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        self.found_models.clear()
        try:
            self.ollama_map = get_ollama_manifest_map()
        except Exception:
            self.ollama_map = {}

        scanned_count = 0
        found_count = 0
        seen_paths = set()

        model_extensions = {".gguf", ".bin"}

        for target in self.target_paths:
            if self.is_cancelled:
                break
            try:
                base_path = target.get("path") if isinstance(target, dict) else str(target)
                if not base_path or not os.path.exists(base_path):
                    continue

                base_depth = base_path.rstrip(os.path.sep).count(os.path.sep)

                for root, dirs, files in os.walk(base_path, topdown=True, onerror=lambda err: None):
                    if self.is_cancelled:
                        break

                    current_depth = root.rstrip(os.path.sep).count(os.path.sep) - base_depth
                    if current_depth >= self.max_depth:
                        dirs.clear()
                        continue

                    # Skip system folders
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() not in {"node_modules", "windows", "$recycle.bin", "system volume information", "program files", "program files (x86)"}]

                    self.file_scanned.emit(root)

                    for f in files:
                        if self.is_cancelled:
                            break

                        scanned_count += 1
                        full_path = os.path.join(root, f)

                        if full_path in seen_paths:
                            continue

                        ext = os.path.splitext(f)[1].lower()

                        # Check if GGUF or Ollama blob
                        if ext in model_extensions or ext == "" or f.startswith("sha256-"):
                            if is_valid_gguf_file(full_path):
                                try:
                                    inspector = GGUFInspector(full_path)
                                    if inspector.is_valid:
                                        # Exclude mmproj vision projectors from primary LLM selector
                                        is_mmproj = f.startswith("mmproj-") or inspector.architecture == "clip" or inspector.metadata.get("general.type") == "mmproj"
                                        if is_mmproj:
                                            continue

                                        seen_paths.add(full_path)
                                        found_count += 1
                                        source = self._detect_source(full_path)

                                        # Check friendly name from Ollama manifests or GGUF metadata
                                        display_name = None
                                        if f in self.ollama_map:
                                            display_name = f"{self.ollama_map[f]} [Ollama]"
                                        elif inspector.model_name and not inspector.model_name.startswith("sha256-"):
                                            display_name = inspector.model_name
                                        elif f.startswith("sha256-"):
                                            arch = inspector.architecture if inspector.architecture != "unknown" else "LLM"
                                            params = f" {inspector.parameter_count}" if inspector.parameter_count != "N/A" else ""
                                            display_name = f"Ollama {arch.upper()}{params} ({f[7:17]}...)"
                                        else:
                                            display_name = clean_display_name(f)

                                        model_info = {
                                            "file_name": display_name,
                                            "file_path": full_path,
                                            "file_size_gb": round(inspector.file_size / (1024**3), 2),
                                            "file_size_mb": round(inspector.file_size / (1024**2), 0),
                                            "architecture": inspector.architecture,
                                            "quantization": inspector.quantization_type,
                                            "parameter_count": inspector.parameter_count,
                                            "context_length": inspector.context_length,
                                            "source": source,
                                        }
                                        self.found_models.append(model_info)
                                        self.model_found.emit(model_info)
                                except Exception as e:
                                    print(f"Error inspecting model {full_path}: {e}")

                        if scanned_count % 20 == 0:
                            self.progress_updated.emit(scanned_count, found_count)
            except Exception as e:
                print(f"[Scanner Warning] Skipping path due to error: {e}")

        self.progress_updated.emit(scanned_count, found_count)
        self.scan_finished.emit(self.found_models)

    def _detect_source(self, path: str) -> str:
        p_lower = path.lower()
        if ".lmstudio" in p_lower or "lm-studio" in p_lower:
            return "LM Studio"
        elif "huggingface" in p_lower:
            return "Hugging Face"
        elif ".ollama" in p_lower:
            return "Ollama"
        elif "gpt4all" in p_lower:
            return "GPT4All"
        elif "jan" in p_lower:
            return "Jan.ai"
        elif "downloads" in p_lower:
            return "Downloads"
        elif "models" in p_lower:
            return "Workspace"
        return "System Drive"


def clean_display_name(name: str) -> str:
    """Formats raw model filenames into clean human-readable titles."""
    base = os.path.basename(name)
    if base.startswith("sha256-"):
        return f"Ollama Model ({base[7:17]}...)"
    for ext in [".gguf", ".bin"]:
        if base.lower().endswith(ext):
            base = base[:-len(ext)]
    return base.replace("_", " ").title()


def scan_for_gguf_models(custom_path: Optional[str] = None, deep: bool = False) -> List[Dict[str, Any]]:
    """Synchronously scans system locations or a custom path and returns valid GGUF models."""
    if custom_path and os.path.exists(custom_path):
        locations = [{"name": f"Custom: {os.path.basename(custom_path) or custom_path}", "path": os.path.abspath(custom_path)}]
        worker = ModelScannerWorker(locations, max_depth=10)
    else:
        locations = get_default_scan_locations(deep=deep)
        max_depth = 4 if deep else 8
        worker = ModelScannerWorker(locations, max_depth=max_depth)

    worker.run()
    formatted = []
    seen = set()
    for m in worker.found_models:
        fpath = m.get("file_path", "")
        if fpath and fpath not in seen:
            seen.add(fpath)
            formatted.append({
                "name": m.get("file_name", clean_display_name(fpath)),
                "path": fpath,
                "size_gb": m.get("file_size_gb", 0),
                "architecture": m.get("architecture", "GGUF"),
                "quantization": m.get("quantization", "Q4_K_M"),
                "context_length": m.get("context_length", 4096),
                "source": m.get("source", "System Drive"),
            })
    return formatted
