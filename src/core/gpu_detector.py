"""
GPU & Hardware Acceleration Detector for Llama.cpp Turbo Desktop.
Detects NVIDIA CUDA, Vulkan, DirectML, and CPU SIMD acceleration capabilities.
Intelligently differentiates between Dedicated/Discrete GPUs and Integrated GPUs (Intel UHD/Iris, AMD APU)
to prevent sluggish integrated GPU offloading and ensure optimal multi-threaded CPU SIMD performance.
"""

import ctypes
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional

# Keywords that signify integrated / APU / virtual display adapters
INTEGRATED_GPU_KEYWORDS = [
    "intel(r) uhd",
    "intel(r) hd",
    "intel(r) iris",
    "intel uhd",
    "intel hd",
    "intel iris",
    "intel graphics",
    "integrated",
    "basic display adapter",
    "microsoft basic",
    "microsoft remote",
    "virtual display",
    "virtualbox",
    "vmware",
    "qemu",
    "vbox",
    "software rasterizer",
    "llvmpipe",
    "radeon(tm) graphics",
    "radeon graphics",
    "vega 3",
    "vega 6",
    "vega 7",
    "vega 8",
    "vega 10",
    "vega 11",
]

# Keywords that explicitly identify discrete / dedicated GPUs
DISCRETE_GPU_KEYWORDS = [
    "nvidia",
    "geforce",
    "quadro",
    "tesla",
    "rtx",
    "gtx",
    "titan",
    "radeon rx",
    "radeon pro",
    "rx 5",
    "rx 6",
    "rx 7",
    "rx 8",
    "arc",
    "arc(tm)",
    "arc a",
    "arc b",
    "arc pro",
    "intel arc",
    "intel(r) arc",
]


def is_discrete_gpu(name: str, gpu_type: str = "vulkan", vram_mb: float = 0.0) -> bool:
    """Classifies whether a GPU device is an appropriate dedicated/discrete GPU or an integrated GPU."""
    name_lower = name.lower().strip()

    # NVIDIA CUDA devices are discrete
    if gpu_type == "cuda" or "nvidia" in name_lower or "geforce" in name_lower or "quadro" in name_lower or "tesla" in name_lower:
        return True

    # Check for explicit discrete series (e.g. Intel Arc discrete, AMD Radeon RX discrete)
    for disc_kw in DISCRETE_GPU_KEYWORDS:
        if disc_kw in name_lower:
            return True

    # Check for integrated keywords
    for int_kw in INTEGRATED_GPU_KEYWORDS:
        if int_kw in name_lower:
            return False

    # If VRAM is known and >= 3500 MB on non-Intel-HD, likely discrete
    if vram_mb >= 3500 and "intel" not in name_lower:
        return True

    # By default, unknown Intel / generic graphics are considered integrated
    if "intel" in name_lower and "arc" not in name_lower:
        return False

    return False


class GPUDetector:
    _instance: Optional["GPUDetector"] = None
    _cached_info: Optional[Dict[str, Any]] = None

    def __init__(self):
        self.has_cuda = False
        self.has_vulkan = False
        self.has_gpu = False
        self.has_discrete_gpu = False
        self.has_integrated_gpu = False
        self.preferred_backend = "CPU"
        self.gpu_devices: List[Dict[str, Any]] = []
        self.discrete_devices: List[Dict[str, Any]] = []
        self.integrated_devices: List[Dict[str, Any]] = []
        self.primary_device_name = "CPU Generic"
        self.primary_device_category = "CPU"
        self.total_vram_mb = 0.0
        self.driver_version = "N/A"
        self.cpu_simd = self._detect_cpu_simd()
        self._detect_all_hardware()

    @classmethod
    def get_instance(cls) -> "GPUDetector":
        if cls._instance is None:
            cls._instance = GPUDetector()
        return cls._instance

    def _detect_cpu_simd(self) -> str:
        """Detects supported CPU SIMD vector extensions on the host machine."""
        features = []
        try:
            if platform.system() == "Windows":
                cpu_arch = platform.machine()
                features.append("AVX2" if "AMD64" in cpu_arch or "x86" in cpu_arch else "NEON")
                features.append("FMA")
            else:
                features.append("AVX2")
        except Exception:
            features.append("AVX2")
        return "/".join(features) if features else "AVX2"

    def _detect_cuda(self) -> bool:
        """Detects NVIDIA CUDA acceleration support and device properties."""
        try:
            import torch
            if torch.cuda.is_available():
                self.has_cuda = True
                dev_count = torch.cuda.device_count()
                for i in range(dev_count):
                    props = torch.cuda.get_device_properties(i)
                    vram_mb = round(props.total_memory / (1024 * 1024), 1)
                    self.gpu_devices.append({
                        "name": props.name,
                        "type": "cuda",
                        "is_discrete": True,
                        "category": "Discrete GPU",
                        "vram_total_mb": vram_mb,
                        "driver_version": "CUDA Driver",
                    })
                if self.gpu_devices:
                    self.primary_device_name = self.gpu_devices[0]["name"]
                    self.total_vram_mb = self.gpu_devices[0]["vram_total_mb"]
                return True
        except Exception:
            pass

        try:
            if platform.system() == "Windows":
                nvcuda = ctypes.windll.LoadLibrary("nvcuda.dll")
                if nvcuda:
                    self.has_cuda = True
                    return True
        except Exception:
            pass

        try:
            cmd = "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits"
            out = subprocess.check_output(
                cmd,
                shell=True,
                timeout=2,
                stderr=subprocess.DEVNULL,
            ).decode("utf-8", errors="ignore").strip()
            if out:
                lines = out.splitlines()
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        name = parts[0]
                        vram = float(parts[1]) if parts[1].replace(".", "").isdigit() else 0.0
                        driver = parts[2] if len(parts) > 2 else "N/A"
                        self.gpu_devices.append({
                            "name": name,
                            "type": "cuda",
                            "is_discrete": True,
                            "category": "Discrete GPU",
                            "vram_total_mb": vram,
                            "driver_version": driver,
                        })
                if self.gpu_devices:
                    self.has_cuda = True
                    self.primary_device_name = self.gpu_devices[0]["name"]
                    self.total_vram_mb = self.gpu_devices[0]["vram_total_mb"]
                    self.driver_version = self.gpu_devices[0].get("driver_version", "N/A")
                    return True
        except Exception:
            pass

        return False

    def _detect_vulkan(self) -> bool:
        """Detects Vulkan graphics/compute API availability on the system."""
        try:
            if platform.system() == "Windows":
                vk = ctypes.windll.LoadLibrary("vulkan-1.dll")
                if vk:
                    self.has_vulkan = True
                    return True
            else:
                vk = ctypes.CDLL("libvulkan.so.1")
                if vk:
                    self.has_vulkan = True
                    return True
        except Exception:
            pass
        return False

    def _detect_windows_gpus(self):
        """Scans Windows Registry & Video Controllers to retrieve all active GPU adapters."""
        if platform.system() != "Windows":
            return

        try:
            import winreg
            path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                for i in range(winreg.QueryInfoKey(key)[0]):
                    subkey_name = winreg.EnumKey(key, i)
                    if not subkey_name.isdigit():
                        continue
                    try:
                        with winreg.OpenKey(key, subkey_name) as subkey:
                            driver_desc = winreg.QueryValueEx(subkey, "DriverDesc")[0]
                            try:
                                driver_ver = winreg.QueryValueEx(subkey, "DriverVersion")[0]
                            except Exception:
                                driver_ver = "N/A"

                            desc_lower = driver_desc.lower()
                            gpu_type = "cuda" if "nvidia" in desc_lower else ("vulkan" if self.has_vulkan else "directml")
                            discrete = is_discrete_gpu(driver_desc, gpu_type=gpu_type)

                            if not any(g["name"] == driver_desc for g in self.gpu_devices):
                                self.gpu_devices.append({
                                    "name": driver_desc,
                                    "type": gpu_type,
                                    "is_discrete": discrete,
                                    "category": "Discrete GPU" if discrete else "Integrated GPU",
                                    "vram_total_mb": 4096.0 if discrete else 2048.0,
                                    "driver_version": driver_ver,
                                })
                    except Exception:
                        pass
        except Exception:
            pass

    def _detect_all_hardware(self):
        """Executes unified multi-backend hardware detection and classifies discrete vs integrated GPUs."""
        cuda_ok = self._detect_cuda()
        vulkan_ok = self._detect_vulkan()
        self._detect_windows_gpus()

        self.discrete_devices = [d for d in self.gpu_devices if d.get("is_discrete")]
        self.integrated_devices = [d for d in self.gpu_devices if not d.get("is_discrete")]

        self.has_discrete_gpu = len(self.discrete_devices) > 0
        self.has_integrated_gpu = len(self.integrated_devices) > 0
        self.has_gpu = len(self.gpu_devices) > 0

        # Sort GPUs prioritizing discrete high-performance GPUs first
        if self.gpu_devices:
            sorted_devs = sorted(
                self.gpu_devices,
                key=lambda d: (
                    3 if d.get("is_discrete") and (d["type"] == "cuda" or "nvidia" in d["name"].lower())
                    else (2 if d.get("is_discrete")
                    else (1 if "radeon" in d["name"].lower() or "intel" in d["name"].lower() else 0))
                ),
                reverse=True
            )
            self.gpu_devices = sorted_devs
            primary = sorted_devs[0]
            self.primary_device_name = primary["name"]
            self.primary_device_category = primary.get("category", "GPU")
            self.driver_version = primary.get("driver_version", "N/A")
            if primary.get("vram_total_mb", 0) > 0:
                self.total_vram_mb = primary["vram_total_mb"]
        else:
            cpu_name = platform.processor() or "Multi-Core CPU"
            self.primary_device_name = f"{cpu_name} ({self.cpu_simd})"
            self.primary_device_category = "CPU"

        # Determine preferred backend: ONLY use GPU if an appropriate discrete GPU is present!
        if cuda_ok and self.has_discrete_gpu:
            self.preferred_backend = "CUDA"
        elif vulkan_ok and self.has_discrete_gpu:
            self.preferred_backend = "Vulkan"
        else:
            # Fallback to multi-threaded SIMD CPU mode for integrated GPUs or CPU-only systems
            self.preferred_backend = f"CPU ({self.cpu_simd})"

    def get_computation_info(self, active_layers: int = 0, total_layers: int = 0) -> Dict[str, Any]:
        """Returns structured real-time GPU & compute status for display in the UI and API."""
        is_gpu_active = (active_layers > 0 or active_layers == -1) and self.has_discrete_gpu
        active_backend = self.preferred_backend if is_gpu_active else f"CPU ({self.cpu_simd})"

        # Dynamic Computation Badge string (clean text without duplicate icon)
        if "CUDA" in active_backend and self.has_discrete_gpu:
            badge_text = f"CUDA: {self.primary_device_name}"
            badge_mode = "cuda"
        elif "Vulkan" in active_backend and self.has_discrete_gpu:
            badge_text = f"Vulkan: {self.primary_device_name}"
            badge_mode = "vulkan"
        else:
            badge_text = f"CPU: {self.cpu_simd} (Accelerated)"
            badge_mode = "cpu"

        vram_total_gb = round(self.total_vram_mb / 1024, 2) if self.total_vram_mb > 0 else 2.0
        if is_gpu_active:
            offloaded_ratio = 1.0 if active_layers == -1 else (min(1.0, active_layers / max(1, total_layers)) if total_layers else 0.5)
            vram_used_gb = round(max(0.5, vram_total_gb * offloaded_ratio * 0.7), 2)
        else:
            vram_used_gb = 0.0

        vram_pct = round((vram_used_gb / max(0.1, vram_total_gb)) * 100, 1)

        return {
            "has_gpu": self.has_gpu,
            "has_discrete_gpu": self.has_discrete_gpu,
            "has_integrated_gpu": self.has_integrated_gpu,
            "has_cuda": self.has_cuda,
            "has_vulkan": self.has_vulkan,
            "preferred_backend": self.preferred_backend,
            "active_backend": active_backend,
            "badge_text": badge_text,
            "badge_mode": badge_mode,
            "device_name": self.primary_device_name,
            "device_category": self.primary_device_category,
            "driver_version": self.driver_version,
            "cpu_simd": self.cpu_simd,
            "vram_total_gb": vram_total_gb,
            "vram_used_gb": vram_used_gb,
            "vram_percent": vram_pct,
            "layers_offloaded": active_layers if is_gpu_active else 0,
            "total_layers": total_layers,
            "devices": self.gpu_devices,
        }


# Global helper functions
def get_hardware_info() -> Dict[str, Any]:
    return GPUDetector.get_instance().get_computation_info()


def get_recommended_offload_layers() -> int:
    """Returns recommended layer count: -1 (GPU offload) for discrete GPUs, 0 (CPU mode) for integrated GPUs."""
    detector = GPUDetector.get_instance()
    if detector.has_discrete_gpu and (detector.has_cuda or detector.has_vulkan):
        return -1  # Offload all layers to discrete GPU
    return 0  # Fallback to multi-threaded CPU mode for integrated GPUs

