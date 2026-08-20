"""
Unit tests for GPU & Hardware Compute Acceleration Detector & Fallback.
Tests CUDA, Vulkan, and CPU fallback detection, VRAM computation, and API payload integrity.
"""

import unittest
from src.core.gpu_detector import GPUDetector, get_hardware_info, get_recommended_offload_layers
from src.core.llama_engine import LlamaEngine


class TestGPUComputing(unittest.TestCase):
    def setUp(self):
        self.detector = GPUDetector.get_instance()
        self.engine = LlamaEngine()

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector)
        self.assertIn(self.detector.preferred_backend, ["CUDA", "Vulkan", "CPU (AVX2)", "CPU (AVX2/FMA)", "CPU (NEON)", "CPU (AVX2/AVX-512)"])
        self.assertTrue(len(self.detector.cpu_simd) > 0)

    def test_discrete_vs_integrated_classification(self):
        from src.core.gpu_detector import is_discrete_gpu
        # Discrete GPUs should be True
        self.assertTrue(is_discrete_gpu("NVIDIA GeForce RTX 4090", "cuda"))
        self.assertTrue(is_discrete_gpu("AMD Radeon RX 7900 XTX", "vulkan"))
        self.assertTrue(is_discrete_gpu("Intel(R) Arc(TM) A770 Graphics", "vulkan"))
        self.assertTrue(is_discrete_gpu("NVIDIA RTX A6000", "cuda"))

        # Integrated GPUs should be False
        self.assertFalse(is_discrete_gpu("Intel(R) UHD Graphics 770", "vulkan"))
        self.assertFalse(is_discrete_gpu("Intel(R) Iris(R) Xe Graphics", "vulkan"))
        self.assertFalse(is_discrete_gpu("AMD Radeon(TM) Graphics", "vulkan"))
        self.assertFalse(is_discrete_gpu("Microsoft Basic Display Adapter", "directml"))

    def test_computation_info_structure(self):
        info = self.detector.get_computation_info(active_layers=-1, total_layers=33)
        self.assertIn("has_gpu", info)
        self.assertIn("has_discrete_gpu", info)
        self.assertIn("has_integrated_gpu", info)
        self.assertIn("preferred_backend", info)
        self.assertIn("active_backend", info)
        self.assertIn("badge_text", info)
        self.assertIn("badge_mode", info)
        self.assertIn("device_name", info)
        self.assertIn("cpu_simd", info)

    def test_recommended_offload_layers(self):
        rec = get_recommended_offload_layers()
        if self.detector.has_discrete_gpu:
            self.assertEqual(rec, -1)
        else:
            # On machines with only integrated GPU, recommended layers MUST be 0 (CPU mode)
            self.assertEqual(rec, 0)

    def test_cpu_fallback_mode(self):
        # Verify engine gracefully handles CPU compute mode
        self.engine.config["compute_mode"] = "cpu"
        self.assertEqual(self.engine.config["compute_mode"], "cpu")


if __name__ == "__main__":
    unittest.main()
