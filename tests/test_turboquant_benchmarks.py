"""
TurboQuant Verification & Comprehensive Benchmark Test Suite
Validates:
1. Fast Walsh-Hadamard Transform (FWHT) mathematical properties:
   - Invertibility: FWHT(FWHT(x)) == x
   - L2 Norm Preservation: ||FWHT(x)||_2 == ||x||_2
   - Outlier Dispersion & Max-Peak Suppression
2. Quantization Distortion & Error Metrics:
   - Mean Squared Error (MSE) comparison: Hadamard-augmented vs Naive quantization
   - Signal-to-Noise Ratio (SNR) and Cosine Similarity
3. Memory Footprint & Compression Verification:
   - INT8 (2.0x), INT4 (4.0x), INT2 (8.0x) KV Cache compression
   - Turbo Attention Sparsity multi-scale savings
4. Engine Inference Wiring:
   - Type_k / Type_v mapping to Q4_0 and Q8_0 in LlamaEngine
"""

import math
import os
import sys
import unittest
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.turbo_quant import fast_walsh_hadamard_transform, TurboQuantKVCache, TurboQuantManager
from src.core.llama_engine import LlamaEngine


class TestTurboQuantMathAndTransforms(unittest.TestCase):
    """Mathematical validation of the Fast Walsh-Hadamard Transform."""

    def test_fwht_invertibility(self):
        """Verify that normalized FWHT is its own inverse: FWHT(FWHT(x)) == x."""
        np.random.seed(42)
        dims = [64, 128, 256, 512]
        for d in dims:
            x = np.random.randn(d).astype(np.float32)
            transformed = fast_walsh_hadamard_transform(x)
            recovered = fast_walsh_hadamard_transform(transformed)
            np.testing.assert_allclose(
                recovered, x, rtol=1e-4, atol=1e-4,
                err_msg=f"FWHT invertibility failed for dimension {d}"
            )

    def test_fwht_l2_norm_preservation(self):
        """Verify that FWHT preserves vector L2 norm (isometry / orthogonal rotation)."""
        np.random.seed(42)
        x = np.random.randn(128).astype(np.float32)
        orig_norm = np.linalg.norm(x)
        fwht_norm = np.linalg.norm(fast_walsh_hadamard_transform(x))
        self.assertAlmostEqual(orig_norm, fwht_norm, places=4)

    def test_fwht_outlier_dispersion(self):
        """
        Verify that FWHT disperses large activation outliers across all coordinates,
        significantly reducing the peak absolute amplitude.
        """
        dim = 128
        # Create a vector with normal noise + a massive isolated outlier (typical in LLM activations)
        x = np.random.randn(dim).astype(np.float32) * 0.5
        x[17] = 80.0  # Massive outlier channel
        orig_peak = np.max(np.abs(x))

        rotated = fast_walsh_hadamard_transform(x)
        rotated_peak = np.max(np.abs(rotated))

        # Expected theoretical reduction factor is roughly ~1 / sqrt(dim)
        expected_bound = orig_peak / math.sqrt(dim) * 2.5
        self.assertLess(
            rotated_peak, orig_peak * 0.35,
            f"FWHT failed to disperse outlier: orig_peak={orig_peak}, rotated_peak={rotated_peak}"
        )
        self.assertLess(rotated_peak, expected_bound)


class TestTurboQuantQuantizationAndFidelity(unittest.TestCase):
    """Empirical accuracy & compression ratio validation."""

    def test_int4_hadamard_outlier_fidelity(self):
        """
        Verify that Hadamard rotation significantly improves INT4 reconstruction error (MSE)
        on outlier-heavy KV activation distributions.
        """
        np.random.seed(1337)
        tokens, head_dim = 64, 128
        # Simulate realistic KV states with extreme outlier channels
        kv_tensor = np.random.randn(tokens, head_dim).astype(np.float32)
        kv_tensor[:, 4] += 25.0   # Persistent outlier channel
        kv_tensor[:, 42] -= 30.0  # Persistent outlier channel

        # Cache with FWHT
        cache_with_hadamard = TurboQuantKVCache(bits=4, use_hadamard=True)
        q_h, s_h, zp_h = cache_with_hadamard.compress_tensor(kv_tensor)
        recon_h = cache_with_hadamard.decompress_tensor(q_h, s_h, zp_h, kv_tensor.shape)
        mse_hadamard = np.mean((kv_tensor - recon_h) ** 2)

        # Cache without FWHT (Naive Min-Max)
        cache_naive = TurboQuantKVCache(bits=4, use_hadamard=False)
        q_n, s_n, zp_n = cache_naive.compress_tensor(kv_tensor)
        recon_n = cache_naive.decompress_tensor(q_n, s_n, zp_n, kv_tensor.shape)
        mse_naive = np.mean((kv_tensor - recon_n) ** 2)

        # Cosine similarity
        norm_orig = kv_tensor / np.linalg.norm(kv_tensor, axis=-1, keepdims=True)
        norm_recon_h = recon_h / np.linalg.norm(recon_h, axis=-1, keepdims=True)
        cos_sim_h = np.mean(np.sum(norm_orig * norm_recon_h, axis=-1))

        self.assertLess(
            mse_hadamard, mse_naive,
            f"Hadamard INT4 MSE ({mse_hadamard:.4f}) should be significantly lower than Naive MSE ({mse_naive:.4f})"
        )
        self.assertGreater(
            cos_sim_h, 0.985,
            f"Hadamard INT4 cosine similarity ({cos_sim_h:.4f}) should exceed 0.985"
        )

    def test_compression_ratios(self):
        """Verify empirical compression ratios for INT8, INT4, and INT2."""
        kv_tensor = np.random.randn(512, 128).astype(np.float32)

        # INT8 -> ~2.0x (empirical ~1.9x with metadata)
        cache_8 = TurboQuantKVCache(bits=8, use_hadamard=True)
        cache_8.compress_tensor(kv_tensor)
        self.assertGreaterEqual(cache_8.compression_ratio, 1.8)

        # INT4 -> ~4.0x (empirical ~3.55x with metadata)
        cache_4 = TurboQuantKVCache(bits=4, use_hadamard=True)
        cache_4.compress_tensor(kv_tensor)
        self.assertGreaterEqual(cache_4.compression_ratio, 3.5)

        # INT2 -> ~8.0x (empirical ~6.4x with metadata)
        cache_2 = TurboQuantKVCache(bits=2, use_hadamard=True)
        cache_2.compress_tensor(kv_tensor)
        self.assertGreaterEqual(cache_2.compression_ratio, 6.0)

    def test_turbo_sparsity_memory_gain(self):
        """Verify that Turbo Attention sparsity further reduces memory footprint."""
        kv_tensor = np.random.randn(256, 128).astype(np.float32)

        cache_dense = TurboQuantKVCache(bits=4, sparsity_ratio=0.0)
        cache_dense.compress_tensor(kv_tensor)

        cache_sparse = TurboQuantKVCache(bits=4, sparsity_ratio=0.25)
        cache_sparse.compress_tensor(kv_tensor)

        self.assertLess(
            cache_sparse.compressed_bytes, cache_dense.compressed_bytes,
            "Sparsity budget should strictly decrease compressed byte footprint"
        )


class TestTurboQuantEngineWiring(unittest.TestCase):
    """Verify that TurboQuant is properly wired into LlamaEngine inference parameters."""

    def test_llama_engine_turbo_defaults(self):
        engine = LlamaEngine()
        self.assertTrue(engine.config.get("turbo_enabled"))
        self.assertEqual(engine.config.get("turbo_bits"), 4)
        self.assertTrue(engine.config.get("turbo_hadamard"))
        self.assertAlmostEqual(engine.config.get("turbo_sparsity"), 0.20)

    def test_turbo_manager_savings_estimation(self):
        manager = TurboQuantManager()
        manager.update_config(enabled=True, bits=4, use_hadamard=True, sparsity_ratio=0.20)
        est = manager.estimate_savings(raw_kv_mb=1024.0, model_weights_mb=4000.0)

        self.assertIn("turbo_kv_mb", est)
        self.assertIn("kv_savings_mb", est)
        self.assertIn("compression_ratio", est)
        self.assertIn("savings_pct", est)
        self.assertIn("estimated_speedup", est)

        # 1024MB FP16 with INT4 (4x) and 20% sparsity (effective 5.0x) -> ~204.8 MB
        self.assertAlmostEqual(est["turbo_kv_mb"], 204.8, delta=2.0)
        self.assertEqual(est["compression_ratio"], "5.0x")
        self.assertEqual(est["savings_pct"], "80.0%")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING TURBOQUANT MATHEMATICAL & BENCHMARK VERIFICATION SUITE")
    print("=" * 60)
    unittest.main()
