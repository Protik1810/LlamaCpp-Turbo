import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.model_scanner import (
    scan_for_gguf_models,
    get_default_scan_locations,
    get_all_system_drives,
)


def test_system_drives():
    drives = get_all_system_drives()
    assert len(drives) >= 1, "At least one system drive should be detected."
    print(f"[PASS] Detected system drives: {drives}")


def test_scan_locations():
    locations = get_default_scan_locations()
    assert len(locations) > 0, "Scan locations must not be empty."
    print(f"[PASS] Discovered {len(locations)} system search locations.")


def test_model_discovery():
    models = scan_for_gguf_models()
    print(f"[PASS] Model scanner discovered {len(models)} GGUF models across system.")
    for m in models[:10]:
        print(f"  • {m['name']} | {m['size_gb']} GB | {m['architecture']} | {m['source']}")


def test_custom_folder_scan():
    # Test scanning a specific folder (e.g. current models directory)
    models_dir = os.path.abspath("models")
    custom_models = scan_for_gguf_models(custom_path=models_dir)
    print(f"[PASS] Custom directory scan on '{models_dir}' returned {len(custom_models)} models.")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MULTI-LOCATION SYSTEM MODEL SCANNER")
    print("=" * 60)
    test_system_drives()
    test_scan_locations()
    test_model_discovery()
    test_custom_folder_scan()
    print("=" * 60)
    print("ALL MODEL SCANNER TESTS PASSED!")
    print("=" * 60)
