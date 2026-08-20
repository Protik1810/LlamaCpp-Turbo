import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.server import get_project_contributors
from src.core.llama_engine import LlamaEngine

# 1. Test Project Contributors Endpoint Logic
print("[1/3] Testing Dynamic Contributors Registry...")
contrib_data = get_project_contributors()
assert "primary" in contrib_data, "Missing primary contributor"
assert contrib_data["primary"]["name"] == "Protik Das", f"Unexpected primary name: {contrib_data['primary']['name']}"
assert len(contrib_data["contributors"]) >= 1, "No contributors returned"
print(f"  [PASS] Primary: {contrib_data['primary']['name']} ({contrib_data['primary']['role']})")
print(f"  [PASS] Total Contributors: {len(contrib_data['contributors'])}")

# 2. Test Default GBNF Engine Behavior (Must return None for default/none)
print("[2/3] Testing GBNF Default Reset...")
engine = LlamaEngine()
grammar_none = engine._prepare_grammar("none", "")
assert grammar_none is None, f"Expected None for 'none', got {grammar_none}"

grammar_default = engine._prepare_grammar("default", "")
assert grammar_default is None, f"Expected None for 'default', got {grammar_default}"

grammar_empty = engine._prepare_grammar("", "")
assert grammar_empty is None, f"Expected None for empty grammar, got {grammar_empty}"
print("  [PASS] GBNF default is None (Unconstrained Freeform)")

# 3. Test Diagnostic Stream with <think> Process
print("[3/3] Testing Thinking Process Chat Stream...", flush=True)
engine.is_loaded = True
engine.model_name = "Qwen2.5-Coder-7B-Instruct.Q4_K_M.gguf"
stream = engine.generate_chat_stream([{"role": "user", "content": "Hello"}], {"temperature": 0.7})
full_stream_text = "".join(list(stream))
assert "<think>" in full_stream_text and "</think>" in full_stream_text, "Missing thinking tags in stream"
print("  [PASS] Thinking process tags present in chat stream output", flush=True)

print("============================================================")
print("ALL CONTRIBUTORS, GBNF & THINKING TESTS PASSED (3/3)!")
print("============================================================")
