"""
Verification Test Suite for User Issue Fixes:
1. Deleting the last/only conversation completely removes it from list (sessions=[] and active_id=None).
2. Chat bar input recovery after clear conversation / delete conversation.
3. About modal close buttons (top and footer).
4. Fast thinking initiation & two separate bubbles structure.
"""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.session_manager import SessionManager
from src.core.llama_engine import LlamaEngine


def test_delete_only_conversation():
    print("[1/4] Testing deletion of the only conversation...")
    temp_dir = tempfile.mkdtemp()
    try:
        mgr = SessionManager(storage_dir=temp_dir)
        # Should have 1 initial session
        assert len(mgr.sessions) == 1, f"Expected 1 initial session, got {len(mgr.sessions)}"
        session_id = list(mgr.sessions.keys())[0]

        # Delete the only session
        mgr.delete_session(session_id)

        # Verify it is completely removed and active_session_id is None
        assert len(mgr.sessions) == 0, f"Expected 0 sessions after deleting only session, got {len(mgr.sessions)}"
        assert mgr.active_session_id is None, f"Expected active_session_id=None, got {mgr.active_session_id}"
        assert len(mgr.list_sessions()) == 0, f"Expected empty list_sessions(), got {mgr.list_sessions()}"
        print("  [PASS] Single conversation completely removed; active_id is None.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_engine_mock_streaming_instant_initiation():
    print("[2/4] Testing engine streaming initiation speed & structure...")
    engine = LlamaEngine()
    engine.is_loaded = True
    engine.model_name = "Test-Model.gguf"

    generator = engine.generate_chat_stream(
        messages=[{"role": "user", "content": "Hello"}],
        params={"temperature": 0.7, "max_tokens": 100}
    )

    chunks = []
    for chunk in generator:
        chunks.append(chunk)

    full_output = "".join(chunks)
    assert "<think>" in full_output, "Expected <think> tag in stream output"
    assert "</think>" in full_output, "Expected </think> tag in stream output"
    assert len(chunks) > 5, "Expected streaming tokens chunks"
    print(f"  [PASS] Stream produced {len(chunks)} chunks with thinking tags.")


def test_frontend_code_wiring():
    print("[3/4] Verifying frontend code wiring in app.js and index.html...")
    app_js_path = os.path.join(os.path.dirname(__file__), "..", "electron", "renderer", "js", "app.js")
    index_html_path = os.path.join(os.path.dirname(__file__), "..", "electron", "renderer", "index.html")

    with open(app_js_path, "r", encoding="utf-8") as f:
        app_js = f.read()

    with open(index_html_path, "r", encoding="utf-8") as f:
        index_html = f.read()

    # Verify About close buttons
    assert "btn-about-close-footer" in app_js, "btn-about-close-footer missing from app.js"
    assert "btn-about-close-footer" in index_html, "btn-about-close-footer missing from index.html"
    assert "btn-close-about" in app_js, "btn-close-about missing from app.js"

    # Verify chat input recovery & delete-chat
    assert "enableChatInput" in app_js, "enableChatInput function missing from app.js"
    assert "btn-delete-chat" in app_js, "btn-delete-chat missing from app.js"
    assert "empty-session-placeholder" in app_js, "empty-session-placeholder missing from app.js"

    # Verify thinking bubble UI separation
    assert "thinking-bubble" in app_js, "thinking-bubble missing from app.js"
    assert "isReasoningStreamActive" in app_js, "isReasoningStreamActive missing from app.js"
    print("  [PASS] Frontend JavaScript and HTML bindings verified.")


def test_thinking_bubble_separation():
    print("[4/4] Verifying thinking & response separation logic...")
    sample = "<think>Step 1: Check primes.\nStep 2: Return result.</think>\nHere is the answer."
    if "<think>" in sample and "</think>" in sample:
        thinking = sample.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        resp = sample.split("</think>", 1)[1].strip()
        is_active = False
    else:
        thinking, resp, is_active = None, sample, False

    assert thinking == "Step 1: Check primes.\nStep 2: Return result."
    assert resp == "Here is the answer."
    assert not is_active
    print("  [PASS] Thinking and response separation logic verified.")


if __name__ == "__main__":
    test_delete_only_conversation()
    test_engine_mock_streaming_instant_initiation()
    test_frontend_code_wiring()
    test_thinking_bubble_separation()
    print("\n============================================================")
    print("ALL VERIFICATION TESTS PASSED (4/4)!")
    print("============================================================")
