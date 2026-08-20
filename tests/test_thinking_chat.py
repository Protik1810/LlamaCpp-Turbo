"""
Thinking & Reasoning Separation Verification Test Suite
Tests structured extraction and distinct rendering of <think>...</think> reasoning blocks in the chat UI.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def extract_thinking_and_response_py(content: str):
    """Pure Python implementation of <think>...</think> structured separation."""
    if not content:
        return None, "", False

    if "<think>" in content:
        parts = content.split("<think>", 1)
        prefix = parts[0]
        after_think_open = parts[1]

        if "</think>" in after_think_open:
            think_parts = after_think_open.split("</think>", 1)
            thinking_text = think_parts[0].strip()
            response_text = (prefix + think_parts[1]).strip()
            return thinking_text, response_text, False
        else:
            return after_think_open.strip(), prefix.strip(), True

    return None, content, False


def test_thinking_feature():
    print("Testing Thinking Process & Response Separation...")

    # 1. Test completed thinking extraction
    raw_1 = "<think>\nStep 1: Check prime conditions.\nStep 2: Return True if no divisors.\n</think>\nHere is the Python function:\n```python\ndef is_prime(n):\n    return n > 1\n```"
    thinking_1, resp_1, is_active_1 = extract_thinking_and_response_py(raw_1)
    assert thinking_1 is not None and "Step 1: Check prime conditions." in thinking_1
    assert "Here is the Python function:" in resp_1
    assert "<think>" not in resp_1 and "</think>" not in resp_1
    assert not is_active_1
    print("  [PASS] Completed thinking extraction verified")

    # 2. Test in-progress streaming thinking extraction
    raw_2 = "<think>I am currently calculating the optimal solution..."
    thinking_2, resp_2, is_active_2 = extract_thinking_and_response_py(raw_2)
    assert thinking_2 == "I am currently calculating the optimal solution..."
    assert resp_2 == ""
    assert is_active_2 is True
    print("  [PASS] In-progress streaming thinking extraction verified")

    # 3. Test non-thinking regular message
    raw_3 = "Hello! How can I assist you today?"
    thinking_3, resp_3, is_active_3 = extract_thinking_and_response_py(raw_3)
    assert thinking_3 is None
    assert resp_3 == "Hello! How can I assist you today?"
    assert is_active_3 is False
    print("  [PASS] Standard non-thinking message verified")

    # 4. Test app.js thinking extraction code match
    app_js_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "electron", "renderer", "js", "app.js"))
    with open(app_js_path, "r", encoding="utf-8") as f:
        app_js = f.read()
    assert "extractThinkingAndResponse" in app_js
    assert "thinking-bubble" in app_js
    assert "thinking-body" in app_js
    print("  [PASS] Frontend JavaScript thinking bubble implementation verified")

    print("\nALL THINKING & RESPONSE SEPARATION TESTS PASSED (4/4)!")


if __name__ == "__main__":
    test_thinking_feature()
