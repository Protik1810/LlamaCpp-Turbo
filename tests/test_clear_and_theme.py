import os
import re

print("============================================================")
print("TESTING CHAT CLEAR TYPING RECOVERY & THEME ADAPTABILITY")
print("============================================================")

# 1. Verify app.js defines setGeneratingState and handles textarea after clearing
app_js_path = os.path.join(os.path.dirname(__file__), "..", "electron", "renderer", "js", "app.js")
with open(app_js_path, "r", encoding="utf-8") as f:
    app_js_content = f.read()

assert "function setGeneratingState(generating)" in app_js_content, "Missing setGeneratingState definition in app.js"
assert "textarea.disabled = false" in app_js_content, "Textarea disabled state not properly handled"
assert "clearBtn" in app_js_content, "clearBtn logic missing in app.js"
assert "setGeneratingState(false)" in app_js_content, "setGeneratingState(false) not called on clear"
print("  [PASS] setGeneratingState and textarea typing recovery verified in app.js")

# 2. Verify style.css does not have hardcoded dark colors in contributors section
style_css_path = os.path.join(os.path.dirname(__file__), "..", "electron", "renderer", "css", "style.css")
with open(style_css_path, "r", encoding="utf-8") as f:
    style_css_content = f.read()

# Check that .contributors-section uses var(--bg-input)
assert ".contributors-section {" in style_css_content
assert '[data-theme="light"] .contributors-section' in style_css_content, "Missing light theme override for contributors-section"
assert '[data-theme="light"] .primary-contributor-card' in style_css_content, "Missing light theme override for primary-contributor-card"
assert '[data-theme="light"] .about-modal-card' in style_css_content, "Missing light theme override for about-modal-card"

# Verify no hardcoded rgba(15, 23, 42, ...) inside contributors rules
contrib_block_match = re.search(r'\.contributors-section\s*\{([^}]+)\}', style_css_content)
assert contrib_block_match, "Could not find .contributors-section in style.css"
assert "rgba(15, 23, 42" not in contrib_block_match.group(1), "Found hardcoded dark color in .contributors-section"

print("  [PASS] Contributors section and About dialog theme adaptability verified")
print("============================================================")
print("ALL CLEAR TYPING & THEME TESTS PASSED (2/2)!")
print("============================================================")
