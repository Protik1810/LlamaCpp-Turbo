"""
Icon & Logo Assets Verification Test Suite
Tests loading, validation, and rendering of application icons, window icons, and chat avatar emblems.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image


def test_icon_and_logo_assets():
    print("Testing Icon and Logo Assets Integrity...")

    # 1. Verify existence of asset files
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
    assert os.path.exists(os.path.join(assets_dir, "icon.png")), "icon.png missing"
    assert os.path.exists(os.path.join(assets_dir, "icon.ico")), "icon.ico missing"
    assert os.path.exists(os.path.join(assets_dir, "logo.png")), "logo.png missing"
    assert os.path.exists(os.path.join(assets_dir, "avatar.png")), "avatar.png missing"
    print("  [PASS] All asset files (icon.png, icon.ico, logo.png, avatar.png) verified on disk")

    # 2. Test icon.png loading
    with Image.open(os.path.join(assets_dir, "icon.png")) as img:
        assert img.size[0] >= 64 and img.size[1] >= 64
        print(f"  [PASS] icon.png verified ({img.size[0]}x{img.size[1]}, mode={img.mode})")

    # 3. Test icon.ico loading
    with Image.open(os.path.join(assets_dir, "icon.ico")) as ico:
        assert ico.size[0] >= 16
        print(f"  [PASS] icon.ico verified ({ico.size[0]}x{ico.size[1]})")

    # 4. Test logo.png loading
    with Image.open(os.path.join(assets_dir, "logo.png")) as logo:
        assert logo.size[0] > 0 and logo.size[1] > 0
        print(f"  [PASS] logo.png dimensions ({logo.size[0]}x{logo.size[1]}) verified")

    # 5. Test avatar.png loading
    with Image.open(os.path.join(assets_dir, "avatar.png")) as avatar:
        assert avatar.size[0] > 0 and avatar.size[1] > 0
        print(f"  [PASS] avatar.png dimensions ({avatar.size[0]}x{avatar.size[1]}) verified")

    print("\nALL ICON AND LOGO INTEGRITY TESTS PASSED (5/5)!")


if __name__ == "__main__":
    test_icon_and_logo_assets()
