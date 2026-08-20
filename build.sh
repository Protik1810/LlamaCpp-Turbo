#!/usr/bin/env bash
# ⚡ Llama.cpp Turbo Desktop (Linux / macOS Build Script)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Building Llama.cpp Turbo Desktop Standalone Packages"
echo "============================================================"

# Detect OS
OS_TYPE="$(uname -s)"
echo "[*] Detected Host OS: $OS_TYPE"

# 1. Compile Backend with PyInstaller
echo "[1/3] Compiling Python Backend Server..."
if [ -f ".venv/bin/pyinstaller" ]; then
    .venv/bin/pyinstaller --noconfirm --distpath dist_backend server.spec
else
    pyinstaller --noconfirm --distpath dist_backend server.spec
fi

# 2. Package Electron App
echo "[2/3] Packaging Electron Desktop Application..."
if [ "$OS_TYPE" = "Darwin" ]; then
    echo "[*] Building macOS DMG / App..."
    npx electron-builder --mac
elif [ "$OS_TYPE" = "Linux" ]; then
    echo "[*] Building Linux AppImage & Directory..."
    npx electron-builder --linux AppImage dir
else
    npx electron-builder --dir
fi

echo "============================================================"
echo "  BUILD COMPLETE!"
echo "  Output generated in: dist_app/"
echo "============================================================"
