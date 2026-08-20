#!/usr/bin/env bash
# ⚡ Llama.cpp Turbo Desktop (Linux / macOS Launch Script)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  ⚡ Starting Llama.cpp Turbo Desktop (Linux / macOS)"
echo "============================================================"

# Detect Python executable
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python 3 is not installed or not in PATH."
    exit 1
fi

# 1. Virtual Environment Setup
if [ ! -f ".venv/bin/python" ]; then
    echo "[1/3] Setting up Python virtual environment..."
    "$PYTHON_CMD" -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# 2. Node Dependencies
if [ ! -d "node_modules/electron" ]; then
    echo "[2/3] Installing Electron desktop dependencies..."
    npm install
fi

# 3. Launch Application
echo "[3/3] Launching Llama.cpp Turbo Desktop..."
npx electron .
