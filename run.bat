@echo off
title ⚡ Llama.cpp Turbo Desktop (Electron + Google TurboQuant)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Setting up Python virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate
    pip install -r requirements.txt
)

if not exist "node_modules\electron" (
    echo [2/3] Installing Electron desktop dependencies...
    call npm.cmd install
    call node node_modules\electron\install.js
)

echo [3/3] Launching Llama.cpp Turbo Desktop (Electron)...
call npx.cmd electron .
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
