@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Building Llama.cpp Turbo Desktop Standalone EXE (v1.0)
echo ============================================================
echo.

echo [*] Terminating any running application processes...
taskkill /F /IM LlamaCppTurboDesktop.exe /IM server.exe /IM electron.exe 2>nul
timeout /t 1 /nobreak >nul
if exist "dist_app" rd /s /q "dist_app" 2>nul

echo [1/3] Compiling Python Backend Server with PyInstaller...
call .\.venv\Scripts\pyinstaller.exe --noconfirm --distpath dist_backend server.spec
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to compile Python backend server.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Packaging Standalone and Portable Desktop Executables...
call cmd.exe /c "npx.cmd electron-builder --win dir portable"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to package Electron desktop application.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/4] Setting up Additional Data Folders...
if not exist "dist_app\win-unpacked\data\sessions" mkdir "dist_app\win-unpacked\data\sessions"
if not exist "dist_app\win-unpacked\models" mkdir "dist_app\win-unpacked\models"
if not exist "dist_app\win-unpacked\assets" xcopy "assets" "dist_app\win-unpacked\assets" /E /I /Y >nul 2>&1

echo.
echo [4/4] Compiling Windows Setup Installer with Inno Setup...
set "ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist "%ISCC_PATH%" set "ISCC_PATH=C:\Program Files\Inno Setup 6\ISCC.exe"

if exist "%ISCC_PATH%" (
    "%ISCC_PATH%" "installer.iss"
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Inno Setup compilation had issues.
    ) else (
        echo [OK] Setup Installer generated successfully.
    )
) else (
    echo [INFO] Inno Setup compiler not found at standard path. Skipping installer compilation.
)

echo.
echo ============================================================
echo   BUILD COMPLETE!
echo.
echo   1. Windows Setup Installer:
echo      dist_app\LlamaCppTurboDesktop-v1.0-Setup.exe
echo.
echo   2. Standalone Executable Directory:
echo      dist_app\win-unpacked\LlamaCppTurboDesktop.exe
echo.
echo   3. Portable Single-File Executable:
echo      dist_app\LlamaCppTurboDesktop-v1.0-Portable.exe
echo.
echo   Additional Data Folders Included:
echo   - dist_app\win-unpacked\data\sessions\  (Chat history)
echo   - dist_app\win-unpacked\models\         (Local GGUF models)
echo   - dist_app\win-unpacked\assets\         (App icons and graphics)
echo ============================================================
echo.
