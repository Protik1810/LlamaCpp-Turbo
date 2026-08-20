# Contributing to Llama.cpp Turbo Desktop

Thank you for your interest in contributing to **Llama.cpp Turbo Desktop**! We welcome contributions from developers, researchers, and open-source enthusiasts.

---

## 🛠️ Development Setup

### 1. Prerequisites
- **Python**: 3.10, 3.11, or 3.12 (with `pip` and `venv`)
- **Node.js**: v18.0.0 or higher (with `npm`)
- **Git**: Latest version
- **C/C++ Build Tools** *(optional, for compiling custom llama.cpp wheels with CUDA/Vulkan support)*

### 2. Quick Local Setup

```bash
# Clone the repository
git clone https://github.com/Protik1810/llamacpp-turbo.git
cd llamacpp-turbo

# Windows quick launch
run.bat

# Linux / macOS quick launch
chmod +x run.sh
./run.sh
```

### 3. Manual Step-by-Step Setup

```bash
# Set up Python virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
pip install flake8 pytest

# Install Electron dependencies
npm install

# Launch desktop app in dev mode
npm start
```

---

## 🧪 Testing & Verification

Before submitting any code changes, ensure all tests pass:

```bash
# Run full unit test suite
python -m unittest discover tests

# Run TurboQuant mathematical and empirical benchmarks
python tests/test_turboquant_benchmarks.py

# Run Flake8 linter
flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
```

---

## 📂 Project Architecture

```
llamacpp-turbo/
├── assets/                 # App icons, screenshots, and visual branding
├── docs/                   # GitHub Pages documentation and benchmark reports
├── electron/               # Electron main process and Chromium lifecycle
│   ├── main.js             # Window manager, IPC bridges, and process spawning
│   └── preload.js          # Secure context bridge between Electron and Web UI
├── src/                    # Python Backend Core
│   ├── server.py           # FastAPI + Uvicorn server (REST & SSE streaming)
│   └── core/
│       ├── turbo_quant.py  # Google TurboQuant FWHT rotation & KV quantization
│       ├── llama_engine.py # llama-cpp-python engine & inference workers
│       ├── gpu_detector.py # CUDA / Vulkan / AVX2 hardware detection & routing
│       ├── model_scanner.py# Multi-threaded filesystem model crawler
│       ├── downloader.py   # Multi-threaded Hugging Face chunk downloader
│       └── session_manager.py # Chat session persistence (JSON/Markdown)
├── tests/                  # Unit and integration test suite
├── index.html              # Modern Web UI (Glassmorphic dark/light themes)
├── js/app.js               # Frontend controller, stream handler, thinking parser
└── css/style.css           # Vanilla CSS design system and animations
```

---

## 🔀 Submitting a Pull Request

1. **Fork** the repository and create a descriptive branch:
   ```bash
   git checkout -b feature/turboquant-fp8-support
   ```
2. **Commit** your changes with clear messages adhering to conventional commits:
   ```bash
   git commit -m "feat(turboquant): add FP8 KV cache quantization support"
   ```
3. **Push** to your fork and submit a Pull Request targeting the `main` branch.
4. Ensure all CI checks pass.

---

## 📜 Code Style Guidelines
- **Python**: PEP 8 compliance, type annotations where practical.
- **JavaScript**: Clean ES6+, modular functions, descriptive variable names.
- **CSS**: Predefined CSS custom property tokens, responsive layouts, no inline magic numbers.

---

## 💬 Community & Questions
Feel free to open an issue or start a discussion on GitHub if you have any questions or architectural suggestions!
