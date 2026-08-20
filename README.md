# ⚡ Llama.cpp Turbo Desktop (Qt + Google TurboQuant™)

A high-performance, modern desktop GUI application built in **Python with PySide6 (Qt 6)** to run `llama.cpp` GGUF models locally with full hardware acceleration, Google **TurboQuant™** optimization, system-wide model discovery, interactive playgrounds, GGUF metadata inspector, Hugging Face model downloader, and built-in OpenAI-compatible API server.
https://protik1810.github.io/LlamaCpp-Turbo/
---

## 🌟 Key Features

### 1. 🔍 System Model Discovery & Scanner
- **Multi-Threaded System Scan**: Automatically searches your computer for all compatible GGUF models.
- **Auto-Detects Popular LLM Tools**:
  - *LM Studio Models Cache* (`~/.cache/lm-studio/models` across all drives)
  - *Hugging Face Hub Cache* (`~/.cache/huggingface/hub`)
  - *Ollama Model Blobs* (`~/.ollama/models`)
  - *GPT4All & Jan.ai Models*
  - *User Downloads, Documents, Desktop, & Workspace*
  - *Custom Folder Picker* for external drives.
- **Deep GGUF Header Validation**: Verifies binary magic bytes, architecture, parameter count, quantization format, and file size.
- **1-Click Integration**: Load discovered models directly into the inference engine, inspect them in the GGUF viewer, or add them all to the quick-selector dropdown.

### 2. 🚀 Google TurboQuant™ Optimization
- **Fast Walsh-Hadamard Transform (FWHT)**: Eliminates activation and KV-cache outliers via orthogonal rotation before quantization.
- **TurboQuant KV Compression (INT2 / INT4 / INT8)**: Compresses KV-cache memory footprints by **4.0x to 8.0x** with near-zero perplexity loss.
- **Turbo Attention Sparsity Budget**: Token importance filter for ultra-long context acceleration.
- **Analytical Memory Estimator**: Real-time calculation of VRAM/RAM savings and memory bandwidth speedups.

### 3. 💬 Interactive Chat Playground
- Multi-turn conversation bubble interface with avatar markers and copy buttons.
- Full **Markdown formatting** with syntax-highlighted code blocks.
- Real-time streaming generation with live **tokens/sec (`tok/s`)**, **Time to First Token (`TTFT`)**, and token counters.
- Chat session persistence (Save/Load/Delete) and export to **Markdown (`.md`)** and **JSON (`.json`)**.
- System prompt presets: *Helpful Assistant*, *Senior Software Engineer*, *Creative Writer*, *JSON Extractor*, *Executive Summarizer*, *Math Reasoner*.

### 4. ✍️ Raw Completion Playground
- Freeform prompt completion playground for code generation, infilling (FIM), and text continuation.
- Pre-loaded sample prompt templates (Python generators, FastAPI endpoints, sci-fi stories, JSON schemas).

### 5. 🔍 GGUF Model Inspector & Visualizer
- Deep binary GGUF header parser: architecture, tensor count, block/layer count, context length, embedding dimensions, attention heads, and quantization format.
- Interactive memory estimator based on layer offload and context window.
- Filterable and searchable raw KV metadata table.

### 6. 🏬 Model Store & Hugging Face Downloader
- One-click downloader for curated popular GGUF models (*Llama 3.2 1B/3B, Qwen 2.5 0.5B/1.5B/3B, DeepSeek R1 Distill 1.5B, Phi-3.5 Mini, Gemma 2 2B*).
- Hugging Face search to find and download any GGUF repository directly.
- Chunked streaming downloads with resume support, speed meter, ETA, and cancellation.

### 7. 🌐 Built-in OpenAI-Compatible API Server
- Embedded **FastAPI / Uvicorn** server exposing `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, and `/health`.
- Connect **VS Code Continue**, **Cursor**, **Open WebUI**, **LangChain**, or custom Python / cURL scripts directly to your local desktop model.
- Live HTTP request & access log terminal with In-App API tester.

### 8. ⚙️ Sampling & Hardware Controls
- Complete hyperparameter tuning: Temperature, Top-P, Top-K, Min-P, Repeat Penalty, Presence/Frequency Penalties, Seed, Max Tokens, Stop Sequences.
- GBNF Grammar support (strict JSON mode or custom grammar parser).
- Hardware offloading: GPU offload layers (`n_gpu_layers`), context size (`n_ctx`), CPU threads, batch size, and Flash Attention.

---

## 🛠️ Quick Start

### 1. Launch in Development Mode
Double-click `run.bat` or run:
```cmd
run.bat
```

### 2. Build Standalone Executable (v1.0)
Double-click `build.bat` or run:
```cmd
build.bat
```
This builds:
- **Standalone App Directory**: `dist_app\win-unpacked\Llama.cpp Turbo Desktop.exe`
- **Single-File Portable EXE**: `dist_app\LlamaCppTurboDesktop-v1.0-Portable.exe`
- **Data Folders**:
  - `data/sessions/`: Persistent chat conversations
  - `models/`: Local GGUF models directory
  - `assets/`: App icons, logo, and artwork

