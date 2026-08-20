/**
 * ⚡ Llama.cpp Turbo Desktop - Renderer Application Script
 * Orchestrates UI interactions, SSE token streaming, thinking process separation,
 * GGUF inspection, Hugging Face model search/download, and TurboQuant configuration.
 */

// API Base URL
const API_BASE = window.electronAPI ? window.electronAPI.backendUrl : 'http://127.0.0.1:8008';

// State Management
const state = {
  activeTab: 'tab-chat',
  theme: 'dark',
  activeSessionId: null,
  sessions: [],
  currentModel: null,
  isGenerating: false,
  abortController: null,
  activeDownloadKey: null,
  downloadPollInterval: null,
};

// Preset System Instructions
const SYSTEM_PRESETS = {
  'General Assistant': 'You are a helpful, expert AI assistant with broad knowledge across science, mathematics, coding, and writing.',
  'Expert Coder': 'You are a senior software engineer and system architect. Provide clean, modular, production-ready code with concise explanations and best practices.',
  'Chain-of-Thought Reasoner': 'You are a deep analytical thinker. Break down problems step-by-step using first principles. Explain your reasoning carefully before reaching final conclusions.',
  'Concise & Direct': 'You are a direct, concise assistant. Answer questions immediately with minimal fluff and maximum clarity.',
  'Creative Writer': 'You are an evocative creative writer and storyteller. Use vivid imagery, dynamic pacing, and nuanced characterization.'
};

// Prompt Completion Templates
const SAMPLE_PROMPTS = {
  code_infill: `# Complete the binary search algorithm with edge case handling:\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    `,
  docstring: `def compute_kv_cache_savings(tokens: int, bits: int = 4, hadamard: bool = True):\n    """`,
  json_extract: `Extract all customer details into JSON from this text:\n"Dr. Aris Thorne (aris@quantum.io) ordered 3 H100 GPUs to San Francisco, CA on 2026-08-18."\n\n{\n  "name":`,
  sql_query: `-- Write an optimized PostgreSQL query to find top 5 customers with highest lifetime spend in 2026:\nSELECT`,
  creative_story: `The neural link flickered once, then flashed brilliant cyan as the quantum processor awakened.`
};

// Integration Snippets
const CODE_SNIPPETS = {
  'snip-cursor': `{
  "models": [
    {
      "title": "Local Llama.cpp Turbo",
      "provider": "openai",
      "model": "local-llama",
      "apiBase": "${API_BASE}/v1"
    }
  ]
}`,
  'snip-python': `import openai

client = openai.OpenAI(
    base_url="${API_BASE}/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local-llama",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Google TurboQuant."}
    ],
    temperature=0.7,
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
`,
  'snip-node': `import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "${API_BASE}/v1",
  apiKey: "not-needed",
});

const stream = await openai.chat.completions.create({
  model: "local-llama",
  messages: [{ role: "user", content: "Explain Google TurboQuant." }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || "");
}
`,
  'snip-langchain': `from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="${API_BASE}/v1",
    api_key="not-needed",
    model="local-llama",
    temperature=0.7
)

response = llm.invoke("What are the benefits of KV cache quantization?")
print(response.content)
`,
  'snip-curl': `curl -X POST ${API_BASE}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "local-llama",
    "messages": [{"role": "user", "content": "Hello local LLM!"}],
    "temperature": 0.7
  }'`
};

/* ==========================================================================
   Initialization
   ========================================================================== */
document.addEventListener('DOMContentLoaded', () => {
  initWindowControls();
  initTheme();
  initNavigationTabs();
  initDrawerTabs();
  initSamplingSliders();
  initHardwareControls();
  initChatPlayground();
  initCompletionPlayground();
  initInspector();
  initModelStore();
  initServerManager();
  initAboutModal();

  // Initial Data Fetch
  fetchSystemStats();
  fetchModelsList();
  fetchSessions();
  fetchCuratedModels();

  // Periodic Health & Stats Polling
  setInterval(fetchSystemStats, 2500);
});

/* ==========================================================================
   About Modal & Contributors
   ========================================================================== */
function initAboutModal() {
  const aboutBtn = document.getElementById('btn-about');
  const aboutModal = document.getElementById('about-modal');
  const closeBtn = document.getElementById('btn-close-about');
  const closeFooterBtn = document.getElementById('btn-about-close-footer');

  const openAbout = () => {
    if (aboutModal) aboutModal.classList.add('active');
    loadProjectContributors();
  };

  const closeAbout = () => {
    if (aboutModal) aboutModal.classList.remove('active');
  };

  aboutBtn?.addEventListener('click', openAbout);
  closeBtn?.addEventListener('click', closeAbout);
  closeFooterBtn?.addEventListener('click', closeAbout);

  aboutModal?.addEventListener('click', (e) => {
    if (e.target === aboutModal) closeAbout();
  });
}

async function loadProjectContributors() {
  const container = document.getElementById('contributors-list-container');
  const badge = document.getElementById('contributors-count-badge');
  if (!container) return;

  try {
    const res = await fetch(`${API_BASE}/v1/project/contributors`);
    const data = await res.json();
    const contributors = data.contributors || [];
    if (badge) badge.textContent = `${contributors.length} Registered`;

    container.innerHTML = '';
    contributors.forEach(c => {
      const isPrimary = c.is_primary;
      const card = document.createElement('div');
      card.className = `contributor-card ${isPrimary ? 'primary-contributor-card' : ''}`;

      const avatarGlyph = isPrimary ? '👑' : '💻';
      const roleText = isPrimary
        ? 'Primary Contributor & Original Creator'
        : `${c.commits ? `${c.commits} commit${c.commits > 1 ? 's' : ''}` : 'Community Contributor'}${c.email ? ` • ${c.email}` : ''}`;

      const badgeClass = isPrimary ? 'badge badge-cyan' : 'badge';
      const badgeText = isPrimary ? '⭐ Founder' : (c.badge || '⚡ Contributor');

      card.innerHTML = `
        <div class="contributor-avatar">${avatarGlyph}</div>
        <div class="contributor-info">
          <div class="contributor-name">
            ${escapeHtml(c.name)}
            ${isPrimary ? '<span class="contributor-tag">Lead Architect</span>' : ''}
          </div>
          <div class="contributor-role">${escapeHtml(roleText)}</div>
        </div>
        <span class="${badgeClass}" style="font-size: 9px;">${badgeText}</span>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error('Error fetching contributors:', err);
  }
}

/* ==========================================================================
   Window Controls & Frameless Titlebar
   ========================================================================== */
function initWindowControls() {
  const btnMin = document.getElementById('btn-win-min');
  const btnMax = document.getElementById('btn-win-max');
  const btnClose = document.getElementById('btn-win-close');

  if (window.electronAPI) {
    btnMin?.addEventListener('click', () => window.electronAPI.minimize());
    btnMax?.addEventListener('click', () => window.electronAPI.maximize());
    btnClose?.addEventListener('click', () => window.electronAPI.close());

    window.electronAPI.onMaximizedState?.((isMax) => {
      if (btnMax) btnMax.textContent = isMax ? '❐' : '▢';
    });
  } else {
    // Web browser fallback
    if (btnMin) btnMin.style.display = 'none';
    if (btnMax) btnMax.style.display = 'none';
    if (btnClose) btnClose.style.display = 'none';
  }
}

/* ==========================================================================
   Theme Management (Dark / Light)
   ========================================================================== */
function initTheme() {
  const btnToggle = document.getElementById('btn-theme-toggle');
  const themeIcon = document.getElementById('theme-icon');

  const savedTheme = localStorage.getItem('theme') || 'dark';
  applyTheme(savedTheme);

  btnToggle?.addEventListener('click', () => {
    const newTheme = state.theme === 'dark' ? 'light' : 'dark';
    applyTheme(newTheme);
  });
}

function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);

  const themeIcon = document.getElementById('theme-icon');
  if (themeIcon) {
    themeIcon.textContent = theme === 'dark' ? '🌙' : '☀️';
  }
}

/* ==========================================================================
   Navigation Tabs
   ========================================================================== */
function initNavigationTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');
      state.activeTab = targetId;
    });
  });
}

/* ==========================================================================
   Drawer Tabs
   ========================================================================== */
function initDrawerTabs() {
  const tabs = document.querySelectorAll('.drawer-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.drawer-pane').forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.getAttribute('data-dtab');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');
    });
  });
}

/* ==========================================================================
   Toast Notifications
   ========================================================================== */
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  
  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️';
  toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

/* ==========================================================================
   System Stats & Health
   ========================================================================== */
async function fetchSystemStats() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) return;
    const data = await res.json();

    // CPU & RAM
    const cpuEl = document.getElementById('cpu-stat');
    const ramEl = document.getElementById('ram-stat');
    const vramEl = document.getElementById('vram-stat');
    const vramDivider = document.getElementById('vram-divider');

    if (cpuEl && data.system) cpuEl.textContent = `CPU: ${data.system.cpu_percent}%`;
    if (ramEl && data.system) ramEl.textContent = `RAM: ${data.system.ram_used_gb}/${data.system.ram_total_gb} GB (${data.system.ram_percent}%)`;

    // GPU Computation Top Display & VRAM
    const gpuBadge = document.getElementById('gpu-compute-badge');
    const gpuBadgeText = document.getElementById('gpu-badge-text');
    const gpuBadgeIcon = document.getElementById('gpu-badge-icon');

    if (data.gpu_computation) {
      const gpu = data.gpu_computation;
      if (gpuBadgeText) {
        let text = gpu.badge_text || 'GPU: Active';
        text = text.replace(/^[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}⚡💻\s]+/u, '').trim();
        gpuBadgeText.textContent = text;
      }

      if (gpuBadge) {
        gpuBadge.className = `badge badge-gpu badge-gpu-${gpu.badge_mode || 'cpu'}`;
        gpuBadge.title = `${gpu.device_name} • Backend: ${gpu.active_backend} • ${gpu.driver_version}`;
      }

      if (gpuBadgeIcon) {
        gpuBadgeIcon.textContent = gpu.badge_mode === 'cpu' ? '💻' : '⚡';
      }

      if (vramEl) {
        if (gpu.has_discrete_gpu && gpu.vram_total_gb > 0) {
          vramEl.textContent = `VRAM: ${gpu.vram_used_gb}/${gpu.vram_total_gb} GB`;
          vramEl.style.display = 'inline';
          if (vramDivider) vramDivider.style.display = 'inline';
        } else {
          vramEl.style.display = 'none';
          if (vramDivider) vramDivider.style.display = 'none';
        }
      }

      // Update Drawer GPU Hardware Card
      const drawerGpuName = document.getElementById('drawer-gpu-name');
      const drawerGpuBackend = document.getElementById('drawer-gpu-backend');
      const drawerVram = document.getElementById('drawer-vram-usage');
      const drawerDriver = document.getElementById('drawer-gpu-driver');
      const gpuStatusPill = document.getElementById('gpu-status-pill');

      const category = gpu.device_category || (gpu.has_discrete_gpu ? 'Discrete GPU' : (gpu.has_integrated_gpu ? 'Integrated GPU' : 'CPU'));
      if (drawerGpuName) drawerGpuName.textContent = `${gpu.device_name || 'Generic CPU'} [${category}]`;
      if (drawerGpuBackend) drawerGpuBackend.textContent = data.active_compute_backend || gpu.active_backend || 'CPU';
      if (drawerVram) drawerVram.textContent = gpu.has_discrete_gpu ? `${gpu.vram_used_gb} / ${gpu.vram_total_gb} GB (${gpu.vram_percent}%)` : `Shared Memory (CPU Multi-Threaded)`;
      if (drawerDriver) drawerDriver.textContent = `${gpu.driver_version} • ${gpu.cpu_simd}`;
      if (gpuStatusPill) {
        if (data.model_loaded && data.active_compute_backend && !data.active_compute_backend.includes('CPU')) {
          gpuStatusPill.textContent = 'GPU Active';
        } else if (gpu.has_discrete_gpu) {
          gpuStatusPill.textContent = 'Discrete GPU Ready';
        } else if (gpu.has_integrated_gpu) {
          gpuStatusPill.textContent = 'CPU (Optimized)';
        } else {
          gpuStatusPill.textContent = 'CPU Mode';
        }
      }
    }

    // Status Dot
    const dot = document.getElementById('model-status-dot');
    if (dot) {
      if (data.model_loaded) {
        dot.classList.add('online');
      } else {
        dot.classList.remove('online');
      }
    }

    // TurboQuant Badge
    const tqBadge = document.getElementById('turbo-badge-text');
    if (tqBadge && data.turbo_quant) {
      tqBadge.textContent = data.turbo_quant.enabled
        ? `TurboQuant: ${data.turbo_quant.ratio} Active`
        : 'TurboQuant: Disabled';
    }

    // Update Drawer Model Status Box
    const drawerModel = document.getElementById('drawer-model-name');
    const drawerTq = document.getElementById('drawer-turbo-status');
    if (drawerModel) {
      drawerModel.textContent = data.model_loaded ? data.model_name : 'No Model Loaded';
      drawerModel.title = data.model_loaded ? data.model_name : '';
    }
    if (drawerTq && data.turbo_quant) {
      drawerTq.textContent = data.turbo_quant.enabled
        ? `INT${data.turbo_quant.bits} (${data.turbo_quant.ratio} Active)`
        : 'Disabled';
    }

    state.currentModel = data.model_name;
  } catch (err) {
    // Backend still starting up
  }
}

/* ==========================================================================
   Model Loading, Browsing & Scanning
   ========================================================================== */
async function fetchModelsList() {
  try {
    const res = await fetch(`${API_BASE}/v1/models`);
    if (!res.ok) return;
    const data = await res.json();

    const select = document.getElementById('model-select');
    if (!select) return;

    const models = data.models || data.data || [];
    select.innerHTML = '<option value="">⚡ Select a Model to Load...</option>';
    if (models && models.length > 0) {
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.path;
        opt.textContent = `${m.name} (${m.size_gb} GB)`;
        opt.dataset.displayName = m.name;
        if (data.active_model && data.active_model.is_loaded && data.active_model.path === m.path) {
          opt.selected = true;
        }
        select.appendChild(opt);
      });
    }
  } catch (err) {
    console.error('Error fetching models:', err);
  }
}

function initHardwareControls() {
  const modelSelect = document.getElementById('model-select');
  const btnBrowse = document.getElementById('btn-browse-model');
  const btnScan = document.getElementById('btn-scan-models');
  const btnUnload = document.getElementById('btn-unload-model');
  const btnCancelLoading = document.getElementById('btn-cancel-loading');
  const computeModeSelect = document.getElementById('select-compute-mode');
  const gpuSlider = document.getElementById('slider-gpulayers');
  const valGpuLayers = document.getElementById('val-gpulayers');

  computeModeSelect?.addEventListener('change', (e) => {
    const mode = e.target.value;
    if (mode === 'cpu') {
      if (gpuSlider) gpuSlider.value = 0;
      if (valGpuLayers) valGpuLayers.textContent = 'CPU Only (0)';
    } else {
      if (gpuSlider && gpuSlider.value === '0') {
        gpuSlider.value = -1;
        if (valGpuLayers) valGpuLayers.textContent = 'All (-1)';
      }
    }
  });

  btnCancelLoading?.addEventListener('click', () => {
    const modal = document.getElementById('loading-modal');
    if (modal) modal.classList.remove('active');
  });

  btnUnload?.addEventListener('click', async () => {
    try {
      await fetch(`${API_BASE}/v1/models/unload`, { method: 'POST' });
      showToast('Model unloaded from memory.', 'info');
      fetchSystemStats();
      fetchModelsList();
    } catch (err) {
      showToast('Unload failed: ' + err.message, 'error');
    }
  });

  modelSelect?.addEventListener('change', (e) => {
    const fpath = e.target.value;
    const opt = e.target.options[e.target.selectedIndex];
    const displayName = opt ? opt.dataset.displayName : null;
    if (fpath) loadModelPath(fpath, displayName);
  });

  const hiddenModelInput = document.getElementById('hidden-model-file-input');
  hiddenModelInput?.addEventListener('change', async (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      const fpath = file.path || file.name;
      if (fpath) {
        await loadModelPath(fpath);
        fetchModelsList();
      }
    }
  });

  btnBrowse?.addEventListener('click', async () => {
    try {
      if (window.electronAPI && typeof window.electronAPI.selectFile === 'function') {
        const fpath = await window.electronAPI.selectFile({
          title: 'Select GGUF Model File',
          filters: [
            { name: 'GGUF Models (*.gguf, *.bin)', extensions: ['gguf', 'bin', 'GGUF', 'BIN'] },
            { name: 'All Files (*.*)', extensions: ['*'] }
          ]
        });
        if (fpath) {
          await loadModelPath(fpath);
          fetchModelsList();
          return;
        }
      }
    } catch (err) {
      console.warn('[Model Selector] Electron selectFile failed, falling back to input:', err);
    }
    hiddenModelInput?.click();
  });

  btnScan?.addEventListener('click', () => {
    openScannerModal();
  });

  // TurboQuant Config Controls
  const chkTq = document.getElementById('chk-tq-enable');
  const selectTqBits = document.getElementById('select-tq-bits');
  const chkHadamard = document.getElementById('chk-tq-hadamard');
  const sliderSparsity = document.getElementById('slider-tq-sparsity');

  const updateTQ = async () => {
    const enabled = chkTq?.checked ?? true;
    const bits = parseInt(selectTqBits?.value || '4');
    const hadamard = chkHadamard?.checked ?? true;
    const sparsity = parseInt(sliderSparsity?.value || '20') / 100.0;

    try {
      await fetch(`${API_BASE}/v1/turboquant/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled, bits, hadamard, sparsity })
      });
      fetchSystemStats();
    } catch (err) {
      console.error('Error updating TurboQuant:', err);
    }
  };

  chkTq?.addEventListener('change', updateTQ);
  selectTqBits?.addEventListener('change', updateTQ);
  chkHadamard?.addEventListener('change', updateTQ);
  sliderSparsity?.addEventListener('input', (e) => {
    const valEl = document.getElementById('val-tq-sparsity');
    if (valEl) valEl.textContent = `${e.target.value}%`;
    updateTQ();
  });
}

async function loadModelPath(fpath, displayName = null) {
  const modal = document.getElementById('loading-modal');
  const title = document.getElementById('loading-modal-title');
  const subtitle = document.getElementById('loading-modal-subtitle');

  let cleanName = displayName;
  if (!cleanName) {
    const base = fpath.split(/[\\/]/).pop();
    cleanName = base.startsWith('sha256-') ? `Ollama Model (${base.slice(7, 17)}...)` : base;
  }
  if (cleanName.length > 38) {
    cleanName = cleanName.slice(0, 35) + '...';
  }

  if (title) title.textContent = `Loading ${cleanName}...`;
  if (subtitle) subtitle.textContent = 'Allocating GPU VRAM / TurboQuant KV Cache Buffers...';
  if (modal) modal.classList.add('active');

  const computeMode = document.getElementById('select-compute-mode')?.value || 'auto';
  const gpuLayers = parseInt(document.getElementById('slider-gpulayers')?.value || '-1');
  const ctx = parseInt(document.getElementById('select-ctx')?.value || '4096');
  const flashAttn = document.getElementById('chk-flash-attn')?.checked ?? true;
  const tqBits = parseInt(document.getElementById('select-tq-bits')?.value || '4');
  const tqHadamard = document.getElementById('chk-tq-hadamard')?.checked ?? true;
  const tqSparsity = parseInt(document.getElementById('slider-tq-sparsity')?.value || '20') / 100.0;

  try {
    const res = await fetch(`${API_BASE}/v1/models/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_path: fpath,
        compute_mode: computeMode,
        n_gpu_layers: gpuLayers,
        n_ctx: ctx,
        flash_attn: flashAttn,
        turbo_bits: tqBits,
        turbo_hadamard: tqHadamard,
        turbo_sparsity: tqSparsity,
      })
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Failed to load model.');
    }

    const computeBackend = data.active_compute_backend || 'Hardware Accelerated';
    showToast(`Loaded ${data.model_name}! [${computeBackend}]`, 'success');
    fetchSystemStats();
  } catch (err) {
    showToast(`Model Load Error: ${err.message}`, 'error');
  } finally {
    if (modal) modal.classList.remove('active');
  }
}

/* ==========================================================================
   Model Scanner Modal
   ========================================================================== */
function openScannerModal() {
  const modal = document.getElementById('scanner-modal');
  if (!modal) return;
  modal.classList.add('active');

  const btnClose = document.getElementById('btn-close-scanner');
  if (btnClose && !btnClose.dataset.bound) {
    btnClose.dataset.bound = 'true';
    btnClose.addEventListener('click', () => modal.classList.remove('active'));
  }

  const btnRefresh = document.getElementById('btn-run-full-scan');
  if (btnRefresh && !btnRefresh.dataset.bound) {
    btnRefresh.dataset.bound = 'true';
    btnRefresh.addEventListener('click', () => triggerSystemScan(false, null));
  }

  const btnCustom = document.getElementById('btn-scan-custom-folder');
  if (btnCustom && !btnCustom.dataset.bound) {
    btnCustom.dataset.bound = 'true';
    btnCustom.addEventListener('click', async () => {
      try {
        if (window.electronAPI && typeof window.electronAPI.selectDirectory === 'function') {
          const selectedDir = await window.electronAPI.selectDirectory({
            title: 'Select Folder or Drive Containing GGUF Models'
          });
          if (selectedDir) {
            triggerSystemScan(false, selectedDir);
            return;
          }
        }
      } catch (e) {
        console.warn('Folder selection error:', e);
      }
      showToast('Please select a valid folder on your computer to scan.', 'info');
    });
  }

  const btnDeep = document.getElementById('btn-deep-drive-scan');
  if (btnDeep && !btnDeep.dataset.bound) {
    btnDeep.dataset.bound = 'true';
    btnDeep.addEventListener('click', () => triggerSystemScan(true, null));
  }

  modal.onclick = (e) => {
    if (e.target === modal) modal.classList.remove('active');
  };

  triggerSystemScan(false, null);
}

async function triggerSystemScan(deep = false, customPath = null) {
  const countLbl = document.getElementById('scan-count-lbl');
  const tbody = document.querySelector('#scanned-models-table tbody');
  const btnRefresh = document.getElementById('btn-run-full-scan');
  const btnCustom = document.getElementById('btn-scan-custom-folder');
  const btnDeep = document.getElementById('btn-deep-drive-scan');

  const disableAll = (disabled) => {
    if (btnRefresh) btnRefresh.disabled = disabled;
    if (btnCustom) btnCustom.disabled = disabled;
    if (btnDeep) btnDeep.disabled = disabled;
  };

  disableAll(true);
  let statusText = 'Scanning system drives & LLM directories...';
  if (customPath) {
    statusText = `Scanning custom folder: ${customPath}...`;
  } else if (deep) {
    statusText = 'Performing deep drive scan across all disks (C:, D:, etc.)...';
  }

  if (countLbl) countLbl.textContent = 'Scanning in progress...';
  if (tbody) tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 28px; color: var(--text-muted);"><span class="spin-loader">⚡</span> ${escapeHtml(statusText)}</td></tr>`;

  try {
    let url = `${API_BASE}/v1/models/scan`;
    const params = new URLSearchParams();
    if (deep) params.append('deep', 'true');
    if (customPath) params.append('custom_path', customPath);
    if (params.toString()) {
      url += `?${params.toString()}`;
    }

    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ custom_path: customPath, deep: Boolean(deep) })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const models = data.models || data.data || [];
    const totalCount = typeof data.count === 'number' ? data.count : models.length;

    if (countLbl) countLbl.textContent = `${totalCount} models found`;
    if (!tbody) return;

    if (models.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 24px; color: var(--text-faint);">
        No GGUF models found in scanned locations.<br>
        <small style="margin-top: 8px; display: inline-block;">Tip: Click <b>"📂 Scan Specific Folder"</b> to pick any drive/folder where your models are stored.</small>
      </td></tr>`;
      return;
    }

    tbody.innerHTML = '';
    models.forEach(m => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><b>${escapeHtml(m.name)}</b></td>
        <td>${m.size_gb} GB</td>
        <td><span class="badge badge-cyan">${escapeHtml(m.architecture || 'GGUF')}</span></td>
        <td style="color: var(--text-faint); font-size: 11px;" title="${escapeHtml(m.path)}">${escapeHtml(m.source || 'Drive')}</td>
        <td><button class="btn btn-primary btn-sm load-scanned-btn" data-path="${escapeHtml(m.path)}" data-name="${escapeHtml(m.name)}">⚡ Load</button></td>
      `;
      tbody.appendChild(tr);
    });

    document.querySelectorAll('.load-scanned-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const targetBtn = e.currentTarget || e.target.closest('[data-path]');
        const path = targetBtn?.getAttribute('data-path');
        const name = targetBtn?.getAttribute('data-name');
        if (path) {
          document.getElementById('scanner-modal')?.classList.remove('active');
          await loadModelPath(path, name);
          await fetchModelsList();
        }
      });
    });

    // Also refresh sidebar dropdown
    await fetchModelsList();
  } catch (err) {
    console.error('System scan error:', err);
    if (countLbl) countLbl.textContent = 'Scan failed';
    if (tbody) tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 20px; color: var(--accent-red);">Scan error: ${escapeHtml(err.message)}</td></tr>`;
    showToast('Model scan failed: ' + err.message, 'error');
  } finally {
    disableAll(false);
  }
}

/* ==========================================================================
   Chat Sessions Management
   ========================================================================== */
async function fetchSessions() {
  try {
    const res = await fetch(`${API_BASE}/v1/sessions`);
    if (!res.ok) return;
    const data = await res.json();

    state.sessions = data.sessions || [];
    state.activeSessionId = data.active_id || null;
    renderSessionList();

    if (state.activeSessionId) {
      await loadActiveSession();
    } else {
      // No sessions at all — clear the chat view and re-enable input
      const container = document.getElementById('chat-messages');
      if (container) container.innerHTML = '';
      enableChatInput(true);
    }
  } catch (err) {
    console.error('Error fetching sessions:', err);
  } finally {
    enableChatInput(false);
  }
}

function enableChatInput(clearText = false) {
  state.isGenerating = false;
  const textarea = document.getElementById('chat-textarea');
  const sendBtn = document.getElementById('btn-send-chat');
  const stopBtn = document.getElementById('btn-stop-chat');

  if (sendBtn) sendBtn.classList.remove('hidden');
  if (stopBtn) stopBtn.classList.add('hidden');

  if (textarea) {
    textarea.disabled = false;
    textarea.readOnly = false;
    textarea.removeAttribute('disabled');
    textarea.removeAttribute('readonly');
    textarea.style.pointerEvents = 'auto';
    textarea.style.cursor = 'text';
    if (clearText) {
      textarea.value = '';
      textarea.style.height = 'auto';
    }
    // Multi-stage focus to ensure keyboard input is immediately captured
    try { textarea.focus(); } catch (e) {}
    requestAnimationFrame(() => {
      try { textarea.focus(); } catch (e) {}
    });
    setTimeout(() => {
      try { textarea.focus(); } catch (e) {}
    }, 40);
  }
}

function renderSessionList() {
  const container = document.getElementById('session-list');
  const countEl = document.getElementById('chat-count');
  if (!container) return;

  if (countEl) countEl.textContent = state.sessions.length;
  container.innerHTML = '';

  if (state.sessions.length === 0) {
    const emptyEl = document.createElement('div');
    emptyEl.className = 'empty-session-placeholder';
    emptyEl.style.cssText = 'padding: 20px 12px; text-align: center; color: var(--text-faint); font-size: 11px; line-height: 1.5;';
    emptyEl.innerHTML = '<span>No saved chats.</span><br><span style="font-size: 10px; color: var(--primary);">Click <b>+ New Chat</b> to begin.</span>';
    container.appendChild(emptyEl);
    return;
  }

  state.sessions.forEach(s => {
    const item = document.createElement('div');
    item.className = `session-item ${s.id === state.activeSessionId ? 'active' : ''}`;
    item.innerHTML = `
      <span class="session-title-text">💬 ${escapeHtml(s.title || 'Conversation')}</span>
      <button class="btn-icon del-session-btn" data-id="${s.id}" title="Delete chat">✕</button>
    `;

    item.addEventListener('click', async (e) => {
      if (e.target.closest('.del-session-btn')) return;
      state.activeSessionId = s.id;
      renderSessionList();
      await loadActiveSession();
      enableChatInput(false);
    });

    container.appendChild(item);
  });

  // Attach delete listeners
  document.querySelectorAll('.del-session-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = e.target.closest('[data-id]')?.getAttribute('data-id') || e.target.getAttribute('data-id');
      if (!id) return;
      if (confirm('Delete this conversation permanently?')) {
        setGeneratingState(false);
        try {
          await fetch(`${API_BASE}/v1/sessions/${id}`, { method: 'DELETE' });
        } catch (err) {
          console.error('Error deleting session:', err);
        }
        await fetchSessions();
        enableChatInput(true);
        if (window.focus) window.focus();
      }
    });
  });
}

async function loadActiveSession() {
  if (!state.activeSessionId) {
    const container = document.getElementById('chat-messages');
    if (container) container.innerHTML = '';
    enableChatInput(false);
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/v1/sessions/${state.activeSessionId}`);
    if (!res.ok) return;
    const session = await res.json();

    // Set System Prompt Input
    const sysInput = document.getElementById('sys-prompt-input');
    if (sysInput) sysInput.value = session.system_prompt || '';

    // Render Messages
    renderMessages(session.messages || []);
  } catch (err) {
    console.error('Error loading active session:', err);
  } finally {
    enableChatInput(false);
  }
}

/* ==========================================================================
   Reasoning & Thinking Process Parser (Outside Chat Bubble)
   ========================================================================== */
function extractThinkingAndResponse(text) {
  let thinking = '';
  let response = text || '';
  let isThinkingActive = false;

  if (!text) return { thinking: '', response: '', isThinkingActive: false };

  // 1. Check for standard closed tags (<think>...</think>, <thought>...</thought>, <reasoning>...</reasoning>, [THOUGHT]...[/THOUGHT])
  const closedTagRegex = /<(?:think|thought|reasoning|\|thought\|)>([\s\S]*?)<\/(?:think|thought|reasoning|\|thought\|)>/i;
  const squareClosedRegex = /\[(?:THOUGHT|THINK|REASONING)\]([\s\S]*?)\[\/(?:THOUGHT|THINK|REASONING)\]/i;

  let closedMatch = text.match(closedTagRegex) || text.match(squareClosedRegex);
  if (closedMatch) {
    thinking = closedMatch[1].trim();
    response = text.replace(closedMatch[0], '').trim();
    return { thinking, response, isThinkingActive: false };
  }

  // 2. Check for active streaming open tags (<think>..., <thought>..., <reasoning>..., [THOUGHT]...)
  const openTagRegex = /<(?:think|thought|reasoning|\|thought\|)>([\s\S]*)$/i;
  const squareOpenRegex = /\[(?:THOUGHT|THINK|REASONING)\]([\s\S]*)$/i;

  let openMatch = text.match(openTagRegex) || text.match(squareOpenRegex);
  if (openMatch) {
    thinking = openMatch[1].trim();
    response = text.substring(0, openMatch.index).trim();
    return { thinking, response, isThinkingActive: true };
  }

  // 3. Check for Markdown-style thinking headers (e.g. "### Thinking Process\n...\n### Response\n...")
  const mdThinkingMatch = text.match(/(?:^|\n)(?:###?\s*(?:Thinking Process|Reasoning|Thoughts?):\s*\n)([\s\S]*?)(?=(?:\n###?\s*(?:Response|Answer|Final Answer):\s*\n)|$)/i);
  if (mdThinkingMatch) {
    thinking = mdThinkingMatch[1].trim();
    response = text.replace(mdThinkingMatch[0], '').replace(/^###?\s*(?:Response|Answer|Final Answer):\s*\n/i, '').trim();
    return { thinking, response, isThinkingActive: false };
  }

  return { thinking, response, isThinkingActive: false };
}

function buildThinkingHtml(thinking, isThinkingActive) {
  if (!thinking && !isThinkingActive) return '';

  const titleText = isThinkingActive ? '💭 Thinking...' : '🧠 Thought Process';
  const bubbleClass = isThinkingActive ? 'thinking-bubble streaming' : 'thinking-bubble';
  const toggleLabel = isThinkingActive ? '···' : 'Collapse';

  return `
    <div class="${bubbleClass}">
      <div class="thinking-header" onclick="window.toggleThinkingBox(this)">
        <div class="thinking-title">
          <span class="thinking-icon">🧠</span>
          <span>${titleText}</span>
        </div>
        <div class="thinking-toggle-btn">
          <span class="toggle-text">${toggleLabel}</span>
          <span class="toggle-chevron">▼</span>
        </div>
      </div>
      <div class="thinking-body">${escapeHtml(thinking || '')}</div>
    </div>
  `;
}

window.toggleThinkingBox = function(headerEl) {
  const box = headerEl.closest('.thinking-bubble');
  if (!box) return;
  const isCollapsed = box.classList.toggle('collapsed');
  const toggleText = box.querySelector('.toggle-text');
  if (toggleText) {
    toggleText.textContent = isCollapsed ? 'Expand' : 'Collapse';
  }
};

function renderMessages(messages) {
  const container = document.getElementById('chat-messages');
  if (!container) return;

  container.innerHTML = '';
  messages.forEach(msg => {
    appendMessageElement(msg.role, msg.content, msg.metrics);
  });

  scrollChatToBottom();
}

function appendMessageElement(role, content, metrics = null) {
  const container = document.getElementById('chat-messages');
  if (!container) return;

  const msgDiv = document.createElement('div');
  msgDiv.className = `chat-msg ${role}`;

  const avatar = role === 'user' ? '👤' : '⚡';

  const { thinking, response, isThinkingActive } = extractThinkingAndResponse(content);

  // Build thinking bubble (separate, compact, collapsible)
  const thinkingHtml = (role === 'assistant') ? buildThinkingHtml(thinking, isThinkingActive) : '';

  // Build response bubble
  let responseBubbleHtml = '';
  if (role === 'user') {
    const rawHtml = marked.parse(content || '');
    responseBubbleHtml = `<div class="chat-bubble markdown-body">${rawHtml}</div>`;
  } else {
    // Only show response bubble when there is actual response text
    const displayResponse = response || (isThinkingActive ? '' : (content || '...' ));
    if (displayResponse) {
      const rawHtml = marked.parse(displayResponse);
      responseBubbleHtml = `<div class="chat-bubble markdown-body">${rawHtml}</div>`;
    }
  }

  let metricsHtml = '';
  if (metrics && role === 'assistant') {
    const tokS = metrics.tok_per_sec || 0;
    const tokens = metrics.tokens || 0;
    const tq = metrics.turbo_ratio || '4.0x';
    metricsHtml = `
      <div class="chat-meta">
        <span>Generated ${tokens} tokens @ ${tokS.toFixed(1)} tok/s</span>
        <span>|</span>
        <span class="badge badge-cyan" style="padding: 1px 6px; font-size: 9px;">TurboQuant ${tq}</span>
        <button class="copy-btn" onclick="navigator.clipboard.writeText(this.closest('.chat-msg-body').querySelector('.chat-bubble')?.innerText || ''); showToast('Copied to clipboard!', 'success');">📋 Copy Response</button>
      </div>
    `;
  }

  msgDiv.innerHTML = `
    <div class="chat-avatar">${avatar}</div>
    <div class="chat-msg-body">
      ${thinkingHtml}
      ${responseBubbleHtml}
      ${metricsHtml}
    </div>
  `;

  // Apply syntax highlighting
  msgDiv.querySelectorAll('pre code').forEach((block) => {
    hljs.highlightElement(block);
    const pre = block.parentElement;
    if (!pre.querySelector('.copy-code-btn')) {
      const copyBtn = document.createElement('button');
      copyBtn.className = 'btn btn-secondary btn-sm copy-code-btn';
      copyBtn.style.cssText = 'position: absolute; top: 6px; right: 6px; font-size: 10px; padding: 2px 6px;';
      copyBtn.textContent = '📋 Copy';
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(block.innerText);
        copyBtn.textContent = '✅ Copied';
        setTimeout(() => copyBtn.textContent = '📋 Copy', 2000);
      };
      pre.style.position = 'relative';
      pre.appendChild(copyBtn);
    }
  });

  container.appendChild(msgDiv);
  return msgDiv;
}

function scrollChatToBottom() {
  const container = document.getElementById('chat-messages');
  if (container) {
    container.scrollTop = container.scrollHeight;
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/* ==========================================================================
   Chat Playground Actions (SSE Stream)
   ========================================================================== */
function setGeneratingState(generating) {
  state.isGenerating = generating;
  const sendBtn = document.getElementById('btn-send-chat');
  const stopBtn = document.getElementById('btn-stop-chat');
  const textarea = document.getElementById('chat-textarea');

  if (sendBtn) sendBtn.classList.toggle('hidden', generating);
  if (stopBtn) stopBtn.classList.toggle('hidden', !generating);
  if (textarea) {
    textarea.disabled = false;
    textarea.removeAttribute('disabled');
    if (!generating) {
      setTimeout(() => textarea.focus(), 50);
    }
  }
}

function initChatPlayground() {
  const sendBtn = document.getElementById('btn-send-chat');
  const stopBtn = document.getElementById('btn-stop-chat');
  const clearBtn = document.getElementById('btn-clear-chat');
  const textarea = document.getElementById('chat-textarea');
  const newChatBtn = document.getElementById('btn-new-chat');
  const sysPreset = document.getElementById('sys-preset-select');
  const sysInput = document.getElementById('sys-prompt-input');

  // Auto-resize textarea
  textarea?.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
  });

  textarea?.addEventListener('focus', () => {
    textarea.disabled = false;
    textarea.readOnly = false;
    textarea.removeAttribute('disabled');
    textarea.removeAttribute('readonly');
  });

  textarea?.addEventListener('click', () => {
    enableChatInput(false);
  });

  // Clicking anywhere inside input box activates textarea
  document.querySelector('.chat-input-box')?.addEventListener('click', (e) => {
    if (e.target.tagName !== 'BUTTON' && !e.target.closest('button')) {
      enableChatInput(false);
    }
  });

  textarea?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!state.isGenerating) {
        sendChatMessage();
      }
    }
  });

  sendBtn?.addEventListener('click', () => {
    if (!state.isGenerating) {
      sendChatMessage();
    }
  });

  stopBtn?.addEventListener('click', () => {
    if (state.abortController) {
      state.abortController.abort();
    }
    fetch(`${API_BASE}/v1/generation/stop`, { method: 'POST' });
    setGeneratingState(false);
    enableChatInput(false);
  });

  clearBtn?.addEventListener('click', async () => {
    if (state.abortController) {
      state.abortController.abort();
    }
    setGeneratingState(false);
    if (state.activeSessionId) {
      try {
        await fetch(`${API_BASE}/v1/sessions/${state.activeSessionId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: [] })
        });
      } catch (err) {
        console.error('Error clearing conversation messages:', err);
      }
      await loadActiveSession();
    } else {
      const container = document.getElementById('chat-messages');
      if (container) container.innerHTML = '';
    }
    enableChatInput(true);
    showToast('Conversation cleared', 'info');
  });

  newChatBtn?.addEventListener('click', async () => {
    if (state.abortController) {
      state.abortController.abort();
    }
    setGeneratingState(false);
    try {
      const res = await fetch(`${API_BASE}/v1/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New Conversation' })
      });
      const session = await res.json();
      state.activeSessionId = session.id;
      await fetchSessions();
      await loadActiveSession();
    } catch (err) {
      console.error('Error creating new conversation:', err);
    }
    enableChatInput(true);
  });

  // Wire sidebar footer Delete button
  const deleteChatBtn = document.getElementById('btn-delete-chat');
  deleteChatBtn?.addEventListener('click', async () => {
    if (state.abortController) {
      state.abortController.abort();
    }
    setGeneratingState(false);
    if (state.activeSessionId) {
      try {
        await fetch(`${API_BASE}/v1/sessions/${state.activeSessionId}`, { method: 'DELETE' });
      } catch (err) {
        console.error('Error deleting current session:', err);
      }
    }
    await fetchSessions();
    enableChatInput(true);
    showToast('Conversation deleted', 'info');
  });

  sysPreset?.addEventListener('change', (e) => {
    const prompt = SYSTEM_PRESETS[e.target.value] || '';
    if (sysInput) sysInput.value = prompt;
    saveSystemPrompt(prompt);
  });

  sysInput?.addEventListener('change', (e) => {
    saveSystemPrompt(e.target.value);
  });
}

async function saveSystemPrompt(prompt) {
  if (!state.activeSessionId) return;
  try {
    await fetch(`${API_BASE}/v1/sessions/${state.activeSessionId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ system_prompt: prompt })
    });
  } catch (e) {
    // Ignore silent save error
  }
}

async function sendChatMessage() {
  const textarea = document.getElementById('chat-textarea');
  const prompt = textarea?.value?.trim();
  if (!prompt || state.isGenerating) return;

  // Add User Message to UI
  appendMessageElement('user', prompt);
  textarea.value = '';
  textarea.style.height = 'auto';
  scrollChatToBottom();

  const defaultThinking = 'Formulating structured reasoning and analyzing query parameters with Google TurboQuant KV acceleration...';
  let capturedThinking = defaultThinking;
  let responseText = '';
  let fullResponse = `<think>${defaultThinking}</think>\n\n`;

  // Create Active Assistant Bubble with thinking bubble visible
  const assistantBubble = appendMessageElement('assistant', fullResponse);
  scrollChatToBottom();

  setGeneratingState(true);
  state.abortController = new AbortController();

  let isReasoningStreamActive = false;
  let customThinkingReceived = false;
  const startTime = performance.now();
  let firstTokenTime = null;
  let tokenCount = 0;

  try {
    // Ensure active session exists without destroying active streaming DOM
    if (!state.activeSessionId) {
      try {
        const createRes = await fetch(`${API_BASE}/v1/sessions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: prompt.substring(0, 32) })
        });
        const createdSession = await createRes.json();
        state.activeSessionId = createdSession.id;
        state.sessions.unshift({ id: createdSession.id, title: createdSession.title || prompt.substring(0, 32) });
        renderSessionList();
      } catch (e) {
        console.warn('Auto-session create warning:', e);
      }
    }

    // Retrieve active session history
    let history = [];
    const sysPrompt = document.getElementById('sys-prompt-input')?.value || '';
    if (sysPrompt) history.push({ role: 'system', content: sysPrompt });

    let sessionData = { messages: [], title: 'New Conversation' };
    if (state.activeSessionId) {
      try {
        const currentSessionRes = await fetch(`${API_BASE}/v1/sessions/${state.activeSessionId}`);
        if (currentSessionRes.ok) {
          sessionData = await currentSessionRes.json();
          (sessionData.messages || []).forEach(m => history.push({ role: m.role, content: m.content }));
        }
      } catch (e) {
        console.warn('Session history fetch warning:', e);
      }
    }
    history.push({ role: 'user', content: prompt });

    const temp = parseFloat(document.getElementById('slider-temp')?.value || '70') / 100.0;
    const topP = parseFloat(document.getElementById('slider-topp')?.value || '95') / 100.0;
    const topK = parseInt(document.getElementById('slider-topk')?.value || '40');
    const minP = parseFloat(document.getElementById('slider-minp')?.value || '5') / 100.0;
    const repPen = parseFloat(document.getElementById('slider-reppen')?.value || '110') / 100.0;
    const maxTok = parseInt(document.getElementById('slider-maxtok')?.value || '2048');
    const grammarEnabled = document.getElementById('chk-grammar-enable')?.checked ?? false;
    const grammar = grammarEnabled ? (document.getElementById('grammar-select')?.value || 'json') : 'none';
    const customGbnf = grammarEnabled ? (document.getElementById('custom-gbnf-input')?.value || '') : '';

    const response = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: history,
        temperature: temp,
        top_p: topP,
        top_k: topK,
        min_p: minP,
        repeat_penalty: repPen,
        max_tokens: maxTok,
        stream: true,
        grammar_type: grammar,
        custom_grammar: customGbnf
      }),
      signal: state.abortController.signal
    });

    if (!response.ok) {
      const errJson = await response.json();
      throw new Error(errJson.detail || 'Inference error.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const dataStr = line.replace('data: ', '').trim();
          if (dataStr === '[DONE]') break;

          try {
            const parsed = JSON.parse(dataStr);
            const deltaObj = parsed.choices?.[0]?.delta || {};
            const content = deltaObj.content;
            const reasoning = deltaObj.reasoning_content || deltaObj.reasoning || deltaObj.thought;

            if (reasoning) {
              if (!customThinkingReceived) {
                capturedThinking = '';
                customThinkingReceived = true;
              }
              capturedThinking += reasoning;
              isReasoningStreamActive = true;
              tokenCount++;
              if (firstTokenTime === null) firstTokenTime = performance.now();
            } else if (content) {
              if (firstTokenTime === null) firstTokenTime = performance.now();
              tokenCount++;

              // Check if model natively generated <think>...</think> in content
              if (content.includes('<think>')) {
                customThinkingReceived = true;
                capturedThinking = '';
              }
              
              if (customThinkingReceived && !content.includes('</think>') && !isReasoningStreamActive && content.startsWith('<think>')) {
                capturedThinking += content.replace('<think>', '');
              } else {
                responseText += content;
              }
            }

            // Construct unified response containing thinking block + response
            const activeThinking = capturedThinking || defaultThinking;
            
            // Check if responseText already has <think> tags natively
            if (responseText.includes('<think>')) {
              fullResponse = responseText;
            } else {
              fullResponse = `<think>${activeThinking}</think>\n\n${responseText}`;
            }

            // Re-render active assistant message with streaming thoughts AND response
            updateAssistantMessage(assistantBubble, fullResponse);
            scrollChatToBottom();
          } catch (e) {
            // Ignore partial SSE JSON
          }
        }
      }
    }

    // Finalize metrics
    const endTime = performance.now();
    const totalElapsed = (endTime - startTime) / 1000.0;
    const genElapsed = firstTokenTime ? (endTime - firstTokenTime) / 1000.0 : totalElapsed;
    const tokPerSec = genElapsed > 0.05 ? (tokenCount / genElapsed) : 0.0;

    const metrics = {
      tokens: tokenCount,
      tok_per_sec: tokPerSec,
      elapsed_s: totalElapsed,
      turbo_ratio: '4.0x'
    };

    updateAssistantMessage(assistantBubble, fullResponse, metrics);

    // Save Messages into Session
    if (state.activeSessionId) {
      const newMessages = sessionData.messages || [];
      newMessages.push({ role: 'user', content: prompt });
      newMessages.push({ role: 'assistant', content: fullResponse, metrics });

      let title = sessionData.title || 'New Conversation';
      if ((newMessages.length === 2 || title === 'New Conversation') && prompt) {
        title = prompt.substring(0, 32);
      }

      await fetch(`${API_BASE}/v1/sessions/${state.activeSessionId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, messages: newMessages })
      });

      fetchSessions();
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      const bodyEl = assistantBubble.querySelector('.chat-msg-body');
      if (bodyEl) {
        bodyEl.innerHTML = `<div class="chat-bubble markdown-body"><span style="color:#ef4444;">⚠️ <b>Error:</b> ${escapeHtml(err.message)}</span></div>`;
      }
    }
  } finally {
    setGeneratingState(false);
    enableChatInput();
  }
}

function updateAssistantMessage(bubbleEl, rawContent, metrics = null) {
  const { thinking, response, isThinkingActive } = extractThinkingAndResponse(rawContent);

  const bodyEl = bubbleEl.querySelector('.chat-msg-body');
  if (!bodyEl) return;

  const thinkingHtml = buildThinkingHtml(thinking, isThinkingActive);

  // Response bubble — only shown when there is actual response text
  let responseBubbleHtml = '';
  if (response) {
    const rawHtml = marked.parse(response);
    responseBubbleHtml = `<div class="chat-bubble markdown-body">${rawHtml}</div>`;
  } else if (!isThinkingActive && !thinking) {
    // Pure non-thinking message: show raw content
    const rawHtml = marked.parse(rawContent || '...');
    responseBubbleHtml = `<div class="chat-bubble markdown-body">${rawHtml}</div>`;
  } else if (isThinkingActive) {
    // Still thinking — show empty response placeholder
    responseBubbleHtml = '';
  }

  let metricsHtml = '';
  if (metrics) {
    const tokS = metrics.tok_per_sec || 0;
    const tokens = metrics.tokens || 0;
    const tq = metrics.turbo_ratio || '4.0x';
    metricsHtml = `
      <div class="chat-meta">
        <span>Generated ${tokens} tokens @ ${tokS.toFixed(1)} tok/s</span>
        <span>|</span>
        <span class="badge badge-cyan" style="padding: 1px 6px; font-size: 9px;">TurboQuant ${tq}</span>
        <button class="copy-btn" onclick="navigator.clipboard.writeText(this.closest('.chat-msg-body').querySelector('.chat-bubble')?.innerText || ''); showToast('Copied response!', 'success');">📋 Copy Response</button>
      </div>
    `;
  }

  bodyEl.innerHTML = `
    ${thinkingHtml}
    ${responseBubbleHtml}
    ${metricsHtml}
  `;

  // Syntax highlight
  bodyEl.querySelectorAll('pre code').forEach((block) => {
    hljs.highlightElement(block);
  });
}

function setGeneratingState(generating) {
  state.isGenerating = generating;
  const sendBtn = document.getElementById('btn-send-chat');
  const stopBtn = document.getElementById('btn-stop-chat');

  if (sendBtn) sendBtn.classList.toggle('hidden', generating);
  if (stopBtn) stopBtn.classList.toggle('hidden', !generating);
}

/* ==========================================================================
   Raw Text Completion Playground
   ========================================================================== */
function initCompletionPlayground() {
  const templateSelect = document.getElementById('comp-template-select');
  const promptInput = document.getElementById('comp-prompt-input');
  const outputView = document.getElementById('comp-output-view');
  const runBtn = document.getElementById('btn-run-comp');
  const stopBtn = document.getElementById('btn-stop-comp');
  const clearBtn = document.getElementById('btn-clear-comp');
  const copyBtn = document.getElementById('btn-copy-comp');
  const metricsLbl = document.getElementById('comp-metrics-lbl');

  templateSelect?.addEventListener('change', (e) => {
    const t = SAMPLE_PROMPTS[e.target.value];
    if (t && promptInput) promptInput.value = t;
  });

  clearBtn?.addEventListener('click', () => {
    if (promptInput) promptInput.value = '';
    if (outputView) outputView.value = '';
    if (metricsLbl) metricsLbl.textContent = 'Cleared.';
  });

  copyBtn?.addEventListener('click', () => {
    if (outputView?.value) {
      navigator.clipboard.writeText(outputView.value);
      showToast('Copied completion output!', 'success');
    }
  });

  runBtn?.addEventListener('click', async () => {
    const prompt = promptInput?.value?.trim();
    if (!prompt) {
      showToast('Please enter an input prompt.', 'warning');
      return;
    }

    if (outputView) outputView.value = '';
    if (metricsLbl) metricsLbl.textContent = 'Generating completion...';
    runBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');

    try {
      const res = await fetch(`${API_BASE}/v1/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt,
          max_tokens: 1024,
          stream: true
        })
      });

      if (!res.ok) throw new Error('Completion error');
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      let tokenCount = 0;
      const start = performance.now();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const lines = decoder.decode(value).split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace('data: ', '').trim();
            if (dataStr === '[DONE]') break;
            try {
              const p = JSON.parse(dataStr);
              const text = p.choices?.[0]?.text;
              if (text && outputView) {
                tokenCount++;
                outputView.value += text;
                outputView.scrollTop = outputView.scrollHeight;
              }
            } catch (e) {}
          }
        }
      }

      const elapsed = (performance.now() - start) / 1000.0;
      const tokS = (tokenCount / elapsed).toFixed(1);
      if (metricsLbl) metricsLbl.textContent = `Completed: ${tokenCount} tokens @ ${tokS} tok/s`;
    } catch (err) {
      showToast('Completion failed: ' + err.message, 'error');
    } finally {
      runBtn.classList.remove('hidden');
      stopBtn.classList.add('hidden');
    }
  });

  stopBtn?.addEventListener('click', () => {
    fetch(`${API_BASE}/v1/generation/stop`, { method: 'POST' });
    runBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
  });
}

/* ==========================================================================
   GGUF Model Inspector
   ========================================================================== */
function initInspector() {
  const dropzone = document.getElementById('inspector-dropzone');
  const browseBtn = document.getElementById('btn-inspector-browse');
  const resultsDiv = document.getElementById('inspector-results');
  const loadBtn = document.getElementById('btn-inspector-load-model');

  let inspectedFilePath = null;

  const hiddenInspectorInput = document.getElementById('hidden-inspector-file-input');
  hiddenInspectorInput?.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      const fpath = file.path || file.name;
      if (fpath) inspectFile(fpath);
    }
  });

  browseBtn?.addEventListener('click', async () => {
    try {
      if (window.electronAPI && typeof window.electronAPI.selectFile === 'function') {
        const fpath = await window.electronAPI.selectFile({
          title: 'Select GGUF File to Inspect',
          filters: [
            { name: 'GGUF Models (*.gguf, *.bin)', extensions: ['gguf', 'bin', 'GGUF', 'BIN'] },
            { name: 'All Files (*.*)', extensions: ['*'] }
          ]
        });
        if (fpath) {
          inspectFile(fpath);
          return;
        }
      }
    } catch (err) {
      console.warn('[Inspector Selector] Electron selectFile failed, falling back to input:', err);
    }
    hiddenInspectorInput?.click();
  });

  loadBtn?.addEventListener('click', () => {
    if (inspectedFilePath) {
      loadModelPath(inspectedFilePath);
    }
  });

  dropzone?.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--primary)';
  });

  dropzone?.addEventListener('dragleave', () => {
    dropzone.style.borderColor = 'var(--border-color)';
  });

  dropzone?.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--border-color)';
    if (e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.path) {
        inspectFile(file.path);
      }
    }
  });

  async function inspectFile(fpath) {
    inspectedFilePath = fpath;
    try {
      const res = await fetch(`${API_BASE}/v1/models/inspect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: fpath })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Inspection failed.');

      if (resultsDiv) resultsDiv.classList.remove('hidden');

      document.getElementById('meta-arch').textContent = data.architecture || 'GGUF';
      document.getElementById('meta-params').textContent = data.total_parameters || 'N/A';
      document.getElementById('meta-quant').textContent = data.file_type || 'Q4_K_M';
      document.getElementById('meta-ctx').textContent = (data.context_length || 4096).toLocaleString();
      document.getElementById('meta-size').textContent = `${data.file_size_gb || 0} GB`;

      // TurboQuant Estimates
      const est = data.turboquant_estimates || {};
      const tqDiv = document.getElementById('tq-estimates-content');
      if (tqDiv) {
        tqDiv.innerHTML = `
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-label">RAW FP16 KV CACHE</div>
              <div class="metric-value">${est.raw_kv_cache_mb || 0} MB</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">TURBOQUANT INT4 KV</div>
              <div class="metric-value" style="color: #10b981;">${est.turboquant_kv_mb || 0} MB</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">KV MEMORY SAVED</div>
              <div class="metric-value" style="color: var(--primary);">${est.savings_pct || '75%'} (${est.compression_ratio || '4.0x'})</div>
            </div>
          </div>
        `;
      }

      // Metadata Table
      const tbody = document.querySelector('#metadata-table tbody');
      if (tbody) {
        tbody.innerHTML = '';
        const metaObj = data.metadata || {};
        for (const [k, v] of Object.entries(metaObj)) {
          const tr = document.createElement('tr');
          tr.innerHTML = `<td style="font-family:'JetBrains Mono'; font-size:10px;">${k}</td><td>${escapeHtml(String(v))}</td>`;
          tbody.appendChild(tr);
        }
      }
    } catch (err) {
      showToast('Inspector failed: ' + err.message, 'error');
    }
  }
}

/* ==========================================================================
   Hugging Face Model Store & Downloads
   ========================================================================== */
function initModelStore() {
  const searchInput = document.getElementById('hf-search-input');
  const searchBtn = document.getElementById('btn-hf-search');
  const btnPauseResume = document.getElementById('btn-dl-pause-resume');
  const btnCancel = document.getElementById('btn-dl-cancel');

  searchBtn?.addEventListener('click', performHFSearch);
  searchInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') performHFSearch();
  });

  // Global Active Tracker Pause / Resume
  btnPauseResume?.addEventListener('click', () => {
    if (!state.activeDownloadRepo || !state.activeDownloadFile) return;
    if (state.isDownloadPaused) {
      resumeDownload(state.activeDownloadRepo, state.activeDownloadFile, state.activeDownloadSafeKey);
    } else {
      pauseDownload(state.activeDownloadRepo, state.activeDownloadFile, state.activeDownloadSafeKey);
    }
  });

  // Global Active Tracker Stop / Cancel
  btnCancel?.addEventListener('click', () => {
    if (!state.activeDownloadRepo || !state.activeDownloadFile) return;
    stopDownload(state.activeDownloadRepo, state.activeDownloadFile, state.activeDownloadSafeKey);
  });
}

async function pauseDownload(repoId, filename, safeKey = null) {
  try {
    await fetch(`${API_BASE}/v1/downloader/pause`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id: repoId, filename })
    });
    state.isDownloadPaused = true;
    showToast(`Paused download for ${filename}`, 'info');

    const sKey = safeKey || state.activeDownloadSafeKey;
    const pill = document.getElementById('dl-status-pill');
    const btn = document.getElementById('btn-dl-pause-resume');
    if (pill) {
      pill.className = 'badge';
      pill.style.cssText = 'background: rgba(234, 179, 8, 0.2); color: #eab308; border: 1px solid rgba(234, 179, 8, 0.4);';
      pill.textContent = '⏸️ PAUSED';
    }
    if (btn) btn.textContent = '▶️ Resume';

    const inlinePauseBtn = document.getElementById(`inline-pause-${sKey}`);
    if (inlinePauseBtn) inlinePauseBtn.textContent = '▶️';
  } catch (err) {
    showToast('Error pausing download: ' + err.message, 'error');
  }
}

async function resumeDownload(repoId, filename, safeKey = null) {
  try {
    await fetch(`${API_BASE}/v1/downloader/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id: repoId, filename })
    });
    state.isDownloadPaused = false;
    showToast(`Resumed download for ${filename}`, 'info');

    const sKey = safeKey || state.activeDownloadSafeKey;
    const pill = document.getElementById('dl-status-pill');
    const btn = document.getElementById('btn-dl-pause-resume');
    if (pill) {
      pill.className = 'badge badge-cyan';
      pill.textContent = '⚡ DOWNLOADING';
    }
    if (btn) btn.textContent = '⏸️ Pause';

    const inlinePauseBtn = document.getElementById(`inline-pause-${sKey}`);
    if (inlinePauseBtn) inlinePauseBtn.textContent = '⏸️';

    if (!state.downloadPollInterval) {
      state.downloadPollInterval = setInterval(() => pollDownloadProgress(sKey, repoId, filename), 500);
    }
  } catch (err) {
    showToast('Error resuming download: ' + err.message, 'error');
  }
}

async function stopDownload(repoId, filename, safeKey = null) {
  const sKey = safeKey || state.activeDownloadSafeKey;
  try {
    await fetch(`${API_BASE}/v1/downloader/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id: repoId, filename })
    });

    clearInterval(state.downloadPollInterval);
    state.downloadPollInterval = null;
    state.isDownloadPaused = false;
    showToast(`Cancelled download for ${filename}`, 'info');

    // Hide top tracker bar
    document.getElementById('active-download-card')?.classList.add('hidden');

    // Reset card footer
    const footerEl = document.getElementById(`footer-${sKey}`);
    if (footerEl) {
      footerEl.innerHTML = `
        <span class="badge" style="background:#1e293b; color:#94a3b8;">GGUF</span>
        <button class="btn btn-primary btn-sm dl-preset-btn" data-repo="${repoId}" data-file="${filename}" data-safekey="${sKey}">
          📥 1-Click Download
        </button>
      `;
      footerEl.querySelector('.dl-preset-btn')?.addEventListener('click', () => {
        startDownload(repoId, filename, sKey);
      });
    }
  } catch (err) {
    showToast('Error stopping download: ' + err.message, 'error');
  }
}

async function fetchCuratedModels() {
  try {
    const res = await fetch(`${API_BASE}/v1/downloader/presets`);
    const models = await res.json();

    const grid = document.getElementById('curated-models-grid');
    if (!grid) return;

    grid.innerHTML = '';
    models.forEach(m => {
      const safeKey = `${m.repo_id}__${m.filename}`.replace(/[^a-zA-Z0-9]/g, '_');
      const card = document.createElement('div');
      card.className = 'model-store-card';
      card.innerHTML = `
        <div class="store-card-header">
          <div class="store-model-name">${m.name}</div>
          <span class="badge badge-cyan">${m.size_gb} GB</span>
        </div>
        <p class="store-model-desc">${m.description}</p>
        <div style="font-size: 10px; color: var(--text-faint);">Repo: ${m.repo_id} | Context: ${m.context.toLocaleString()}</div>
        <div class="store-card-footer" id="footer-${safeKey}">
          <span class="badge" style="background:#1e293b; color:#94a3b8;">${m.params}</span>
          <button class="btn btn-primary btn-sm dl-preset-btn" data-repo="${m.repo_id}" data-file="${m.filename}" data-safekey="${safeKey}">
            📥 1-Click Download
          </button>
        </div>
      `;
      grid.appendChild(card);
    });

    document.querySelectorAll('.dl-preset-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const repo = e.currentTarget.getAttribute('data-repo');
        const file = e.currentTarget.getAttribute('data-file');
        const safeKey = e.currentTarget.getAttribute('data-safekey');
        if (repo && file) startDownload(repo, file, safeKey);
      });
    });
  } catch (err) {
    console.error('Error fetching curated models:', err);
  }
}

async function performHFSearch() {
  const query = document.getElementById('hf-search-input')?.value?.trim();
  if (!query) return;

  const resultsSection = document.getElementById('search-results-section');
  const grid = document.getElementById('search-models-grid');
  if (resultsSection) resultsSection.classList.remove('hidden');
  if (grid) grid.innerHTML = '<div style="padding: 20px; color: var(--text-muted);">Searching Hugging Face Hub...</div>';

  try {
    const res = await fetch(`${API_BASE}/v1/downloader/search?query=${encodeURIComponent(query)}`);
    const results = await res.json();

    if (!grid) return;
    grid.innerHTML = '';

    if (results.length === 0) {
      grid.innerHTML = '<div style="padding: 20px; color: var(--text-muted);">No GGUF models found for query.</div>';
      return;
    }

    results.forEach(m => {
      const card = document.createElement('div');
      card.className = 'model-store-card';
      card.innerHTML = `
        <div class="store-card-header">
          <div class="store-model-name">${m.id}</div>
          <span class="badge" style="background:#1e293b; color:#94a3b8;">❤️ ${m.likes}</span>
        </div>
        <div style="font-size: 11px; color: var(--text-muted);">Author: ${m.author} | Downloads: ${m.downloads.toLocaleString()}</div>
        <div class="repo-files-container" id="files-${m.id.replace(/[^a-zA-Z0-9]/g, '_')}">
          <button class="btn btn-secondary btn-sm fetch-files-btn" data-repo="${m.id}">
            🔍 View Quantizations (.gguf)
          </button>
        </div>
      `;
      grid.appendChild(card);
    });

    document.querySelectorAll('.fetch-files-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const repo = e.currentTarget.getAttribute('data-repo');
        const container = document.getElementById(`files-${repo.replace(/[^a-zA-Z0-9]/g, '_')}`);
        if (!container) return;

        container.innerHTML = '<span style="font-size: 10px; color: var(--text-faint);">Fetching .gguf files...</span>';
        try {
          const fRes = await fetch(`${API_BASE}/v1/downloader/repo_files?repo_id=${encodeURIComponent(repo)}`);
          const files = await fRes.json();

          if (files.length === 0) {
            container.innerHTML = '<span style="font-size: 10px; color: #ef4444;">No GGUF files found in root.</span>';
            return;
          }

          container.innerHTML = '';
          files.forEach(f => {
            const safeKey = `${repo}__${f.filename}`.replace(/[^a-zA-Z0-9]/g, '_');
            const fRow = document.createElement('div');
            fRow.id = `footer-${safeKey}`;
            fRow.style.cssText = 'display:flex; justify-content:space-between; align-items:center; margin-top:6px; font-size:11px; min-height:30px;';
            fRow.innerHTML = `
              <span><b>${f.quant}</b> (${f.size_gb} GB)</span>
              <button class="btn btn-primary btn-sm dl-repo-file-btn" data-repo="${repo}" data-file="${f.filename}" data-safekey="${safeKey}" style="padding: 2px 8px; font-size: 10px;">
                📥 Download
              </button>
            `;
            container.appendChild(fRow);
          });

          container.querySelectorAll('.dl-repo-file-btn').forEach(dBtn => {
            dBtn.addEventListener('click', (ev) => {
              const r = ev.currentTarget.getAttribute('data-repo');
              const f = ev.currentTarget.getAttribute('data-file');
              const sk = ev.currentTarget.getAttribute('data-safekey');
              if (r && f) startDownload(r, f, sk);
            });
          });
        } catch (err) {
          container.innerHTML = `<span style="font-size:10px; color:#ef4444;">Error: ${err.message}</span>`;
        }
      });
    });
  } catch (err) {
    showToast('Search failed: ' + err.message, 'error');
  }
}

async function startDownload(repoId, filename, safeKey = null) {
  const key = `${repoId}/${filename}`;
  state.activeDownloadKey = key;
  state.activeDownloadRepo = repoId;
  state.activeDownloadFile = filename;
  state.isDownloadPaused = false;
  const sKey = safeKey || `${repoId}__${filename}`.replace(/[^a-zA-Z0-9]/g, '_');
  state.activeDownloadSafeKey = sKey;

  // Replace button in card footer with inline progress bar + pause/stop buttons
  const footerEl = document.getElementById(`footer-${sKey}`);
  if (footerEl) {
    footerEl.innerHTML = `
      <div class="card-inline-progress" id="progress-${sKey}">
        <div class="inline-progress-meta">
          <span class="inline-pct" id="pct-${sKey}">0.0%</span>
          <div class="inline-progress-actions">
            <span class="inline-speed" id="spd-${sKey}">Connecting...</span>
            <button class="btn-mini" id="inline-pause-${sKey}" title="Pause / Resume">⏸️</button>
            <button class="btn-mini btn-mini-danger" id="inline-stop-${sKey}" title="Cancel Download">✕</button>
          </div>
        </div>
        <div style="height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden; margin: 2px 0;">
          <div id="fill-${sKey}" style="width: 0%; height: 100%; background: var(--accent-gradient); transition: width 0.25s ease;"></div>
        </div>
        <div class="inline-progress-sub">
          <span id="bytes-${sKey}">0 MB / 0 MB</span>
          <span id="eta-${sKey}">ETA: --</span>
        </div>
      </div>
    `;

    footerEl.querySelector(`#inline-pause-${sKey}`)?.addEventListener('click', () => {
      if (state.isDownloadPaused) {
        resumeDownload(repoId, filename, sKey);
      } else {
        pauseDownload(repoId, filename, sKey);
      }
    });

    footerEl.querySelector(`#inline-stop-${sKey}`)?.addEventListener('click', () => {
      stopDownload(repoId, filename, sKey);
    });
  }

  // Also show top sticky active download bar
  const topCard = document.getElementById('active-download-card');
  const topTitle = document.getElementById('dl-title');
  const pill = document.getElementById('dl-status-pill');
  const btnPause = document.getElementById('btn-dl-pause-resume');
  if (topCard) topCard.classList.remove('hidden');
  if (topTitle) topTitle.textContent = `Downloading ${filename} (${repoId})...`;
  if (pill) {
    pill.className = 'badge badge-cyan';
    pill.textContent = '⚡ DOWNLOADING';
  }
  if (btnPause) btnPause.textContent = '⏸️ Pause';

  try {
    const res = await fetch(`${API_BASE}/v1/downloader/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id: repoId, filename })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Download failed to start.');

    showToast(`Downloading ${filename}...`, 'info');

    if (state.downloadPollInterval) clearInterval(state.downloadPollInterval);
    state.downloadPollInterval = setInterval(() => pollDownloadProgress(sKey, repoId, filename), 500);
  } catch (err) {
    showToast('Failed to start download: ' + err.message, 'error');
    if (footerEl) {
      footerEl.innerHTML = `<span style="color:#ef4444; font-size:10px;">❌ ${escapeHtml(err.message)}</span>`;
    }
  }
}

async function pollDownloadProgress(sKey, repoId, filename) {
  if (!state.activeDownloadKey) return;
  try {
    const res = await fetch(`${API_BASE}/v1/downloader/progress?key=${encodeURIComponent(state.activeDownloadKey)}`);
    const data = await res.json();

    const topStats = document.getElementById('dl-stats');
    const topFill = document.getElementById('dl-progress-fill');
    const pill = document.getElementById('dl-status-pill');
    const btnPause = document.getElementById('btn-dl-pause-resume');

    const pctEl = document.getElementById(`pct-${sKey}`);
    const spdEl = document.getElementById(`spd-${sKey}`);
    const fillEl = document.getElementById(`fill-${sKey}`);
    const bytesEl = document.getElementById(`bytes-${sKey}`);
    const etaEl = document.getElementById(`eta-${sKey}`);
    const footerEl = document.getElementById(`footer-${sKey}`);

    if (data.status === 'downloading') {
      state.isDownloadPaused = false;
      const dMb = (data.downloaded_bytes / (1024 * 1024)).toFixed(1);
      const tMb = (data.total_bytes / (1024 * 1024)).toFixed(1);
      const pctStr = `${data.percent.toFixed(1)}%`;
      const spdStr = `${data.speed_mb_s.toFixed(2)} MB/s`;
      const etaStr = `ETA: ${data.eta_s.toFixed(0)}s`;

      if (topStats) topStats.textContent = `${dMb} MB / ${tMb} MB (${spdStr} | ${etaStr})`;
      if (topFill) topFill.style.width = `${data.percent}%`;
      if (pill) {
        pill.className = 'badge badge-cyan';
        pill.textContent = '⚡ DOWNLOADING';
      }
      if (btnPause) btnPause.textContent = '⏸️ Pause';

      if (pctEl) pctEl.textContent = pctStr;
      if (spdEl) spdEl.textContent = spdStr;
      if (fillEl) fillEl.style.width = `${data.percent}%`;
      if (bytesEl) bytesEl.textContent = `${dMb} MB / ${tMb} MB`;
      if (etaEl) etaEl.textContent = etaStr;
    } else if (data.status === 'paused') {
      state.isDownloadPaused = true;
      const dMb = (data.downloaded_bytes / (1024 * 1024)).toFixed(1);
      const tMb = (data.total_bytes / (1024 * 1024)).toFixed(1);
      if (topStats) topStats.textContent = `PAUSED • ${dMb} MB / ${tMb} MB (${data.percent.toFixed(1)}%)`;
      if (pill) {
        pill.className = 'badge';
        pill.style.cssText = 'background: rgba(234, 179, 8, 0.2); color: #eab308; border: 1px solid rgba(234, 179, 8, 0.4);';
        pill.textContent = '⏸️ PAUSED';
      }
      if (btnPause) btnPause.textContent = '▶️ Resume';

      if (spdEl) spdEl.textContent = 'Paused';
    } else if (data.status === 'completed') {
      clearInterval(state.downloadPollInterval);
      if (topStats) topStats.textContent = 'Download Complete! Model ready.';
      if (topFill) topFill.style.width = '100%';
      if (pill) {
        pill.className = 'badge badge-green';
        pill.textContent = '✅ READY';
      }

      if (footerEl) {
        footerEl.innerHTML = `
          <div class="card-inline-done">
            <span class="badge badge-green">✅ Ready</span>
            <button class="btn btn-primary btn-sm inline-load-btn" data-path="${data.file_path || `models/${filename}`}" data-name="${filename}" style="padding: 3px 10px; font-size: 10px;">
              ⚡ Load Model
            </button>
          </div>
        `;
        footerEl.querySelector('.inline-load-btn')?.addEventListener('click', (e) => {
          const p = e.currentTarget.getAttribute('data-path');
          const n = e.currentTarget.getAttribute('data-name');
          loadModelPath(p, n);
        });
      }

      showToast(`Model ${filename} downloaded successfully!`, 'success');
      fetchModelsList();
    } else if (data.status === 'error' || data.status === 'stopped') {
      clearInterval(state.downloadPollInterval);
      if (topStats) topStats.textContent = `Download: ${data.error || 'Stopped'}`;
      if (footerEl) {
        footerEl.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:center; width:100%;">
            <span style="color:#ef4444; font-size:10px;">${data.status === 'stopped' ? '⏹️ Stopped' : '❌ Error'}</span>
            <button class="btn btn-secondary btn-sm retry-dl-btn" style="padding:2px 6px; font-size:10px;">🔄 Retry</button>
          </div>
        `;
        footerEl.querySelector('.retry-dl-btn')?.addEventListener('click', () => {
          startDownload(repoId, filename, sKey);
        });
      }
      if (data.status === 'error') {
        showToast('Download error: ' + data.error, 'error');
      }
    }
  } catch (err) {
    // Network retry
  }
}

/* ==========================================================================
   Local OpenAI API Server Manager
   ========================================================================== */
function initServerManager() {
  const copyUrlBtn = document.getElementById('btn-copy-server-url');
  const snippetCode = document.getElementById('snippet-code');
  const copySnippetBtn = document.getElementById('btn-copy-snippet');
  const snippetTabs = document.querySelectorAll('.snippet-tab');
  const testSendBtn = document.getElementById('btn-send-api-test');
  const chkServerEnable = document.getElementById('chk-server-enable');
  const liveBadge = document.getElementById('server-live-badge');

  // Server Enable / Disable Toggle Switch
  chkServerEnable?.addEventListener('change', async (e) => {
    const enabled = e.target.checked;
    try {
      const res = await fetch(`${API_BASE}/v1/server/toggle`, { method: 'POST' });
      const data = await res.json();
      const isOnline = data.enabled ?? enabled;

      if (liveBadge) {
        liveBadge.className = isOnline ? 'badge badge-green' : 'badge badge-faint';
        liveBadge.textContent = isOnline ? '🟢 ONLINE : 8008' : '⚪ PAUSED / DISABLED';
      }
      showToast(isOnline ? 'OpenAI API Server is active!' : 'OpenAI API Server paused.', isOnline ? 'success' : 'info');
    } catch (err) {
      showToast('Error toggling server: ' + err.message, 'error');
    }
  });

  // Copy Server Base URL
  copyUrlBtn?.addEventListener('click', () => {
    navigator.clipboard.writeText(`${API_BASE}/v1`);
    showToast('Copied API Base URL: ' + `${API_BASE}/v1`, 'success');
  });

  // Render Initial Snippet
  if (snippetCode) {
    snippetCode.textContent = CODE_SNIPPETS['snip-cursor'];
    hljs.highlightElement(snippetCode);
  }

  // Snippet Tabs
  snippetTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      snippetTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const snipKey = tab.getAttribute('data-snippet');
      if (snippetCode && CODE_SNIPPETS[snipKey]) {
        snippetCode.textContent = CODE_SNIPPETS[snipKey];
        hljs.highlightElement(snippetCode);
      }
    });
  });

  copySnippetBtn?.addEventListener('click', () => {
    if (snippetCode) {
      navigator.clipboard.writeText(snippetCode.textContent);
      showToast('Copied integration snippet!', 'success');
    }
  });

  // Preset payload buttons
  const btnPresetChat = document.getElementById('btn-api-preset-chat');
  const btnPresetComp = document.getElementById('btn-api-preset-comp');
  const reqTextarea = document.getElementById('api-req-payload');

  btnPresetChat?.addEventListener('click', () => {
    if (reqTextarea) {
      reqTextarea.value = JSON.stringify({
        model: "local-llama",
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "Explain Google TurboQuant in 2 sentences." }
        ],
        temperature: 0.7,
        max_tokens: 150
      }, null, 2);
    }
  });

  btnPresetComp?.addEventListener('click', () => {
    if (reqTextarea) {
      reqTextarea.value = JSON.stringify({
        model: "local-llama",
        prompt: "Complete the function:\ndef compute_hadamard_transform(vec):\n    ",
        temperature: 0.2,
        max_tokens: 128
      }, null, 2);
    }
  });

  // Sandbox Tester
  testSendBtn?.addEventListener('click', async () => {
    const reqText = document.getElementById('api-req-payload')?.value;
    const resView = document.getElementById('api-res-payload');
    const timingTag = document.getElementById('api-timing-tag');
    if (!reqText || !resView) return;

    resView.value = 'Sending request to local API server...';
    if (timingTag) {
      timingTag.textContent = '⏳ Processing...';
      timingTag.style.color = 'var(--primary)';
    }

    const t0 = performance.now();
    try {
      const parsedReq = JSON.parse(reqText);
      const endpoint = parsedReq.messages ? `${API_BASE}/v1/chat/completions` : `${API_BASE}/v1/completions`;
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...parsedReq, stream: false })
      });

      const t1 = performance.now();
      const elapsedMs = Math.round(t1 - t0);

      const data = await res.json();
      resView.value = JSON.stringify(data, null, 2);

      if (timingTag) {
        if (res.ok) {
          timingTag.textContent = `⚡ 200 OK (${elapsedMs}ms)`;
          timingTag.style.color = '#10b981';
        } else {
          timingTag.textContent = `❌ ${res.status} Error (${elapsedMs}ms)`;
          timingTag.style.color = '#ef4444';
        }
      }
    } catch (err) {
      resView.value = `Error: ${err.message}`;
      if (timingTag) {
        timingTag.textContent = `❌ Error`;
        timingTag.style.color = '#ef4444';
      }
    }
  });
}

/* ==========================================================================
   Sampling Parameters Sliders & Badges
   ========================================================================== */
function initSamplingSliders() {
  const bindSlider = (sliderId, valId, transform = (v) => v) => {
    const slider = document.getElementById(sliderId);
    const valEl = document.getElementById(valId);
    if (!slider || !valEl) return;

    slider.addEventListener('input', (e) => {
      valEl.textContent = transform(e.target.value);
    });
  };

  bindSlider('slider-temp', 'val-temp', (v) => (v / 100).toFixed(2));
  bindSlider('slider-topp', 'val-topp', (v) => (v / 100).toFixed(2));
  bindSlider('slider-topk', 'val-topk', (v) => v);
  bindSlider('slider-minp', 'val-minp', (v) => (v / 100).toFixed(2));
  bindSlider('slider-reppen', 'val-reppen', (v) => (v / 100).toFixed(2));
  bindSlider('slider-maxtok', 'val-maxtok', (v) => v);
  bindSlider('slider-gpulayers', 'val-gpulayers', (v) => v === '-1' ? 'All (-1)' : v);

  // Grammar Enable / Disable Toggle & Custom GBNF
  const chkGrammar = document.getElementById('chk-grammar-enable');
  const grammarSelect = document.getElementById('grammar-select');
  const grammarControls = document.getElementById('grammar-controls-container');
  const customGbnf = document.getElementById('custom-gbnf-input');

  chkGrammar?.addEventListener('change', (e) => {
    const enabled = e.target.checked;
    if (grammarSelect) grammarSelect.disabled = !enabled;
    if (grammarControls) grammarControls.classList.toggle('disabled-state', !enabled);
    if (customGbnf) {
      customGbnf.classList.toggle('hidden', !enabled || grammarSelect?.value !== 'custom');
    }
  });

  grammarSelect?.addEventListener('change', (e) => {
    if (customGbnf) {
      customGbnf.classList.toggle('hidden', e.target.value !== 'custom');
    }
  });
}
