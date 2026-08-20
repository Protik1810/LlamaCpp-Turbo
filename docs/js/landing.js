// Llama.cpp Turbo Desktop — Landing Page Logic & Interactive Diagnostics

document.addEventListener('DOMContentLoaded', () => {
  // Smooth scroll for nav links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        e.preventDefault();
        targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // Fetch GitHub Stars dynamically if online
  const repoName = 'Protik1810/llamacpp-turbo';
  const starBadge = document.getElementById('gh-star-count');

  if (starBadge) {
    fetch(`https://api.github.com/repos/${repoName}`)
      .then(res => res.json())
      .then(data => {
        if (data.stargazers_count !== undefined) {
          starBadge.textContent = `${data.stargazers_count} ★`;
        }
      })
      .catch(() => {
        starBadge.textContent = '★ Star';
      });
  }

  // Interactive Theme Preview Switcher (Dark / Light)
  const themeBtns = document.querySelectorAll('.theme-toggle-btn');
  const heroMockupImg = document.getElementById('hero-mockup-img');
  const galleryMainImg = document.getElementById('gallery-main-img');
  const galleryAboutImg = document.getElementById('gallery-about-img');

  const themeImages = {
    dark: {
      main: 'assets/screenshot-dark-main.png',
      about: 'assets/screenshot-dark-about.png'
    },
    light: {
      main: 'assets/screenshot-main.png',
      about: 'assets/screenshot-about.png'
    }
  };

  themeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const selectedTheme = btn.getAttribute('data-theme');
      if (!selectedTheme) return;

      themeBtns.forEach(b => b.classList.remove('active'));
      document.querySelectorAll(`[data-theme="${selectedTheme}"]`).forEach(b => b.classList.add('active'));

      // Smooth cross-fade image swap
      if (heroMockupImg) {
        heroMockupImg.classList.add('fade-out');
        setTimeout(() => {
          heroMockupImg.src = themeImages[selectedTheme].main;
          heroMockupImg.classList.remove('fade-out');
        }, 180);
      }

      if (galleryMainImg) {
        galleryMainImg.classList.add('fade-out');
        setTimeout(() => {
          galleryMainImg.src = themeImages[selectedTheme].main;
          galleryMainImg.classList.remove('fade-out');
        }, 180);
      }

      if (galleryAboutImg) {
        galleryAboutImg.classList.add('fade-out');
        setTimeout(() => {
          galleryAboutImg.src = themeImages[selectedTheme].about;
          galleryAboutImg.classList.remove('fade-out');
        }, 180);
      }
    });
  });

  // -------------------------------------------------------------
  // Interactive TurboQuant VRAM Savings Calculator
  // -------------------------------------------------------------
  const modelSelect = document.getElementById('calc-model');
  const ctxSelect = document.getElementById('calc-context');
  const bitsSelect = document.getElementById('calc-bits');
  const sparsityInput = document.getElementById('calc-sparsity');
  const sparsityValLabel = document.getElementById('calc-sparsity-val');

  const statRawVRAM = document.getElementById('stat-raw-vram');
  const statTurboVRAM = document.getElementById('stat-turbo-vram');
  const statRatio = document.getElementById('stat-ratio');
  const statSpeedup = document.getElementById('stat-speedup');
  const statMeterFill = document.getElementById('stat-meter-fill');
  const statSavingsPct = document.getElementById('stat-savings-pct');

  const MODEL_ARCHS = {
    'llama-3.3-70b': { layers: 80, kv_heads: 8, head_dim: 128, name: 'Llama-3.3-70B' },
    'deepseek-r1-32b': { layers: 64, kv_heads: 8, head_dim: 128, name: 'DeepSeek-R1-32B' },
    'qwen-2.5-14b': { layers: 48, kv_heads: 8, head_dim: 128, name: 'Qwen-2.5-14B' },
    'gemma-2-9b': { layers: 42, kv_heads: 8, head_dim: 256, name: 'Gemma-2-9B' },
    'llama-3.1-8b': { layers: 32, kv_heads: 8, head_dim: 128, name: 'Meta-Llama-3.1-8B' }
  };

  function updateVRAMCalculator() {
    if (!modelSelect || !ctxSelect || !bitsSelect || !sparsityInput) return;

    const modelKey = modelSelect.value || 'llama-3.3-70b';
    const ctxLen = parseInt(ctxSelect.value, 10) || 16384;
    const bits = parseInt(bitsSelect.value, 10) || 4;
    const sparsity = (parseFloat(sparsityInput.value) || 0) / 100.0;

    if (sparsityValLabel) {
      sparsityValLabel.textContent = `${Math.round(sparsity * 100)}%`;
    }

    const arch = MODEL_ARCHS[modelKey] || MODEL_ARCHS['llama-3.3-70b'];

    // FP16 Raw KV cache calculation: 2 (K+V) * layers * kv_heads * head_dim * context * 2 bytes
    const rawBytes = 2 * arch.layers * arch.kv_heads * arch.head_dim * ctxLen * 2;
    const rawGB = rawBytes / (1024 * 1024 * 1024);

    // Compression calculations
    const baseRatio = 16.0 / Math.max(1, bits);
    const effectiveRatio = baseRatio * (1.0 / Math.max(0.05, 1.0 - sparsity));
    const turboGB = rawGB / effectiveRatio;
    const savingsPct = Math.max(0, (1.0 - (turboGB / rawGB)) * 100);
    const speedup = 1.0 + (0.35 * (effectiveRatio - 1.0) / effectiveRatio);

    if (statRawVRAM) statRawVRAM.textContent = `${rawGB.toFixed(2)} GB`;
    if (statTurboVRAM) statTurboVRAM.textContent = `${turboGB.toFixed(2)} GB`;
    if (statRatio) statRatio.textContent = `${effectiveRatio.toFixed(1)}x`;
    if (statSpeedup) statSpeedup.textContent = `${speedup.toFixed(2)}x`;
    if (statSavingsPct) statSavingsPct.textContent = `${savingsPct.toFixed(1)}% VRAM Saved`;
    if (statMeterFill) statMeterFill.style.width = `${Math.min(100, savingsPct)}%`;
  }

  if (modelSelect) modelSelect.addEventListener('change', updateVRAMCalculator);
  if (ctxSelect) ctxSelect.addEventListener('change', updateVRAMCalculator);
  if (bitsSelect) bitsSelect.addEventListener('change', updateVRAMCalculator);
  if (sparsityInput) sparsityInput.addEventListener('input', updateVRAMCalculator);

  updateVRAMCalculator();

  // -------------------------------------------------------------
  // Interactive API Code Tabs Switcher
  // -------------------------------------------------------------
  const apiTabBtns = document.querySelectorAll('.api-tab-btn');
  const apiCodePre = document.getElementById('api-code-snippet');

  const API_SNIPPETS = {
    python: `import openai

client = openai.OpenAI(
    base_url="http://localhost:8008/v1",
    api_key="not-needed"  # Local offline engine
)

# Live streaming with thinking tag extraction
response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "system", "content": "You are an expert AI assistant."},
        {"role": "user", "content": "Write a fast Python quicksort with detailed reasoning."}
    ],
    temperature=0.7,
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)`,

    curl: `curl http://localhost:8008/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "system", "content": "You are an expert software architect."},
      {"role": "user", "content": "Explain TurboQuant FWHT KV cache compression."}
    ],
    "temperature": 0.7,
    "stream": false
  }'`,

    js: `// Node.js or Browser Fetch with OpenAI format
async function generateChat() {
  const res = await fetch("http://localhost:8008/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "local-model",
      messages: [{ role: "user", content: "Hello from Electron!" }],
      temperature: 0.7,
      stream: false
    })
  });

  const data = await res.json();
  console.log("Model Response:", data.choices[0].message.content);
}

generateChat();`,

    cursor: `// In Cursor / VS Code Continue settings.json:
{
  "models": [
    {
      "title": "Llama.cpp Turbo (Local)",
      "provider": "openai",
      "model": "local-model",
      "apiBase": "http://localhost:8008/v1",
      "apiKey": "local"
    }
  ]
}`
  };

  apiTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const lang = btn.getAttribute('data-lang');
      if (!lang || !API_SNIPPETS[lang]) return;

      apiTabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      if (apiCodePre) {
        apiCodePre.textContent = API_SNIPPETS[lang];
      }
    });
  });

  // Copy code snippet helper
  window.copySnippet = function(elementId, btn) {
    const text = document.getElementById(elementId)?.innerText;
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      const originalText = btn.innerHTML;
      btn.innerHTML = '✓ Copied!';
      btn.style.color = '#34d399';
      setTimeout(() => {
        btn.innerHTML = originalText;
        btn.style.color = '';
      }, 2000);
    });
  };
});
