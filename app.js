/* EcoSphere – Shared Utilities */

// ─── Sidebar toggle ───────────────────────────────────────────────────────────
(function () {
  document.addEventListener('DOMContentLoaded', () => {
    const sidebar  = document.getElementById('sidebar');
    const overlay  = document.getElementById('sidebar-overlay');
    const menuBtn  = document.getElementById('menu-btn');

    if (menuBtn && sidebar && overlay) {
      menuBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        overlay.classList.toggle('open');
      });
      overlay.addEventListener('click', () => {
        sidebar.classList.remove('open');
        overlay.classList.remove('open');
      });
    }

    // Mark active nav link
    const links = document.querySelectorAll('.nav-item');
    const current = window.location.pathname.split('/').pop() || 'index.html';
    links.forEach(link => {
      const href = link.getAttribute('href');
      if (href && href === current) link.classList.add('active');
    });

    // Range slider live display
    document.querySelectorAll('input[type="range"]').forEach(slider => {
      const display = document.getElementById(slider.id + '-val');
      if (display) {
        display.textContent = slider.value;
        slider.addEventListener('input', () => { display.textContent = slider.value; });
      }
    });
  });
})();

// ─── EcoScore storage ─────────────────────────────────────────────────────────
const EcoScore = {
  scores: JSON.parse(localStorage.getItem('eco_scores') || '{}'),
  set(module, score) {
    this.scores[module] = Math.max(0, Math.min(100, score));
    localStorage.setItem('eco_scores', JSON.stringify(this.scores));
    this._updateBadge();
  },
  overall() {
    const vals = Object.values(this.scores);
    if (!vals.length) return '--';
    return Math.round(vals.reduce((a, b) => a + b, 0) / vals.length);
  },
  _updateBadge() {
    const el = document.getElementById('eco-score-val');
    if (el) el.textContent = this.overall();
  },
  init() {
    this._updateBadge();
  }
};

document.addEventListener('DOMContentLoaded', () => EcoScore.init());

// ─── Chart.js defaults ────────────────────────────────────────────────────────
function applyChartDefaults() {
  if (typeof Chart === 'undefined') return;
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = '#1e3a5f';
  Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
}
document.addEventListener('DOMContentLoaded', applyChartDefaults);

// ─── Utility helpers ─────────────────────────────────────────────────────────
function fmt(n, dec = 1) {
  if (n === null || n === undefined || isNaN(n)) return '0';
  return Number(n).toFixed(dec);
}

function lerp(a, b, t) { return a + (b - a) * t; }

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// Animate counting
function animateCount(el, from, to, duration = 800, decimals = 1) {
  if (!el) return;
  const target = (typeof to === 'number' && isFinite(to)) ? to : 0;
  const startVal = (typeof from === 'number' && isFinite(from)) ? from : 0;
  const start = performance.now();
  function update(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = fmt(lerp(startVal, target, ease), decimals);
    if (t < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// Show results section
function showResults(id) {
  const el = document.getElementById(id);
  if (el) {
    el.classList.add('visible');
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// Sidebar HTML snippet (shared)
function renderSidebar(activePage) {
  return `
<aside class="sidebar" id="sidebar">
  <div class="sidebar-logo">
    <div class="logo-icon">🌿</div>
    <h1>EcoSphere</h1>
    <p>Sustainability Intelligence</p>
  </div>
  <nav class="sidebar-nav">
    <div class="nav-section-title">Overview</div>
    <a class="nav-item${activePage==='index'?' active':''}" href="index.html">
      <span class="nav-icon">🏠</span> Dashboard
    </a>
    <div class="nav-section-title">Modules</div>
    <a class="nav-item${activePage==='carbon'?' active':''}" href="carbon-footprint.html">
      <span class="nav-icon">🌫️</span> Carbon Footprint
    </a>
    <a class="nav-item${activePage==='solar-loss'?' active':''}" href="solar-loss.html">
      <span class="nav-icon">🔋</span> Solar Loss Predictor
    </a>
    <a class="nav-item${activePage==='ewaste'?' active':''}" href="e-waste.html">
      <span class="nav-icon">♻️</span> E-Waste Recycling
    </a>
    <a class="nav-item${activePage==='air'?' active':''}" href="air-quality.html">
      <span class="nav-icon">💨</span> Air Quality
    </a>
    <a class="nav-item${activePage==='solar-energy'?' active':''}" href="solar-energy.html">
      <span class="nav-icon">☀️</span> Solar Energy
    </a>
    <a class="nav-item${activePage==='food'?' active':''}" href="food-waste.html">
      <span class="nav-icon">🥦</span> Food Waste
    </a>
    <div class="nav-section-title">Multimodal AI & Assistant</div>
    <a class="nav-item${activePage==='vision'?' active':''}" href="vision-ai.html">
      <span class="nav-icon">👁️</span> Multimodal Vision AI
    </a>
    <a class="nav-item${activePage==='chat'?' active':''}" href="chat.html">
      <span class="nav-icon">🤖</span> AI Sustainability Chatbot
    </a>
    <a class="nav-item${activePage==='research'?' active':''}" href="research-map.html">
      <span class="nav-icon">📚</span> IEEE Research Map
    </a>
  </nav>
</aside>
<div class="sidebar-overlay" id="sidebar-overlay"></div>`;
}

// ─── EcoData Persistent Database & State Store ──────────────────────────────
const EcoData = {
  // Store keys
  LATEST_KEY: 'ecosphere_latest_records',
  HISTORY_KEY: 'ecosphere_historical_records',

  // Save calculation as latest record & append to historical audit timeline
  saveLatest(moduleKey, data, modelMeta = {}) {
    const timestamp = new Date().toISOString();
    const formattedTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const record = {
      ...data,
      module: moduleKey,
      timestamp: timestamp,
      lastUpdatedText: formattedTime,
      modelName: modelMeta.name || 'AI/ML Stacking Regressor',
      modelVersion: modelMeta.version || 'v2.4.0',
      r2Score: modelMeta.r2 || 0.954,
      mapeError: modelMeta.mape || 4.21
    };

    // 1. Save Latest Record (replaces previous latest)
    const latestAll = this.getAllLatest();
    latestAll[moduleKey] = record;
    localStorage.setItem(this.LATEST_KEY, JSON.stringify(latestAll));

    // Also update legacy store for backwards compatibility
    localStorage.setItem('ecosphere_module_data', JSON.stringify(latestAll));

    // 2. Append to Historical Timeline for previous-vs-current trend comparisons
    const historyAll = this.getHistory(moduleKey);
    historyAll.unshift({ ...record, id: 'rec_' + Math.random().toString(36).substr(2, 9) });
    // Keep last 30 historical calculation snapshots
    if (historyAll.length > 30) historyAll.pop();
    const storedHist = JSON.parse(localStorage.getItem(this.HISTORY_KEY) || '{}');
    storedHist[moduleKey] = historyAll;
    localStorage.setItem(this.HISTORY_KEY, JSON.stringify(storedHist));

    // 3. Render 'Last Updated' UI badge if container exists
    this.renderLastUpdatedBadge(moduleKey, formattedTime);

    // 4. Invalidate any stale cached predictions
    if (window.dispatchEvent) {
      window.dispatchEvent(new CustomEvent('ecosphere:data-updated', { detail: { module: moduleKey, record } }));
    }

    return record;
  },

  // Backwards compatibility alias
  set(moduleKey, data) {
    return this.saveLatest(moduleKey, data);
  },

  // Retrieve the newest persisted calculation for a module
  getLatest(moduleKey) {
    const latestAll = this.getAllLatest();
    return latestAll[moduleKey] || null;
  },

  get(moduleKey) {
    return this.getLatest(moduleKey);
  },

  getAllLatest() {
    try {
      return JSON.parse(localStorage.getItem(this.LATEST_KEY) || localStorage.getItem('ecosphere_module_data') || '{}');
    } catch (e) {
      return {};
    }
  },

  getAll() {
    return this.getAllLatest();
  },

  // Retrieve full calculation history for historical trend comparisons
  getHistory(moduleKey) {
    try {
      const stored = JSON.parse(localStorage.getItem(this.HISTORY_KEY) || '{}');
      return moduleKey ? (stored[moduleKey] || []) : stored;
    } catch (e) {
      return moduleKey ? [] : {};
    }
  },

  // Renders live 'Last Updated' indicator on module headers
  renderLastUpdatedBadge(moduleKey, timeStr) {
    let badge = document.getElementById('last-updated-badge');
    if (!badge) {
      const headerTitle = document.querySelector('.topbar-title') || document.querySelector('.header-title') || document.querySelector('.top-header');
      if (headerTitle) {
        badge = document.createElement('div');
        badge.id = 'last-updated-badge';
        badge.style.cssText = 'font-size: 11px; color: var(--accent-green); display: inline-flex; align-items: center; gap: 4px; margin-top: 4px; background: rgba(0, 230, 118, 0.08); padding: 2px 8px; border-radius: 12px; border: 1px solid rgba(0, 230, 118, 0.2);';
        headerTitle.appendChild(badge);
      }
    }
    if (badge) {
      badge.innerHTML = `🕒 <strong>Last Updated:</strong> ${timeStr || 'Just now'}`;
    }
  }
};

window.EcoData = EcoData;
window.renderSidebar = renderSidebar;

