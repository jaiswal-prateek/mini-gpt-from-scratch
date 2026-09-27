/* ═══════════════════════════════════════════════════════
   Mini GPT Playground — Unified Frontend
   ═══════════════════════════════════════════════════════ */

const API = "";
const NUM_COLORS = 12;
const PAGE_SIZE = 100;

let debounceTimer = null;
let currentTokenizer = "tinystories_8192";
let currentView = "tokens";
let vocabOffset = 0;
let mergeOffset = 0;

// LLM state
let generating = false;
let generationSteps = []; // [{token, token_id, prob, candidates}, ...]
let activeStepIdx = null;

// ── Init ────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadTheme();
  setupMainTabs();
  setupSubTabs();
  setupTokenizerSelect();
  setupEncodeInput();
  setupDecodeInput();
  setupViewToggle();
  loadTokInfo();
  loadModelInfo();
  showSubSection("encode");
});

// ══════════════════════════════════════════════════════════
//  THEME
// ══════════════════════════════════════════════════════════

function loadTheme() {
  const saved = localStorage.getItem("mgpt-theme") || "light";
  document.documentElement.setAttribute("data-theme", saved);
  document.getElementById("theme-icon").textContent = saved === "dark" ? "🌙" : "☀️";
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("mgpt-theme", next);
  document.getElementById("theme-icon").textContent = next === "dark" ? "🌙" : "☀️";
}

// ══════════════════════════════════════════════════════════
//  MAIN TABS
// ══════════════════════════════════════════════════════════

function setupMainTabs() {
  document.querySelectorAll(".main-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".main-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      document.querySelectorAll(".main-section").forEach(s => s.classList.remove("active"));
      document.getElementById(`main-${tab.dataset.main}`).classList.add("active");
    });
  });
}

// ══════════════════════════════════════════════════════════
//  TOKENIZER — Sub Tabs
// ══════════════════════════════════════════════════════════

function setupSubTabs() {
  document.querySelectorAll(".sub-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".sub-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      showSubSection(tab.dataset.section);
    });
  });
}

function showSubSection(name) {
  document.querySelectorAll("#main-tokenizer .section").forEach(s => s.classList.remove("active"));
  const el = document.getElementById(`section-${name}`);
  if (el) el.classList.add("active");
  if (name === "vocab") loadVocab(0);
  if (name === "merges") loadMerges(0);
  if (name === "tok-info") loadTokInfo();
}

// ══════════════════════════════════════════════════════════
//  TOKENIZER — Encode
// ══════════════════════════════════════════════════════════

function setupTokenizerSelect() {
  const sel = document.getElementById("tokenizer-select");
  sel.addEventListener("change", () => {
    currentTokenizer = sel.value;
    triggerEncode();
    const active = document.querySelector(".sub-tab.active")?.dataset.section;
    if (active === "vocab") loadVocab(0);
    if (active === "merges") loadMerges(0);
    if (active === "tok-info") loadTokInfo();
  });
}

function setupEncodeInput() {
  document.getElementById("encode-input").addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(triggerEncode, 200);
  });
  triggerEncode();
}

async function triggerEncode() {
  const text = document.getElementById("encode-input").value;
  if (!text) { renderEmpty(); return; }
  try {
    const res = await fetch(`${API}/api/encode`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text, tokenizer: currentTokenizer}),
    });
    const data = await res.json();
    renderStats(data);
    renderTokens(data.tokens);
    renderTokenIds(data.token_ids);
    updateVisibility();
  } catch (e) { console.error("Encode error:", e); }
}

function renderEmpty() {
  document.getElementById("stat-tokens").textContent = "0";
  document.getElementById("stat-chars").textContent = "0";
  document.getElementById("stat-ratio").textContent = "—";
  document.getElementById("token-display").innerHTML = '<div class="placeholder">Start typing to see tokens…</div>';
  document.getElementById("token-ids-display").innerHTML = "";
}

function renderStats(data) {
  document.getElementById("stat-tokens").textContent = data.num_tokens.toLocaleString();
  document.getElementById("stat-chars").textContent = data.num_chars.toLocaleString();
  document.getElementById("stat-ratio").textContent = data.num_chars > 0 ? (data.num_chars / data.num_tokens).toFixed(2) : "—";
}

function renderTokens(tokens) {
  const c = document.getElementById("token-display");
  c.innerHTML = "";
  tokens.forEach((tok, i) => {
    const span = document.createElement("span");
    const isSp = tok.text.startsWith("<|") && tok.text.endsWith("|>");
    span.className = `token-chip ${isSp ? "token-special" : `tc${i % NUM_COLORS}`}`;
    span.textContent = tok.text.replace(/\n/g, "↵\n").replace(/\t/g, "→\t");
    const tip = document.createElement("span");
    tip.className = "tip";
    tip.innerHTML = `<div class="tip-row"><span class="tip-label">ID</span><span class="tip-val">${tok.id}</span></div><div class="tip-row"><span class="tip-label">Pos</span><span class="tip-val">${i}</span></div><div class="tip-row"><span class="tip-label">Text</span><span class="tip-val">${esc(JSON.stringify(tok.text))}</span></div>`;
    span.appendChild(tip);
    c.appendChild(span);
  });
}

function renderTokenIds(ids) {
  document.getElementById("token-ids-display").innerHTML = "[" + ids.map((id, i) => `<span class="tid" title="Position ${i}">${id}</span>`).join(", ") + "]";
}

// ── View Toggle ─────────────────────────────────────────
function setupViewToggle() {
  document.querySelectorAll(".view-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".view-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentView = btn.dataset.view;
      updateVisibility();
    });
  });
}

function updateVisibility() {
  document.getElementById("token-display").style.display = currentView === "tokens" ? "" : "none";
  document.getElementById("token-ids-display").style.display = currentView === "ids" ? "" : "none";
}

// ══════════════════════════════════════════════════════════
//  TOKENIZER — Decode
// ══════════════════════════════════════════════════════════

function setupDecodeInput() {
  document.getElementById("decode-input").addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(triggerDecode, 300);
  });
}

async function triggerDecode() {
  const raw = document.getElementById("decode-input").value.trim();
  const out = document.getElementById("decode-output");
  if (!raw) { out.textContent = ""; return; }
  try {
    let ids;
    if (raw.startsWith("[")) ids = JSON.parse(raw);
    else ids = raw.split(/[\s,]+/).filter(Boolean).map(Number);
    const res = await fetch(`${API}/api/decode`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({token_ids: ids, tokenizer: currentTokenizer}),
    });
    const data = await res.json();
    out.textContent = data.text;
  } catch (e) { out.textContent = `Error: ${e.message}`; }
}

// ══════════════════════════════════════════════════════════
//  TOKENIZER — Vocab
// ══════════════════════════════════════════════════════════

async function loadVocab(offset) {
  vocabOffset = offset;
  const search = document.getElementById("vocab-search")?.value || "";
  try {
    const p = new URLSearchParams({tokenizer: currentTokenizer, search, offset: vocabOffset, limit: PAGE_SIZE});
    const res = await fetch(`${API}/api/vocab?${p}`);
    const data = await res.json();
    const tbody = document.getElementById("vocab-tbody");
    tbody.innerHTML = "";
    data.items.forEach(item => {
      const tr = document.createElement("tr");
      let badge = item.is_special ? '<span class="badge badge-special">special</span>' : item.is_byte_token ? '<span class="badge badge-byte">byte</span>' : '<span class="badge badge-merged">merged</span>';
      tr.innerHTML = `<td class="cell-id">${item.id}</td><td class="cell-text">${esc(item.text).replace(/ /g,"·").replace(/\n/g,"↵")}</td><td class="cell-bytes">${item.bytes}</td><td>${item.length}</td><td>${badge}</td>`;
      tbody.appendChild(tr);
    });
    const from = data.offset + 1, to = Math.min(data.offset + data.limit, data.total);
    document.getElementById("vocab-page-info").textContent = `${from}–${to} of ${data.total.toLocaleString()}`;
    document.getElementById("vocab-prev").disabled = data.offset === 0;
    document.getElementById("vocab-next").disabled = data.offset + data.limit >= data.total;
  } catch (e) { console.error(e); }
}
function vocabPrev() { loadVocab(Math.max(0, vocabOffset - PAGE_SIZE)); }
function vocabNext() { loadVocab(vocabOffset + PAGE_SIZE); }
function vocabSearch() { loadVocab(0); }

// ══════════════════════════════════════════════════════════
//  TOKENIZER — Merges
// ══════════════════════════════════════════════════════════

async function loadMerges(offset) {
  mergeOffset = offset;
  try {
    const p = new URLSearchParams({tokenizer: currentTokenizer, offset: mergeOffset, limit: PAGE_SIZE});
    const res = await fetch(`${API}/api/merges?${p}`);
    const data = await res.json();
    const tbody = document.getElementById("merges-tbody");
    tbody.innerHTML = "";
    data.items.forEach(item => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td class="cell-rank">${item.rank}</td><td class="cell-id">${item.pair[0]}</td><td class="cell-text">${esc(item.pair_text[0]).replace(/ /g,"·")}</td><td class="cell-id">${item.pair[1]}</td><td class="cell-text">${esc(item.pair_text[1]).replace(/ /g,"·")}</td><td class="cell-arrow">→</td><td class="cell-id">${item.new_id}</td><td class="cell-text">${esc(item.new_text).replace(/ /g,"·")}</td>`;
      tbody.appendChild(tr);
    });
    const from = data.offset + 1, to = Math.min(data.offset + data.limit, data.total);
    document.getElementById("merges-page-info").textContent = `${from}–${to} of ${data.total.toLocaleString()}`;
    document.getElementById("merges-prev").disabled = data.offset === 0;
    document.getElementById("merges-next").disabled = data.offset + data.limit >= data.total;
  } catch (e) { console.error(e); }
}
function mergesPrev() { loadMerges(Math.max(0, mergeOffset - PAGE_SIZE)); }
function mergesNext() { loadMerges(mergeOffset + PAGE_SIZE); }

// ══════════════════════════════════════════════════════════
//  TOKENIZER — Info
// ══════════════════════════════════════════════════════════

async function loadTokInfo() {
  try {
    const res = await fetch(`${API}/api/info?tokenizer=${currentTokenizer}`);
    const d = await res.json();
    document.getElementById("info-vocab-size").textContent = d.vocab_size.toLocaleString();
    document.getElementById("info-merges").textContent = d.num_merges.toLocaleString();
    document.getElementById("info-special").textContent = d.num_special_tokens;
    document.getElementById("info-special-list").textContent = Object.entries(d.special_tokens).map(([k,v]) => `${k} → ${v}`).join(", ") || "None";
  } catch (e) { console.error(e); }
}

function copyTokenIds() {
  const text = document.getElementById("token-ids-display").textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById("copy-ids-btn");
    btn.textContent = "Copied!"; btn.classList.add("copied");
    setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1500);
  });
}

// ══════════════════════════════════════════════════════════
//  LLM PLAYGROUND
// ══════════════════════════════════════════════════════════

function updateSlider(name) {
  const map = {temp: "slider-temp", topk: "slider-topk", topp: "slider-topp", maxtok: "slider-maxtok"};
  const el = document.getElementById(map[name]);
  const val = el.value;
  const fmt = (name === "topk" || name === "maxtok") ? val : parseFloat(val).toFixed(2);
  document.getElementById(`${name}-val`).textContent = fmt;
}

function useSuggestion(btn) {
  document.getElementById("prompt-input").value = btn.textContent;
  sendPrompt();
}

async function sendPrompt() {
  const input = document.getElementById("prompt-input");
  const prompt = input.value.trim();
  if (!prompt || generating) return;

  generating = true;
  generationSteps = [];
  activeStepIdx = null;
  document.getElementById("send-btn").disabled = true;

  // Clear welcome
  const msgs = document.getElementById("chat-messages");
  const welcome = msgs.querySelector(".chat-welcome");
  if (welcome) welcome.remove();

  // User message
  addMessage("user", prompt);
  input.value = "";

  // Bot message (streaming)
  const botBubble = addMessage("bot", "");
  const textEl = botBubble.querySelector(".msg-text");

  // Reset prob panel
  document.getElementById("prob-hint").style.display = "";
  document.getElementById("prob-panel").style.display = "none";

  const temp = parseFloat(document.getElementById("slider-temp").value);
  const topk = parseInt(document.getElementById("slider-topk").value);
  const topp = parseFloat(document.getElementById("slider-topp").value);
  const maxtok = parseInt(document.getElementById("slider-maxtok").value);

  try {
    const res = await fetch(`${API}/api/generate`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({prompt, max_new_tokens: maxtok, temperature: temp, top_k: topk, top_p: topp}),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = JSON.parse(line.slice(6));

        if (data.error) {
          textEl.textContent = `Error: ${data.error}`;
          break;
        }
        if (data.done) break;

        const idx = generationSteps.length;
        generationSteps.push(data);

        const span = document.createElement("span");
        span.className = "gen-token";
        span.textContent = data.token;
        span.dataset.idx = idx;
        span.addEventListener("click", () => showProb(idx));
        textEl.appendChild(span);

        // Auto-scroll
        msgs.scrollTop = msgs.scrollHeight;
      }
    }
  } catch (e) {
    textEl.textContent = `Error: ${e.message}`;
  }

  generating = false;
  document.getElementById("send-btn").disabled = false;
  msgs.scrollTop = msgs.scrollHeight;
}

function addMessage(role, text) {
  const msgs = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = `msg msg-${role}`;
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  if (role === "bot") {
    const textSpan = document.createElement("span");
    textSpan.className = "msg-text";
    textSpan.textContent = text;
    bubble.appendChild(textSpan);
  } else {
    bubble.textContent = text;
  }
  div.appendChild(bubble);
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

function showProb(idx) {
  // Highlight active token
  document.querySelectorAll(".gen-token.active").forEach(el => el.classList.remove("active"));
  const tokenEl = document.querySelector(`.gen-token[data-idx="${idx}"]`);
  if (tokenEl) tokenEl.classList.add("active");

  activeStepIdx = idx;
  const step = generationSteps[idx];
  if (!step || !step.candidates) return;

  document.getElementById("prob-hint").style.display = "none";
  document.getElementById("prob-panel").style.display = "";

  // Selected info
  const selInfo = document.getElementById("prob-selected-info");
  selInfo.innerHTML = `<span class="ps-token">"${esc(step.token)}"</span> <span class="ps-prob">${(step.prob * 100).toFixed(1)}%</span>`;

  // Candidate list
  const list = document.getElementById("prob-list");
  list.innerHTML = "";

  step.candidates.forEach(c => {
    const row = document.createElement("div");
    let cls = "prob-row";
    if (c.selected) cls += " is-selected";
    if (!c.in_sample_pool) cls += " not-in-pool";
    row.className = cls;

    const pct = (c.prob * 100).toFixed(1);
    const barW = Math.max(2, c.prob * 100);

    row.innerHTML = `
      <div class="prob-token">${esc(c.text)}${c.selected ? '<span class="prob-badge">selected</span>' : ''}</div>
      <div class="prob-pct">${pct}%</div>
      <div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:${barW}%"></div></div>
    `;
    list.appendChild(row);
  });
}

// ── Model Info ──────────────────────────────────────────
async function loadModelInfo() {
  try {
    const res = await fetch(`${API}/api/model_info`);
    const d = await res.json();
    document.getElementById("mi-params").textContent = (d.parameters / 1e6).toFixed(1) + "M";
    document.getElementById("mi-layers").textContent = d.num_blocks;
    document.getElementById("mi-heads").textContent = d.num_heads;
    document.getElementById("mi-dim").textContent = d.d_model;
  } catch (e) { console.error(e); }
}

// ── Util ────────────────────────────────────────────────
function esc(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}
