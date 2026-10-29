/* Bench Press: run a model from the browser. Mirrors benchpress/run.py and benchpress/providers.py: one fresh
   single-turn call per stimulus (no system prompt), the provider's default temperature unless one is set, the paper's
   output budgets, retries with exponential backoff, and the same row format in results.jsonl, so a run started here
   can be scored or resumed with the command-line tools. The API key lives only in this page's memory (or, if asked,
   in sessionStorage for this tab) and is sent only to the chosen provider. */
(function () {
  "use strict";
  const B = window.BenchPress, EX = window.BP_EXPLORER;
  const $ = (id) => document.getElementById(id);
  const el = (tag, attrs = {}, text) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) { if (k === "class") e.className = v; else e.setAttribute(k, v); }
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const pct = (k, n) => (n ? (100 * k / n).toFixed(1) : "");

  // ------------------------------------------------------------ providers (providers.py)
  const PROVIDERS = {
    openai: { label: "OpenAI", base: "https://api.openai.com/v1", key: true, ph: "gpt-4o-mini" },
    anthropic: { label: "Anthropic", base: "https://api.anthropic.com/v1", key: true, sdk: "anthropic", ph: "claude-model-id" },
    google: { label: "Google Gemini", base: "https://generativelanguage.googleapis.com/v1beta/openai", key: true, ph: "gemini-model-id" },
    mistral: { label: "Mistral", base: "https://api.mistral.ai/v1", key: true, ph: "mistral-small-latest" },
    xai: { label: "xAI", base: "https://api.x.ai/v1", key: true, ph: "grok-model-id" },
    groq: { label: "Groq", base: "https://api.groq.com/openai/v1", key: true, ph: "llama-3.1-8b-instant" },
    together: { label: "Together", base: "https://api.together.xyz/v1", key: true, ph: "org/model-id" },
    openrouter: { label: "OpenRouter", base: "https://openrouter.ai/api/v1", key: true, ph: "openai/gpt-oss-20b" },
    deepseek: { label: "DeepSeek", base: "https://api.deepseek.com/v1", key: true, ph: "deepseek-chat" },
    ollama: { label: "Ollama (on this computer)", base: "http://localhost:11434/v1", key: false, ph: "llama3.1:8b",
      note: "Ollama must allow requests from this page: start it with the environment variable OLLAMA_ORIGINS set to this site's address." },
    custom: { label: "Other OpenAI-compatible server", base: "", key: false, keyOptional: true, ph: "model-id",
      note: "Any server with an OpenAI-compatible /chat/completions endpoint (vLLM, LM Studio, an institutional gateway). It must allow requests from web pages (CORS)." },
  };
  const PARAM_HINTS = ["max_tokens", "max_completion_tokens", "temperature", "unsupported", "not supported"];
  const RETRYABLE = (s) => s === null || [408, 409, 425, 429].includes(s) || s >= 500;
  const MAX_RETRIES = 6, ABORT_AFTER_ERRORS = 25, TIMEOUT_MS = 300000;

  class HttpError extends Error { constructor(status, msg) { super(msg); this.status = status; } }

  function client(cfg) {
    const p = PROVIDERS[cfg.provider];
    const base = (cfg.baseUrl || p.base).replace(/\/+$/, "");
    let tokenParam = "max_tokens";            // switched to max_completion_tokens on the first rejection
    async function post(url, headers, body, signal) {
      let r;
      try {
        r = await fetch(url, { method: "POST", headers, body: JSON.stringify(body), signal, referrerPolicy: "no-referrer",
          credentials: "omit" });
      } catch (e) {
        if (e.name === "AbortError") throw e;
        const err = new HttpError(null, "network error: the request did not reach the provider, or the provider does "
          + "not accept requests from web pages (CORS)");
        err.network = true;
        throw err;
      }
      const txt = await r.text();
      let j = null;
      try { j = JSON.parse(txt); } catch (_) { /* not JSON */ }
      if (!r.ok) {
        const msg = (j && (j.error && (j.error.message || j.error) || j.message)) || txt.slice(0, 300);
        throw new HttpError(r.status, `HTTP ${r.status}: ${typeof msg === "string" ? msg : JSON.stringify(msg)}`);
      }
      return j;
    }
    async function callOpenAI(prompt, maxTokens, temperature, signal) {
      let last = null;
      const params = [tokenParam, tokenParam === "max_tokens" ? "max_completion_tokens" : "max_tokens"];
      for (const param of params) {
        const body = { model: cfg.model, messages: [{ role: "user", content: prompt }], [param]: maxTokens };
        if (temperature !== null) body.temperature = temperature;
        const headers = { "Content-Type": "application/json" };
        if (cfg.key) headers.Authorization = "Bearer " + cfg.key;
        let j;
        try { j = await post(base + "/chat/completions", headers, body, signal); } catch (e) {
          if (e.status === 400 && PARAM_HINTS.some((h) => e.message.toLowerCase().includes(h))) { last = e; continue; }
          throw e;
        }
        tokenParam = param;
        const ch = (j.choices || [])[0] || {}, msg = ch.message || {}, u = j.usage || null;
        return { response: msg.content || "", reasoning: msg.reasoning_content || msg.reasoning || null,
          model_version: j.model || cfg.model, finish_reason: ch.finish_reason ?? null,
          input_tokens: u ? u.prompt_tokens ?? null : null, output_tokens: u ? u.completion_tokens ?? null : null };
      }
      throw last;
    }
    async function callAnthropic(prompt, maxTokens, temperature, signal) {
      const body = { model: cfg.model, max_tokens: maxTokens, messages: [{ role: "user", content: prompt }] };
      if (temperature !== null) body.temperature = temperature;
      const j = await post(base + "/messages", { "Content-Type": "application/json", "x-api-key": cfg.key,
        "anthropic-version": "2023-06-01", "anthropic-dangerous-direct-browser-access": "true" }, body, signal);
      const text = (j.content || []).filter((b) => b.type === "text").map((b) => b.text).join("");
      return { response: text, reasoning: null, model_version: j.model || cfg.model, finish_reason: j.stop_reason ?? null,
        input_tokens: j.usage ? j.usage.input_tokens ?? null : null, output_tokens: j.usage ? j.usage.output_tokens ?? null : null };
    }
    const fn = p.sdk === "anthropic" ? callAnthropic : callOpenAI;
    return async function call(prompt, maxTokens, temperature, outerSignal) {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
      const onAbort = () => ctl.abort();
      if (outerSignal) outerSignal.addEventListener("abort", onAbort);
      const t0 = performance.now();
      try {
        const out = await fn(prompt, maxTokens, temperature, ctl.signal);
        out.latency_ms = Math.round(performance.now() - t0);
        return out;
      } catch (e) {
        if (e.name === "AbortError") throw new HttpError(outerSignal && outerSignal.aborted ? -1 : null,
          outerSignal && outerSignal.aborted ? "stopped" : "timeout");
        throw e;
      } finally { clearTimeout(timer); if (outerSignal) outerSignal.removeEventListener("abort", onAbort); }
    };
  }

  // ------------------------------------------------------------ benchmark data
  let ROWS = [], SOURCES = [], CELL_META = new Map();
  async function loadData() {
    let buf;
    try {
      const r = await fetch("data/core.jsonl.gz");
      if (!r.ok) throw new Error("HTTP " + r.status);
      buf = new Uint8Array(await r.arrayBuffer());
    } catch (e) {
      throw new Error("Could not load the benchmark (data/core.jsonl.gz). Open this page through a web server "
        + "(GitHub Pages, or python -m http.server in the docs folder), not as a local file.");
    }
    let text;
    if (buf[0] === 0x1f && buf[1] === 0x8b) {          // still compressed (the server did not decode it)
      const ds = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
      text = await new Response(ds).text();
    } else text = new TextDecoder().decode(buf);
    ROWS = text.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
    const src = new Map();
    for (const r of ROWS) {
      if (!src.has(r.source_id)) src.set(r.source_id, { id: r.source_id, name: r.source_name, cells: new Set() });
      src.get(r.source_id).cells.add(r.cell_id);
      if (!CELL_META.has(r.cell_id)) CELL_META.set(r.cell_id, { source: r.source_name, tid: r.tension_id, tension: r.tension_name });
    }
    SOURCES = [...src.values()].sort((a, b) => a.name.localeCompare(b.name));
  }
  // paper models: counts per cell (x model x configuration), no-rationale arm
  const PAPER = new Map((EX ? EX.cells : []).map((r) => [r[0], r[8]]));
  function paperAcc(m, cells, configs) {
    let k = 0, n = 0;
    for (const c of cells) { const v = PAPER.get(c); if (!v) continue;
      for (const f of configs) { const i = 2 * (m * 5 + f); k += v[i]; n += v[i + 1]; } }
    return [k, n];
  }

  // ------------------------------------------------------------ form
  const selectedCodes = new Set();
  let presetSample = null;
  function formState() {
    const provider = $("provider").value, p = PROVIDERS[provider];
    const model = $("model").value.trim();
    const t = $("temperature").value.trim(), mt = $("maxtokens").value.trim();
    const arms = [...document.querySelectorAll("input[name=arm]:checked")].map((x) => x.value);
    const configs = [...document.querySelectorAll("#configs input:checked")].map((x) => +x.value);
    const pairs = document.querySelector("input[name=pairs]:checked").value;
    const reasoning = $("reasoning").checked || B.isReasoningModel(model);
    const budgets = {};
    arms.forEach((a) => { const b = mt ? +mt : B.DEFAULT_MAX_TOKENS[a];
      budgets[a] = reasoning && !mt ? Math.max(b, B.REASONING_MIN_MAX_TOKENS) : b; });
    return { provider, p, model, key: $("apikey").value.trim(), baseUrl: $("baseurl").value.trim(),
      label: $("label").value.trim() || model, temperature: t === "" ? null : +t, budgets, reasoning,
      workers: Math.max(1, Math.min(16, +$("workers").value || 4)), arms, configs,
      pairsPerCell: pairs === "all" ? null : +pairs, replicates: Math.max(1, +$("replicates").value || 1),
      sources: [...selectedCodes], sample: presetSample };
  }
  function planStimuli(s) {
    return B.select(ROWS, { configurations: s.configs, sources: s.sources, pairsPerCell: s.pairsPerCell,
      sample: s.sample, seed: 0 });
  }
  function providerChanged() {
    const p = PROVIDERS[$("provider").value];
    $("model").placeholder = "e.g. " + p.ph;
    $("baseurl-row").hidden = !(p.base === "" || $("provider").value === "ollama");
    $("baseurl").placeholder = p.base || "https://your-server/v1";
    $("key-row").hidden = !(p.key || p.keyOptional);
    $("provider-note").textContent = p.note || "";
  }
  function applyPreset() {
    const v = $("preset").value;
    if (v === "custom") return;
    document.querySelectorAll("#configs input").forEach((x) => { x.checked = true; });
    document.querySelectorAll("input[name=arm]").forEach((x) => { x.checked = x.value === "letter" || v === "paper"; });
    document.querySelector(`input[name=pairs][value="${v === "lite" ? "1" : "all"}"]`).checked = true;
    $("replicates").value = v === "paper" ? 2 : 1;
    SOURCES.forEach((s) => selectedCodes.add(s.id));
    presetSample = v === "smoke" ? 40 : null;
    renderCodes();
  }
  function toCustom() { if ($("preset").value !== "custom") { $("preset").value = "custom"; presetSample = null; } }
  function renderCodes() {
    const q = $("codes-search").value.trim().toLowerCase(), box = $("codes");
    box.replaceChildren();
    SOURCES.filter((s) => !q || s.name.toLowerCase().includes(q)).forEach((s) => {
      const lab = el("label", { class: "check" });
      const cb = el("input", { type: "checkbox", value: s.id });
      cb.checked = selectedCodes.has(s.id);
      cb.addEventListener("change", () => { toCustom(); if (cb.checked) selectedCodes.add(s.id); else selectedCodes.delete(s.id); updatePlan(); });
      lab.append(cb, document.createTextNode(` ${s.name} (${s.cells.size})`));
      box.append(lab);
    });
    $("codes-count").textContent = `${selectedCodes.size} of ${SOURCES.length} selected`;
    updatePlan();
  }

  // ------------------------------------------------------------ plan
  let RESUME = { rows: [], done: new Set() };
  const taskKey = (sid, arm, rep) => `${sid}\u0000${arm}\u0000${rep}`;
  function tasksFor(s) {
    const stimuli = planStimuli(s), tasks = [];
    for (let rep = 1; rep <= s.replicates; rep++) for (const arm of s.arms) for (const st of stimuli) tasks.push([st, arm, rep]);
    return { stimuli, tasks, todo: tasks.filter(([st, arm, rep]) => !RESUME.done.has(taskKey(st.stimulus_id, arm, rep))) };
  }
  function updatePlan() {
    if (!ROWS.length) return;
    const s = formState();
    $("codes-count").textContent = `${selectedCodes.size} of ${SOURCES.length} selected`;
    const problems = [];
    if (!s.configs.length) problems.push("choose at least one configuration");
    if (!s.arms.length) problems.push("choose an answer format");
    if (!s.sources.length) problems.push("choose at least one code of ethics");
    if (problems.length) { $("plan").textContent = "To plan a run: " + problems.join("; ") + "."; $("run-btn").disabled = true;
      $("example").textContent = ""; return; }
    const { stimuli, tasks, todo } = tasksFor(s);
    const chars = todo.reduce((a, [st, arm]) => a + B.buildPrompt(st.prompt, arm).length, 0);
    const cells = new Set(stimuli.map((x) => x.cell_id)).size;
    $("plan").textContent = `${stimuli.length.toLocaleString("en")} stimuli from ${cells} code × tension cells in `
      + `${new Set(stimuli.map((x) => x.source_id)).size} codes → ${tasks.length.toLocaleString("en")} calls`
      + (tasks.length - todo.length ? `, ${(tasks.length - todo.length).toLocaleString("en")} already done` : "")
      + `, ${todo.length.toLocaleString("en")} to run. About ${(chars / 4 / 1e6).toFixed(2)}M input tokens; output budget per call: `
      + s.arms.map((a) => `${a} ${s.budgets[a]}`).join(", ") + (s.reasoning ? " (reasoning model)" : "")
      + (s.sample ? ". Random sample of 40 (not the same sample as the command-line smoke preset)" : "") + ".";
    $("run-btn").disabled = running || !todo.length;
    const ex = stimuli[0];
    $("example").textContent = ex ? `${ex.stimulus_id} · arm ${s.arms[0]} · code's side ${ex.gold_letter}\n\n`
      + B.buildPrompt(ex.prompt, s.arms[0]) : "";
  }

  // ------------------------------------------------------------ run
  let running = false, ctl = null, ROWS_OUT = [];
  const utcNow = () => new Date().toISOString().slice(0, 19) + "+00:00";
  const sleep = (ms, signal) => new Promise((res) => { const t = setTimeout(res, ms);
    if (signal) signal.addEventListener("abort", () => { clearTimeout(t); res(); }, { once: true }); });
  function checkModel(s) {
    if (!s.model) return "Enter the model id.";
    if (s.p.key && !s.key) return "Enter your API key.";
    if (!s.p.base && !s.baseUrl) return "Enter the server's base URL.";
    return null;
  }
  async function testConnection() {
    const s = formState(), out = $("test-out"), msg = checkModel(s);
    if (msg) { out.textContent = msg; return; }
    out.textContent = "Sending one request…";
    const st = ROWS[0], call = client(s);
    try {
      const r = await call(B.buildPrompt(st.prompt, "letter"), s.reasoning ? B.REASONING_MIN_MAX_TOKENS : 32, s.temperature);
      const [c] = B.parseResponse(r.response);
      out.textContent = `Connected to ${r.model_version} in ${r.latency_ms} ms. Answer: "${r.response.slice(0, 80)}"`
        + (c ? ` → parsed as ${c}.` : " → not parsed as A or B" + (r.finish_reason === "length" ? " (it hit the output limit: tick 'reasoning model' or raise the budget)." : "."));
    } catch (e) {
      out.textContent = "Failed: " + e.message + (e.network ? ". Check the provider and base URL; if they are right, this provider does not "
        + "accept requests from web pages, so use the command-line runner." : "");
    }
  }
  async function run() {
    const s = formState(), msg = checkModel(s);
    if (msg) { $("progress").textContent = msg; return; }
    if (RESUME.rows.length && RESUME.model && RESUME.model !== s.model) {
      $("progress").textContent = `The loaded results are from ${RESUME.model}; use the same model id to resume, or reload the page.`;
      return;
    }
    const { todo } = tasksFor(s);
    if (todo.length > 2000 && !confirm(`Send ${todo.length.toLocaleString("en")} requests to ${s.p.label}? They are billed to your account.`)) return;
    if ($("remember").checked) { try { sessionStorage.setItem("bp-key-" + s.provider, s.key); } catch (_) { /* storage blocked */ } }
    running = true; ctl = new AbortController();
    ROWS_OUT = RESUME.rows.slice();
    setButtons();
    const call = client(s), log = $("log");
    log.hidden = true; log.textContent = "";
    let n = 0, errors = 0, consecutive = 0, unparsed = 0, correct = 0, truncated = 0, i = 0;
    const t0 = performance.now();
    async function work([st, arm, rep]) {
      const prompt = B.buildPrompt(st.prompt, arm);
      const row = { stimulus_id: st.stimulus_id, track: "core", arm, replicate: rep, model: s.label, provider: s.provider,
        model_id: s.model, max_tokens: s.budgets[arm], temperature: s.temperature, prompt_sha256_8: await B.sha8(prompt) };
      let res = null, err = null, netTries = 0;
      for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
        if (ctl.signal.aborted) return null;
        try { res = await call(prompt, s.budgets[arm], s.temperature, ctl.signal); break; } catch (e) {
          err = e;
          if (e.status === -1) return null;                                   // stopped by the user
          if (e.network && ++netTries > 2) break;                             // a blocked request will not recover
          if (!RETRYABLE(e.status ?? null) || attempt === MAX_RETRIES) break;
          await sleep(Math.min(60, 2 * 2 ** attempt) * (0.5 + Math.random()) * 1000, ctl.signal);
        }
      }
      if (!res) return Object.assign(row, { error: `${err.name === "HttpError" ? "APIError" : err.name}: ${String(err.message).slice(0, 500)}`,
        timestamp_utc: utcNow() });
      const [choice, rationale] = B.parseResponse(res.response);
      return Object.assign(row, { model_version: res.model_version, raw_response: res.response, chosen_option: choice,
        rationale: arm === "rationale" ? rationale : null, is_correct: choice === null ? null : choice === st.gold_letter,
        finish_reason: res.finish_reason, input_tokens: res.input_tokens, output_tokens: res.output_tokens,
        latency_ms: res.latency_ms, reasoning: res.reasoning, error: null, timestamp_utc: utcNow() });
    }
    async function worker() {
      while (i < todo.length && !ctl.signal.aborted) {
        const task = todo[i++];
        const row = await work(task);
        if (!row) continue;
        ROWS_OUT.push(row); n++;
        if (row.error) {
          errors++; consecutive++;
          if (consecutive === 1 || errors <= 3) { log.hidden = false; log.textContent += `error on ${row.stimulus_id}: ${row.error.slice(0, 300)}\n`; }
          if (consecutive >= ABORT_AFTER_ERRORS) {
            ctl.abort();
            log.textContent += `Stopped after ${consecutive} consecutive errors. Check the key, the model id and your quota, then press Start to resume.\n`;
          }
        } else {
          consecutive = 0;
          unparsed += row.chosen_option === null;
          correct += row.is_correct === true;
          truncated += ["length", "max_tokens"].includes(row.finish_reason) && row.chosen_option === null;
        }
        progress();
      }
    }
    let lastDraw = 0;
    function progress(final) {
      const now = performance.now();
      if (!final && now - lastDraw < 400) return;
      lastDraw = now;
      const rate = n / Math.max((now - t0) / 1000, 1e-9), parsed = n - errors - unparsed;
      $("bar").style.width = (100 * n / Math.max(todo.length, 1)).toFixed(1) + "%";
      $("progress").textContent = `${n.toLocaleString("en")} / ${todo.length.toLocaleString("en")} calls · ${rate.toFixed(1)} calls/s · `
        + `ETA ${Math.round((todo.length - n) / Math.max(rate, 1e-9) / 60)} min · code-aligned so far ${pct(correct, parsed) || "–"}% · `
        + `unparsed ${unparsed} · errors ${errors}`;
      if (now - lastResults > 3000 || final) { lastResults = now; results(); }
    }
    let lastResults = 0;
    window.addEventListener("beforeunload", warn);
    await Promise.all(Array.from({ length: Math.min(s.workers, todo.length) }, worker));
    window.removeEventListener("beforeunload", warn);
    running = false;
    progress(true);
    $("progress").textContent += ctl.signal.aborted ? " · stopped (press Start to resume)" : " · done";
    if (truncated) { log.hidden = false; log.textContent += `${truncated} answers hit the output limit before giving a letter: `
      + "tick 'Treat as a reasoning model' or raise the output budget.\n"; }
    RESUME = { rows: ROWS_OUT.slice(), done: new Set(ROWS_OUT.filter((r) => !r.error).map((r) => taskKey(r.stimulus_id, r.arm, r.replicate))),
      model: s.model };
    setButtons(); updatePlan();
  }
  const warn = (e) => { e.preventDefault(); e.returnValue = ""; };
  function setButtons() {
    $("run-btn").disabled = running; $("stop-btn").disabled = !running;
    $("dl-btn").disabled = !ROWS_OUT.length; $("resume").disabled = running;
  }
  function download(name, text, type) {
    const a = el("a", { href: URL.createObjectURL(new Blob([text], { type })), download: name });
    document.body.append(a); a.click(); a.remove();
  }
  async function resumeFrom(file) {
    const rows = (await file.text()).split("\n").filter((l) => l.trim()).map((l) => { try { return JSON.parse(l); } catch (_) { return null; } })
      .filter((r) => r && r.stimulus_id);
    const models = [...new Set(rows.map((r) => r.model_id))];
    if (models.length > 1) { $("progress").textContent = `This file holds answers from ${models.join(", ")}; load one model at a time.`; return; }
    RESUME = { rows, done: new Set(rows.filter((r) => !r.error).map((r) => taskKey(r.stimulus_id, r.arm, r.replicate))), model: models[0] };
    ROWS_OUT = rows.slice();
    if (models[0] && !$("model").value) $("model").value = models[0];
    $("progress").textContent = `Loaded ${rows.length.toLocaleString("en")} answers from ${file.name} (${RESUME.done.size.toLocaleString("en")} without error). `
      + "Matching calls will be skipped; the new answers are added to these in the downloaded file.";
    setButtons(); updatePlan(); results();
  }

  // ------------------------------------------------------------ results
  const BY_ID = () => new Map(ROWS.map((r) => [r.stimulus_id, r]));
  let SUMMARY = [];
  function table(box, cols, rows) {
    box.replaceChildren();
    let sortCol = -1, dir = 1;
    const t = el("table"), thead = el("thead"), tbody = el("tbody"), hr = el("tr");
    const isNum = cols.map((_, j) => rows.length && rows.every((r) => r[j] === "" || !isNaN(parseFloat(r[j]))));
    cols.forEach((c, j) => { const th = el("th", { scope: "col", tabindex: "0" }, c);
      th.addEventListener("click", () => { dir = sortCol === j ? -dir : 1; sortCol = j; draw(); }); hr.append(th); });
    thead.append(hr); t.append(thead, tbody); box.append(t);
    function draw() {
      const r = rows.slice();
      if (sortCol >= 0) r.sort((a, b) => isNum[sortCol] ? dir * ((parseFloat(a[sortCol]) || -1) - (parseFloat(b[sortCol]) || -1))
        : dir * String(a[sortCol]).localeCompare(String(b[sortCol])));
      tbody.replaceChildren(...r.map((row) => { const tr = el("tr");
        row.forEach((v, j) => tr.append(el("td", { class: isNum[j] ? "num" : "" }, String(v)))); return tr; }));
    }
    draw();
  }
  let BYID = null;
  function results() {
    if (!ROWS_OUT.length) return;
    $("results-card").hidden = false;
    BYID = BYID || BY_ID();
    const models = EX ? EX.models : [];
    // one pass: counts per (arm, configuration), per code and per cell (the comparison arm only)
    const arms = [...new Set(ROWS_OUT.map((r) => r.arm))].sort();
    const arm0 = arms.includes("letter") ? "letter" : arms[0];
    const G = (m, key) => { if (!m.has(key)) m.set(key, { k: 0, n: 0, u: 0, e: 0, cells: new Set(), cfgs: new Set() }); return m.get(key); };
    const byCfg = new Map(), bySrc = new Map(), byCell = new Map(), srcName = new Map();
    for (const r of ROWS_OUT) {
      const st = BYID.get(r.stimulus_id);
      if (!st) continue;
      const groups = [G(byCfg, r.arm + "|" + st.configuration)];
      if (r.arm === arm0) { groups.push(G(bySrc, st.source_id), G(byCell, st.cell_id)); srcName.set(st.source_id, st.source_name); }
      for (const g of groups) {
        if (r.error) { g.e++; continue; }
        g.cells.add(st.cell_id); g.cfgs.add(st.configuration);
        if (r.chosen_option === null) g.u++; else { g.n++; g.k += r.is_correct ? 1 : 0; }
      }
    }
    const paperCols = (cells, cfgs) => models.map((_, m) => { const [k, n] = paperAcc(m, cells, [...cfgs]); return pct(k, n); });
    const cfgRows = [];
    for (const arm of arms) for (let c = 0; c < 5; c++) {
      const a = byCfg.get(arm + "|" + c);
      if (a) cfgRows.push([`C${c}`, arm, pct(a.k, a.n), a.n, a.u, a.e, ...paperCols(a.cells, [c])]);
    }
    table($("res-config"), ["Config", "Arm", "Your model %", "Parsed", "Unparsed", "Errors", ...models.map((m) => m + " %")], cfgRows);
    const codeRows = [...bySrc].map(([sid, a]) => [srcName.get(sid), pct(a.k, a.n), a.n, ...paperCols(a.cells, a.cfgs)]);
    table($("res-codes"), ["Code", `Your model % (${arm0})`, "Parsed", ...models.map((m) => m + " %")], codeRows.sort((x, y) => (+x[1] || 0) - (+y[1] || 0)));
    const cellRows = [...byCell].map(([cid, a]) => { const m = CELL_META.get(cid);
      return [m.source, `${m.tid}: ${m.tension}`, pct(a.k, a.n), a.n, ...paperCols([cid], a.cfgs)]; });
    const rows = ROWS_OUT.filter((r) => !r.error);
    table($("res-cells"), ["Code", "Tension", `Your model % (${arm0})`, "Parsed", ...models.map((m) => m + " %")],
      cellRows.sort((x, y) => (+x[2] || 0) - (+y[2] || 0)));
    SUMMARY = [["level", "config_or_code", "tension", "arm", "your_model_pct", "parsed", ...models], ...cfgRows.map((r) => ["configuration", r[0], "", r[1], r[2], r[3], ...r.slice(6)]),
      ...codeRows.map((r) => ["code", r[0], "", arm0, r[1], r[2], ...r.slice(3)]), ...cellRows.map((r) => ["tension", r[0], r[1], arm0, r[2], r[3], ...r.slice(4)])];
    // unparsed
    const un = rows.filter((r) => r.chosen_option === null), box = $("res-unparsed");
    box.replaceChildren();
    if (!un.length) { box.append(el("p", { class: "hint" }, "None.")); return; }
    box.append(el("p", { class: "hint" }, `${un.length} answers had no A/B (they are left out of accuracy). First ${Math.min(10, un.length)}:`));
    const ul = el("ul");
    un.slice(0, 10).forEach((r) => ul.append(el("li", {}, `${r.stimulus_id} (${B.classifyUnparsed(r.raw_response)}): ${String(r.raw_response || "").slice(0, 200)}`)));
    box.append(ul);
  }

  // ------------------------------------------------------------ init
  async function init() {
    Object.entries(PROVIDERS).forEach(([k, p]) => $("provider").append(el("option", { value: k }, p.label)));
    ["C0 no framing", "C1 journalism role", "C2 + scenarios", "C3 + policy text", "C4 + policy and scenarios"].forEach((lab, c) => {
      const l = el("label", { class: "check" }), cb = el("input", { type: "checkbox", value: String(c) });
      cb.checked = c === 0 || c === 4;
      cb.addEventListener("change", () => { toCustom(); updatePlan(); });
      l.append(cb, document.createTextNode(" " + lab)); $("configs").append(l);
    });
    $("provider").addEventListener("change", () => {
      providerChanged();
      try { const k = sessionStorage.getItem("bp-key-" + $("provider").value); if (k) { $("apikey").value = k; $("remember").checked = true; } } catch (_) { /* no storage */ }
      updatePlan();
    });
    $("remember").addEventListener("change", () => { if (!$("remember").checked) { try { sessionStorage.removeItem("bp-key-" + $("provider").value); } catch (_) { /* no storage */ } } });
    ["model", "temperature", "maxtokens", "reasoning", "workers"].forEach((id) => $(id).addEventListener("input", updatePlan));
    document.querySelectorAll("input[name=arm], input[name=pairs]").forEach((x) => x.addEventListener("change", () => { toCustom(); updatePlan(); }));
    $("replicates").addEventListener("input", () => { toCustom(); updatePlan(); });
    $("preset").addEventListener("change", () => { applyPreset(); updatePlan(); });
    $("codes-search").addEventListener("input", renderCodes);
    $("codes-all").addEventListener("click", () => { toCustom(); document.querySelectorAll("#codes input").forEach((cb) => selectedCodes.add(cb.value)); renderCodes(); });
    $("codes-none").addEventListener("click", () => { toCustom(); selectedCodes.clear(); renderCodes(); });
    $("test-btn").addEventListener("click", testConnection);
    $("run-btn").addEventListener("click", run);
    $("stop-btn").addEventListener("click", () => ctl && ctl.abort());
    $("dl-btn").addEventListener("click", () => download("results.jsonl", ROWS_OUT.map((r) => JSON.stringify(r)).join("\n") + "\n", "application/x-ndjson"));
    $("csv-btn").addEventListener("click", () => download("bench_press_summary.csv",
      String.fromCharCode(0xfeff) + SUMMARY.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n"), "text/csv"));
    $("resume").addEventListener("change", (e) => { if (e.target.files[0]) resumeFrom(e.target.files[0]); e.target.value = ""; });
    providerChanged();
    try {
      await loadData();
    } catch (e) { $("load-status").textContent = e.message; return; }
    $("load-status").hidden = true; $("setup").hidden = false; $("plan-card").hidden = false;
    const af = SOURCES.find((s) => s.id.startsWith("afp")) || SOURCES[0];
    selectedCodes.add(af.id);
    renderCodes(); setButtons();
  }
  init();
})();
