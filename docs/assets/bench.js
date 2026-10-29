/* Bench Press in the browser: prompt building and answer parsing, ported line by line from the Python package
   (benchpress/prompts.py and benchpress/parsing.py). Results must match the command-line runner, so do not change
   anything here without changing the Python code too; tests/parser_cases are checked against both. */
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- prompts.py
  const FINAL_INSTRUCTION = /(?:Choose A or B\.\s*)?Respond with only\s+["']?A["']?\s+or\s+["']?B["']?\.\s*$/gim;
  const RATIONALE_INSTRUCTION = "Choose A or B. Answer in English, in this format:\n<LETTER> - <rationale of three to five sentences>";
  const DEFAULT_MAX_TOKENS = { letter: 32, rationale: 512 };
  const REASONING_RE = /(^|\/)(o\d|gpt-5|gpt-oss)|deepseek-r1|deepseek-reasoner|qwq|-r1\b|thinking|reasoning/i;
  const REASONING_MIN_MAX_TOKENS = 4096;

  // Python str.strip()/rstrip() whitespace (str.isspace): ASCII \t\n\v\f\r space, \x1c-\x1f, \x85, \xa0 and the
  // Unicode separators. JavaScript's trim() differs at the edges (it strips the byte-order mark U+FEFF, keeps \x1c-\x1f, \x85).
  const PY_WS = "\\t\\n\\x0b\\x0c\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
  const RSTRIP = new RegExp("[" + PY_WS + "]+$"), LSTRIP = new RegExp("^[" + PY_WS + "]+");
  const pyStrip = (s) => s.replace(LSTRIP, "").replace(RSTRIP, "");
  const pyRstrip = (s) => s.replace(RSTRIP, "");

  function buildPrompt(stimulusPrompt, arm) {
    if (arm === "letter") return stimulusPrompt;
    if (arm === "rationale") return pyRstrip(stimulusPrompt.replace(FINAL_INSTRUCTION, "")) + "\n\n" + RATIONALE_INSTRUCTION;
    throw new Error("unknown arm " + arm);
  }
  const isReasoningModel = (id) => REASONING_RE.test(id || "");

  // ---------------------------------------------------------------- parsing.py
  // Python patterns are compiled with IGNORECASE | DOTALL ('is'). The last one keeps the letter case-sensitive
  // ((?-i:[AB]) in Python); JavaScript has no inline flags in older browsers, so that pattern captures [AaBb]
  // and a match whose letter is lower case is skipped, as Python's would be (see searchLast).
  const PATTERNS = [
    /^\s*<\s*(?:LETTER|LETTERA|LETTRE)\s*>\s*[:\-–—]?\s*([AB])\b\s*(?:[-–—:.)]?\s*)?(.*)$/is,
    /^\s*<\s*(?:(?:LETTER|LETTERA|LETTRE)\s*[:\-]?\s*)?([AB])\s*>\s*(?:[-–—:.)]?\s*)?(.*)$/is,
    /^\s*(?:LETTER|LETTERA|LETTRE)\s*[:\-]?\s*([AB])\b\s*(?:[-–—:.)]?\s*)?(.*)$/is,
    /^\s*(?:answer|choice|risposta|réponse|selected|selection)\s*[:\-–—]?\s*(?:option\s*|opzione\s*|version\s*)?([AB])\b\s*(?:[-–—:.)]?\s*)?(.*)$/is,
    /^\s*(?:the\s+)?(?:answer|risposta|réponse)\s+is\s+(?:option\s*)?([AB])\b\s*(.*)$/is,
    /^\s*(?:option|opzione|version|versione|versione)\s+([AB])\b\s*(?:[-–—:.)]?\s*)?(.*)$/is,
    /^\s*[\[(]\s*([AB])\s*[\])]\s*(?:[-–—:.)]?\s*)?(.*)$/is,
    /^\s*([AB])\s*(?:[-–—:.)]\s*|\s+(?:because|perch[éè]|parce que|car)\s+|$)(.*)$/is,
  ];
  const LAST = /\b(?:i\s+)?(?:choose|select|pick|prefer|scelgo|scelta|choisis|préfère)\s+(?:option\s+|opzione\s+|version\s+|versione\s+)?([AaBb])\b\s*(.*)$/gis;
  function searchLast(target) {
    LAST.lastIndex = 0;
    let m;
    while ((m = LAST.exec(target)) !== null) {
      if (m[1] === "A" || m[1] === "B") return m;
      LAST.lastIndex = m.index + 1;               // try the next start position, as re.search does
    }
    return null;
  }
  // Python str.splitlines() line boundaries
  const SPLITLINES = /\r\n|[\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]/;
  const splitlines = (s) => { const l = s.split(SPLITLINES); if (l.length && l[l.length - 1] === "" && SPLITLINES.test(s.slice(-1))) l.pop(); return l; };
  const REFUSAL_RE = /^\W*(I.m (sorry|not able|unable)|I can.t|I don.t know)/i;
  const NEITHER_RE = /^\W*(neither|none|no publication)\b/i;
  const TOOL_RE = /\b(search|browse|query)\b/i;

  function parseResponse(raw) {
    if (raw === null || raw === undefined) return [null, null];
    const text = pyStrip(String(raw));
    if (!text) return [null, null];
    const textForMatch = text.replace(/[*_]+/g, "");
    const firstLine = pyStrip(splitlines(textForMatch)[0] || "");
    const search = Array.from(textForMatch).slice(0, 400).join("");
    const rationale = (rest) => {
      rest = pyStrip(rest).replace(/^[-–—:.) ]+/, "");
      rest = pyStrip(rest);
      if (rest.startsWith("<")) {
        rest = pyRstrip(rest.slice(1));
        if (rest.endsWith(">")) rest = pyRstrip(rest.slice(0, -1));
      }
      if (rest) return rest;
      const restLines = pyStrip(splitlines(text).slice(1).join("\n"));
      return restLines || null;
    };
    for (const target of [firstLine, search]) {
      for (const pat of PATTERNS) {
        const m = pat.exec(target);
        if (m) return [m[1].toUpperCase(), rationale(m[2] || "")];
      }
      const m = searchLast(target);
      if (m) return [m[1].toUpperCase(), rationale(m[2] || "")];
    }
    const cands = firstLine.toUpperCase().match(/\b([AB])\b/g) || [];
    if (cands.length === 1 && Array.from(firstLine).length <= 40) return [cands[0], firstLine === cands[0] ? null : text];
    return [null, text];
  }
  function classifyUnparsed(raw) {
    const t = pyStrip(raw || "");
    if (!t) return "empty";
    if (REFUSAL_RE.test(t)) return "refusal";
    if (NEITHER_RE.test(t)) return "neither / none";
    if (TOOL_RE.test(t.slice(0, 80))) return "leaked tool-use text";
    return "other";
  }

  // ---------------------------------------------------------------- data.py: select()
  function select(rows, o) {
    let out = rows;
    if (o.configurations) { const k = new Set(o.configurations); out = out.filter((r) => k.has(r.configuration)); }
    if (o.sources && o.sources.length) { const k = new Set(o.sources); out = out.filter((r) => k.has(r.source_id)); }
    if (o.paraphrases && o.paraphrases.length) { const k = new Set(o.paraphrases); out = out.filter((r) => k.has(r.paraphrase_id)); }
    if (o.pairsPerCell) {
      const per = new Map();
      for (const r of out) { if (!per.has(r.cell_id)) per.set(r.cell_id, []); const l = per.get(r.cell_id); if (!l.includes(r.paraphrase_id)) l.push(r.paraphrase_id); }
      const keep = new Set();
      for (const [c, ps] of per) ps.slice().sort().slice(0, o.pairsPerCell).forEach((p) => keep.add(c + "\u0000" + p));
      out = out.filter((r) => keep.has(r.cell_id + "\u0000" + r.paraphrase_id));
    }
    if (o.sample != null && o.sample < out.length) {       // seeded shuffle (not the same sample as Python's)
      let s = (o.seed || 0) + 0x9e3779b9;
      const rnd = () => { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      const a = out.slice();
      for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
      out = a.slice(0, o.sample).sort((x, y) => (x.stimulus_id < y.stimulus_id ? -1 : 1));
    }
    return out;
  }

  async function sha8(text) {
    const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return Array.from(new Uint8Array(d)).slice(0, 4).map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  root.BenchPress = { buildPrompt, parseResponse, classifyUnparsed, select, sha8, isReasoningModel,
    DEFAULT_MAX_TOKENS, REASONING_MIN_MAX_TOKENS, RATIONALE_INSTRUCTION };
})(window);
