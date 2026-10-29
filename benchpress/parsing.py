"""Parse a model's answer into a choice (A/B) and an optional rationale.

This is the parser used for every number in the paper. Do not change it if you
want results that are comparable with the paper's: answers it cannot parse
count as unparsed and are excluded from accuracy (they are reported separately).
"""
import re
from typing import Optional, Tuple

_PATTERNS = [
    # Template echo: the model copies the literal "<LETTER>" placeholder before the answer,
    # e.g. "<LETTER> A - rationale...".
    r"^\s*<\s*(?:LETTER|LETTERA|LETTRE)\s*>\s*[:\-–—]?\s*([AB])\b\s*(?:[-–—:.)]?\s*)?(.*)$",
    # Letter written inside the placeholder brackets: "<B> - ...", "< A > - ...", "<LETTER A> - ...".
    r"^\s*<\s*(?:(?:LETTER|LETTERA|LETTRE)\s*[:\-]?\s*)?([AB])\s*>\s*(?:[-–—:.)]?\s*)?(.*)$",
    r"^\s*(?:LETTER|LETTERA|LETTRE)\s*[:\-]?\s*([AB])\b\s*(?:[-–—:.)]?\s*)?(.*)$",
    r"^\s*(?:answer|choice|risposta|réponse|selected|selection)\s*[:\-–—]?\s*(?:option\s*|opzione\s*|version\s*)?([AB])\b\s*(?:[-–—:.)]?\s*)?(.*)$",
    r"^\s*(?:the\s+)?(?:answer|risposta|réponse)\s+is\s+(?:option\s*)?([AB])\b\s*(.*)$",
    r"^\s*(?:option|opzione|version|versione|versione)\s+([AB])\b\s*(?:[-–—:.)]?\s*)?(.*)$",
    r"^\s*[\[(]\s*([AB])\s*[\])]\s*(?:[-–—:.)]?\s*)?(.*)$",
    r"^\s*([AB])\s*(?:[-–—:.)]\s*|\s+(?:because|perch[éè]|parce que|car)\s+|$)(.*)$",
    # Unanchored "I choose A" comes last, and its letter is case-sensitive, so that the article
    # "a" in "prefer a version ..." is not read as option A.
    r"\b(?:i\s+)?(?:choose|select|pick|prefer|scelgo|scelta|choisis|préfère)\s+(?:option\s+|opzione\s+|version\s+|versione\s+)?((?-i:[AB]))\b\s*(.*)$",
]

_REFUSAL_RE = re.compile(r"^\W*(I.m (sorry|not able|unable)|I can.t|I don.t know)", re.I)
_NEITHER_RE = re.compile(r"^\W*(neither|none|no publication)\b", re.I)
_TOOL_RE = re.compile(r"\b(search|browse|query)\b", re.I)


def parse_response(raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (choice, rationale). choice is "A", "B" or None (unparsed)."""
    if raw is None:
        return None, None
    text = str(raw).strip()
    if not text:
        return None, None

    # Strip markdown emphasis (**B** - ...) so it does not block detection.
    text_for_match = re.sub(r"[*_]+", "", text)
    first_line = text_for_match.splitlines()[0].strip()
    search = text_for_match[:400]

    def _rationale(rest: str) -> Optional[str]:
        rest = rest.strip().lstrip("-–—:.) ").strip()
        if rest.startswith("<"):
            rest = rest[1:].rstrip()
            if rest.endswith(">"):
                rest = rest[:-1].rstrip()
        if rest:
            return rest
        rest_lines = "\n".join(text.splitlines()[1:]).strip()
        return rest_lines or None

    for target in (first_line, search):
        for pat in _PATTERNS:
            m = re.search(pat, target, flags=re.IGNORECASE | re.DOTALL)
            if m:
                choice = m.group(1).upper()
                rest = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
                return choice, _rationale(rest)

    # Fallback: a single bare letter in a short first line.
    candidates = re.findall(r"\b([AB])\b", first_line.upper())
    if len(candidates) == 1 and len(first_line) <= 40:
        return candidates[0], None if first_line == candidates[0] else text

    return None, text


def classify_unparsed(raw: Optional[str]) -> str:
    """Kind of an answer that has no A/B: empty, refusal, neither / none, leaked tool-use text, other."""
    t = (raw or "").strip()
    if not t:
        return "empty"
    if _REFUSAL_RE.match(t):
        return "refusal"
    if _NEITHER_RE.match(t):
        return "neither / none"
    if _TOOL_RE.search(t[:80]):
        return "leaked tool-use text"
    return "other"
