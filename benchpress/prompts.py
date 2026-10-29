"""The two answer formats (arms) of the benchmark.

Every stimulus is stored with the letter-only instruction at the end
('Respond with only "A" or "B".'). The rationale arm replaces that line with a
request for the letter first, then a three-to-five-sentence rationale. Letter
first keeps the rationale a justification of a choice already made; the wording
is neutral (it does not ask the model to name a principle or value).
"""
import re

ARMS = ("letter", "rationale")

_FINAL_INSTRUCTION = re.compile(
    r'(?:Choose A or B\.\s*)?Respond with only\s+["\']?A["\']?\s+or\s+["\']?B["\']?\.\s*$',
    re.IGNORECASE | re.MULTILINE)

RATIONALE_INSTRUCTION = ("Choose A or B. Answer in English, in this format:\n"
                         "<LETTER> - <rationale of three to five sentences>")

# Output-token budgets used in the paper. Reasoning models need far more (see providers.py).
DEFAULT_MAX_TOKENS = {"letter": 32, "rationale": 512}


def build_prompt(stimulus_prompt: str, arm: str) -> str:
    """Return the exact text sent to the model for this arm."""
    if arm == "letter":
        return stimulus_prompt
    if arm == "rationale":
        stripped = _FINAL_INSTRUCTION.sub("", stimulus_prompt).rstrip()
        return f"{stripped}\n\n{RATIONALE_INSTRUCTION}"
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
