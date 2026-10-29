import pytest

from benchpress.parsing import classify_unparsed, parse_response
from benchpress.prompts import build_prompt


@pytest.mark.parametrize("raw, choice", [
    ("A", "A"),
    ("B", "B"),
    ("B - Version B protects the source.", "B"),
    ("**A** - It names the official.", "A"),
    ("<LETTER> A - because", "A"),
    ("<B> - rationale", "B"),
    ("Answer: B", "B"),
    ("Option A. It is more accurate.", "A"),
    ("I would choose B because it is verified.", "B"),
    ("Between the two passages I would pick version B for publication.", "B"),
    ("I'm sorry, but I can't help with that.", None),
    ("Neither: both breach the code.", None),
    ("", None),
])
def test_parse(raw, choice):
    assert parse_response(raw)[0] == choice


def test_rationale_extracted():
    choice, rationale = parse_response("B - Version B protects the minor. It omits the name.")
    assert choice == "B" and rationale.startswith("Version B protects")


def test_unparsed_kinds():
    assert classify_unparsed("") == "empty"
    assert classify_unparsed("I'm sorry, but I can't help with that.") == "refusal"
    assert classify_unparsed("None of them.") == "neither / none"


def test_rationale_arm_replaces_final_instruction():
    c1 = 'Below are two versions.\n\nA: x\nB: y\n\nRespond with only "A" or "B".'
    c0 = 'A: x\nB: y\n\nChoose A or B. Respond with only "A" or "B".'
    for p in (c1, c0):
        out = build_prompt(p, "rationale")
        assert "Respond with only" not in out
        assert out.endswith("<LETTER> - <rationale of three to five sentences>")
        assert out.count("Choose A or B.") == 1
    assert build_prompt(c1, "letter") == c1
