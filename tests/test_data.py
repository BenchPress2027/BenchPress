import collections
import hashlib
import json
from pathlib import Path

from benchpress.data import load_track, select

DATA = Path(__file__).resolve().parent.parent / "data"


def test_manifest_checksums():
    manifest = json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == info["sha256"], name


def test_core_shape():
    core = load_track("core")
    assert len(core) == 36590
    assert len({r["stimulus_id"] for r in core}) == len(core)
    assert len({r["cell_id"] for r in core}) == 919
    assert len({r["item_id"] for r in core}) == 3659
    assert len({r["source_id"] for r in core}) == 103
    assert collections.Counter(r["gold_letter"] for r in core) == {"A": 18295, "B": 18295}
    # every pair appears in 5 configurations x 2 orders, and the gold letter flips with the order
    by_item = collections.defaultdict(dict)
    for r in core:
        by_item[r["item_id"]][(r["configuration"], r["order"])] = r["gold_letter"]
    for item, d in by_item.items():
        assert len(d) == 10, item
        for c in range(5):
            assert d[(c, "AB")] != d[(c, "BA")], item


def test_l2_links_to_core():
    core_ids = {r["stimulus_id"] for r in load_track("core")}
    l2 = load_track("l2")
    assert all(r["baseline_stimulus_id"] in core_ids for r in l2)
    assert {r["configuration"] for r in l2} == {1, 4}
    assert sum(r["in_main_analysis"] for r in l2) == 19296


def test_pairs_per_cell_keeps_orders_together():
    sub = select(load_track("core"), pairs_per_cell=1)
    assert len({r["cell_id"] for r in sub}) == 919
    assert len(sub) == 919 * 10
