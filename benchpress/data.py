"""Loading the frozen benchmark files and selecting subsets of them."""
import gzip
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

TRACKS = {"core": "core.jsonl.gz", "l2": "l2.jsonl.gz"}


def data_dir(override: Optional[str] = None) -> Path:
    """The data/ folder: --data-dir, else $BENCHPRESS_DATA, else the repository's data/ folder."""
    d = Path(override or os.environ.get("BENCHPRESS_DATA") or Path(__file__).resolve().parent.parent / "data")
    if not (d / TRACKS["core"]).exists():
        raise FileNotFoundError(f"Benchmark data not found in {d}. Run from a clone of the repository, "
                                f"or point --data-dir / BENCHPRESS_DATA at its data/ folder.")
    return d


def load_track(track: str, directory: Optional[str] = None) -> List[dict]:
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}; expected one of {list(TRACKS)}")
    with gzip.open(data_dir(directory) / TRACKS[track], "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index_by_id(rows: Iterable[dict]) -> Dict[str, dict]:
    return {r["stimulus_id"]: r for r in rows}


def select(rows: List[dict], configurations=None, sources=None, cells=None, paraphrases=None,
           pairs_per_cell: Optional[int] = None, sample: Optional[int] = None, l2_all: bool = False,
           seed: int = 0, limit: Optional[int] = None) -> List[dict]:
    """Subset the stimuli. All filters keep both option orders of a pair together, except `sample`
    and `limit`, which are meant for smoke tests."""
    out = rows
    if "l2_condition" in (rows[0] if rows else {}) and not l2_all:
        out = [r for r in out if r["in_main_analysis"]]
    if configurations is not None:
        keep = {int(c) for c in configurations}
        out = [r for r in out if r["configuration"] in keep]
    if sources:
        keep = set(sources)
        out = [r for r in out if r["source_id"] in keep]
    if cells:
        keep = set(cells)
        out = [r for r in out if r["cell_id"] in keep]
    if paraphrases:
        keep = set(paraphrases)
        out = [r for r in out if r["paraphrase_id"] in keep]
    if pairs_per_cell:
        # the first k paraphrases available in each cell (some cells lost paraphrases in the audit)
        per_cell: Dict[str, List[str]] = {}
        for r in out:
            per_cell.setdefault(r["cell_id"], [])
            if r["paraphrase_id"] not in per_cell[r["cell_id"]]:
                per_cell[r["cell_id"]].append(r["paraphrase_id"])
        keep = {(c, p) for c, ps in per_cell.items() for p in sorted(ps)[:pairs_per_cell]}
        out = [r for r in out if (r["cell_id"], r["paraphrase_id"]) in keep]
    if sample is not None and sample < len(out):
        out = random.Random(seed).sample(out, sample)
        out.sort(key=lambda r: r["stimulus_id"])
    if limit is not None:
        out = out[:limit]
    return out
