"""Run a model on the benchmark and write one JSON line per answer.

Runs are resumable: answers already in results.jsonl (without an error) are
skipped, so the same command can be re-run after an interruption or to retry
failed calls.
"""
import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .data import load_track, select
from .parsing import parse_response
from .prompts import ARMS, DEFAULT_MAX_TOKENS, build_prompt
from .providers import REASONING_MIN_MAX_TOKENS, Client, is_reasoning_model, is_retryable, parse_model

PRESETS = {
    # name: (description, settings)
    "smoke":    ("40 random core stimuli, letter arm: checks your setup in a minute",
                 {"track": "core", "sample": 40, "arms": ["letter"], "replicates": 1}),
    "lite":     ("one minimal pair per cell, all configurations and orders, letter arm (9,190 calls)",
                 {"track": "core", "pairs_per_cell": 1, "arms": ["letter"], "replicates": 1}),
    "standard": ("the full core benchmark, letter arm, one replicate (36,590 calls)",
                 {"track": "core", "arms": ["letter"], "replicates": 1}),
    "paper":    ("the paper's protocol: full core, both arms, two replicates (146,360 calls)",
                 {"track": "core", "arms": ["letter", "rationale"], "replicates": 2}),
    "l2":       ("the exception (L2) track as in the paper: main-analysis items, letter arm (19,296 calls)",
                 {"track": "l2", "arms": ["letter"], "replicates": 1}),
}
CONFIRM_ABOVE = 2000          # ask before starting larger runs (interactive terminals only)
ABORT_AFTER_ERRORS = 25       # consecutive failed calls: stop, the key or model id is probably wrong


def add_arguments(p: argparse.ArgumentParser):
    p.add_argument("--model", "-m", required=True,
                   help="PROVIDER:MODEL_ID, e.g. openai:gpt-4o-mini, anthropic:claude-sonnet-4-5, "
                        "openrouter:openai/gpt-oss-20b, ollama:llama3.1:8b")
    p.add_argument("--preset", choices=PRESETS, default=None,
                   help="; ".join(f"{k}: {v[0]}" for k, v in PRESETS.items()) + ". Explicit options override it.")
    p.add_argument("--track", choices=["core", "l2"], default=None, help="core (default) or l2 (exceptions)")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=None,
                   help="letter (answer with A or B only; default) and/or rationale (letter, then 3-5 sentences)")
    p.add_argument("--replicates", type=int, default=None, help="times each stimulus is asked (default 1)")
    p.add_argument("--label", default=None, help="model name used in results and reports (default: the model id)")
    p.add_argument("--out", default=None, help="run folder (default runs/<label>)")

    s = p.add_argument_group("subsets")
    s.add_argument("--configurations", nargs="+", type=int, choices=range(5), default=None, metavar="N",
                   help="framing configurations to run, 0-4 (default all)")
    s.add_argument("--sources", nargs="+", default=None, help="source_id values (see data/sources.csv)")
    s.add_argument("--cells", nargs="+", default=None, help="cell_id values (see data/cells.csv)")
    s.add_argument("--paraphrases", nargs="+", default=None, help="e.g. P1 P2")
    s.add_argument("--pairs-per-cell", type=int, default=None, help="keep the first K minimal pairs of each cell")
    s.add_argument("--sample", type=int, default=None, help="random sample of N stimuli (smoke tests)")
    s.add_argument("--l2-all", action="store_true",
                   help="l2 track: also run items outside the paper's main analysis (phrase-fit filter)")
    s.add_argument("--limit", type=int, default=None, help="first N stimuli only")
    s.add_argument("--seed", type=int, default=0)

    c = p.add_argument_group("calls")
    c.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    c.add_argument("--max-tokens", type=int, default=None,
                   help=f"output budget per call (default: letter {DEFAULT_MAX_TOKENS['letter']}, rationale "
                        f"{DEFAULT_MAX_TOKENS['rationale']}; at least {REASONING_MIN_MAX_TOKENS} for reasoning models)")
    c.add_argument("--reasoning", action="store_true",
                   help=f"treat the model as a reasoning model (raises the budget to {REASONING_MIN_MAX_TOKENS})")
    c.add_argument("--temperature", type=float, default=None,
                   help="sampling temperature (default: not sent, i.e. the provider default, as in the paper)")
    c.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (custom provider, or to override one)")
    c.add_argument("--api-key-env", default=None, help="environment variable holding the API key (override)")
    c.add_argument("--timeout", type=float, default=300.0, help="seconds per request (default 300)")
    c.add_argument("--max-retries", type=int, default=6, help="retries for rate limits and server errors")
    c.add_argument("--data-dir", default=None, help="benchmark data folder (default: the repository's data/)")
    c.add_argument("--dry-run", action="store_true", help="show the plan and an example prompt; no API calls")
    c.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation before large runs")


def load_dotenv(path=".env"):
    """Minimal .env reader (KEY=VALUE lines); existing environment variables win."""
    import os
    for candidate in (Path(path), Path(__file__).resolve().parent.parent / ".env"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
            return candidate
    return None


def resolve(args):
    preset = PRESETS[args.preset][1] if args.preset else {}
    track = args.track or preset.get("track", "core")
    arms = args.arms or preset.get("arms", ["letter"])
    replicates = args.replicates or preset.get("replicates", 1)
    sample = args.sample if args.sample is not None else preset.get("sample")
    ppc = args.pairs_per_cell if args.pairs_per_cell is not None else preset.get("pairs_per_cell")
    return track, arms, replicates, sample, ppc


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha8(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def load_done(results_path: Path, model_id: str):
    done, models = set(), set()
    if not results_path.exists():
        return done
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue            # a line cut by a crash: that call is simply redone
            models.add(r.get("model_id"))
            if not r.get("error"):
                done.add((r["stimulus_id"], r["arm"], r["replicate"]))
    if models - {model_id}:
        raise SystemExit(f"{results_path} already holds answers from {sorted(models - {model_id})}. "
                         f"Use another --out (or --label) for a different model.")
    return done


def fmt_duration(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def main(args):
    provider, model_id = parse_model(args.model)
    track, arms, replicates, sample, ppc = resolve(args)
    label = args.label or model_id
    out = Path(args.out or Path("runs") / re.sub(r"[^\w.\-]+", "_", label))
    results_path = out / "results.jsonl"

    stimuli = select(load_track(track, args.data_dir), configurations=args.configurations, sources=args.sources,
                     cells=args.cells, paraphrases=args.paraphrases, pairs_per_cell=ppc, sample=sample,
                     l2_all=args.l2_all, seed=args.seed, limit=args.limit)
    if not stimuli:
        raise SystemExit("No stimuli match these options.")

    reasoning = args.reasoning or is_reasoning_model(model_id)
    budgets = {}
    for arm in arms:
        b = args.max_tokens or DEFAULT_MAX_TOKENS[arm]
        budgets[arm] = max(b, REASONING_MIN_MAX_TOKENS) if reasoning and not args.max_tokens else b

    tasks = [(s, arm, rep) for rep in range(1, replicates + 1) for arm in arms for s in stimuli]
    done = set() if args.dry_run else load_done(results_path, model_id)
    todo = [t for t in tasks if (t[0]["stimulus_id"], t[1], t[2]) not in done]

    in_chars = sum(len(build_prompt(s["prompt"], arm)) for s, arm, _ in todo)
    print(f"Bench Press {__version__} | model {provider}:{model_id} (label '{label}')")
    print(f"track {track} | arms {', '.join(arms)} | replicates {replicates} | {len(stimuli):,} stimuli "
          f"-> {len(tasks):,} calls, {len(tasks) - len(todo):,} already done, {len(todo):,} to run")
    print(f"approx. input tokens to send: {in_chars / 4 / 1e6:.2f}M | output budget per call: "
          + ", ".join(f"{a} {b}" for a, b in budgets.items()) + (" (reasoning model)" if reasoning else ""))
    print(f"results -> {results_path}")

    if args.dry_run:
        s = stimuli[0]
        for arm in arms:
            print(f"\n--- example prompt ({s['stimulus_id']}, arm {arm}, gold {s['gold_letter']}) ---")
            print(build_prompt(s["prompt"], arm))
        return
    if not todo:
        print("Nothing to do: every call already has an answer. Score it with: "
              f"python -m benchpress score {out}")
        return
    if len(todo) > CONFIRM_ABOVE and not args.yes and sys.stdin.isatty():
        if input(f"Send {len(todo):,} requests to {provider}? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("Cancelled.")

    load_dotenv()
    client = Client(args.model, base_url=args.base_url, api_key_env=args.api_key_env, timeout=args.timeout)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "invocations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp_utc": utc_now(), "benchpress_version": __version__, "argv": sys.argv[1:],
                            "provider": provider, "model_id": model_id, "label": label, "track": track,
                            "arms": arms, "replicates": replicates, "max_tokens": budgets,
                            "temperature": args.temperature, "base_url": client.base_url,
                            "calls_planned": len(todo)}) + "\n")

    stop = threading.Event()

    def work(task):
        s, arm, rep = task
        prompt = build_prompt(s["prompt"], arm)
        row = {"stimulus_id": s["stimulus_id"], "track": track, "arm": arm, "replicate": rep, "model": label,
               "provider": provider, "model_id": model_id, "max_tokens": budgets[arm],
               "temperature": args.temperature, "prompt_sha256_8": sha8(prompt)}
        err = None
        for attempt in range(args.max_retries + 1):
            if stop.is_set():
                return None
            try:
                res = client.call(prompt, budgets[arm], args.temperature)
                break
            except Exception as e:  # noqa: BLE001
                err = e
                if not is_retryable(e) or attempt == args.max_retries:
                    res = None
                    break
                time.sleep(min(60.0, 2.0 * 2 ** attempt) * (0.5 + random.random()))
        if res is None:
            row.update({"error": f"{type(err).__name__}: {str(err)[:500]}", "timestamp_utc": utc_now()})
            return row
        choice, rationale = parse_response(res["response"])
        row.update({"model_version": res["model_version"], "raw_response": res["response"],
                    "chosen_option": choice, "rationale": rationale if arm == "rationale" else None,
                    "is_correct": None if choice is None else choice == s["gold_letter"],
                    "finish_reason": res["finish_reason"], "input_tokens": res["input_tokens"],
                    "output_tokens": res["output_tokens"], "latency_ms": res["latency_ms"],
                    "reasoning": res.get("reasoning"), "error": None, "timestamp_utc": utc_now()})
        return row

    n = errors = consecutive = unparsed = correct = truncated = 0
    t0, last_print = time.time(), 0.0
    with results_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, t) for t in todo]
        try:
            for fut in as_completed(futures):
                row = fut.result()
                if row is None:
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if row["error"]:
                    errors += 1
                    consecutive += 1
                    if consecutive == 1 or errors <= 3:
                        print(f"  error on {row['stimulus_id']}: {row['error'][:300]}")
                    if consecutive >= ABORT_AFTER_ERRORS:
                        stop.set()
                        for fu in futures:
                            fu.cancel()
                        raise SystemExit(f"Stopped after {consecutive} consecutive errors (last: {row['error'][:300]}). "
                                         "Check the API key, the model id and your quota, then re-run the same "
                                         "command to resume.")
                    continue
                consecutive = 0
                unparsed += row["chosen_option"] is None
                correct += bool(row["is_correct"])
                truncated += row["finish_reason"] in ("length", "max_tokens") and row["chosen_option"] is None
                now = time.time()
                if now - last_print > 10 or n == len(todo):
                    last_print = now
                    rate = n / max(now - t0, 1e-9)
                    parsed = n - errors - unparsed
                    print(f"[{n:,}/{len(todo):,}] {100 * n / len(todo):5.1f}% | {rate:.1f} calls/s | ETA "
                          f"{fmt_duration((len(todo) - n) / max(rate, 1e-9))} | code-aligned so far "
                          f"{100 * correct / max(parsed, 1):.1f}% | unparsed {unparsed} | errors {errors}", flush=True)
        except KeyboardInterrupt:
            stop.set()
            for fu in futures:
                fu.cancel()
            print("\nInterrupted. Answers so far are saved; re-run the same command to resume.")
            raise SystemExit(130)

    print(f"\nDone: {n - errors:,} answers, {errors:,} errors, {unparsed:,} unparsed in {fmt_duration(time.time() - t0)}.")
    if truncated:
        print(f"WARNING: {truncated} answers hit the output limit before giving a letter. If the model reasons "
              f"before answering, re-run with --reasoning or a larger --max-tokens.")
    if errors:
        print("Some calls failed. Re-run the same command to retry them.")
    print(f"Score with: python -m benchpress score {out}")
