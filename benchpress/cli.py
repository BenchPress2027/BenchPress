"""Command line: python -m benchpress {info,run,score} ..."""
import argparse
import collections
import sys

from . import __version__, run, score
from .data import data_dir, load_track
from .providers import PROVIDERS


def info(args):
    d = data_dir(args.data_dir)
    core = load_track("core", args.data_dir)
    l2 = load_track("l2", args.data_dir)
    print(f"Bench Press {__version__} | data: {d}")
    print(f"core: {len(core):,} stimuli | {len({r['item_id'] for r in core}):,} minimal pairs | "
          f"{len({r['cell_id'] for r in core}):,} code x tension cells | {len({r['source_id'] for r in core})} codes")
    print("  per configuration:", dict(sorted(collections.Counter(f"C{r['configuration']}" for r in core).items())))
    main_l2 = [r for r in l2 if r["in_main_analysis"]]
    print(f"l2:   {len(l2):,} stimuli ({len(main_l2):,} in the main analysis) at C1 and C4")
    print("  main analysis by condition:", dict(collections.Counter(r["l2_condition"] for r in main_l2)))
    print("\nPresets for `run`:")
    for k, (desc, _) in run.PRESETS.items():
        print(f"  {k:9s} {desc}")
    print("\nProviders (--model PROVIDER:MODEL_ID) and their API-key variables:")
    for k, p in PROVIDERS.items():
        note = p.env_key or ("no key; needs --base-url (and --api-key-env if it has a key)" if k == "custom"
                             else f"no key; {p.base_url}")
        print(f"  {k:11s} {note}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="benchpress", description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--version", action="version", version=f"bench-press {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("info", help="describe the benchmark data, presets and providers")
    pi.add_argument("--data-dir", default=None)
    pi.set_defaults(func=info)

    pr = sub.add_parser("run", help="run a model on the benchmark",
                        description=run.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run.add_arguments(pr)
    pr.set_defaults(func=run.main)

    ps = sub.add_parser("score", help="score one or more runs and write a report",
                        description=score.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    score.add_arguments(ps)
    ps.set_defaults(func=score.main)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, ImportError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
