"""
mintest.cli
===========

Command-line interface::

    mintest stats    --npz Q.npz
    mintest cover    --npz Q.npz --method greedy|ilp|maxsat [--r 1] [--timeout 300]
    mintest separate --npz Q.npz --method greedy|exact       [--r 1] [--timeout 1200]
"""

from __future__ import annotations

import argparse
import json
import sys

from .qmatrix import compute_stats, load_q_matrix, print_stats
from .cover import greedy_cover, ilp_cover, maxsat_cover
from .separation import exact_separation, greedy_separation


def _cmd_stats(args) -> int:
    Q = load_q_matrix(args.npz)
    stats = compute_stats(Q)
    print_stats(stats, label=args.npz)
    if args.json:
        print(json.dumps(stats, indent=2))
    return 0


def _cmd_cover(args) -> int:
    Q = load_q_matrix(args.npz)
    if args.method == "greedy":
        result = greedy_cover(Q, r=args.r)
    elif args.method == "ilp":
        result = ilp_cover(Q, r=args.r, timeout_sec=args.timeout, verbose=args.verbose)
    elif args.method == "maxsat":
        result = maxsat_cover(Q, r=args.r, timeout_sec=args.timeout)
    else:
        raise ValueError(args.method)

    elapsed = result.get("elapsed_sec")
    print(f"algorithm={result['algorithm']} r={args.r} "
          f"n_selected={result.get('n_selected')} "
          f"coverage_rate={result.get('coverage_rate')} "
          f"elapsed_sec={elapsed:.3f}" if elapsed is not None else str(result))
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "selected"}, indent=2))
    return 0


def _cmd_separate(args) -> int:
    Q = load_q_matrix(args.npz)
    if args.method == "greedy":
        result = greedy_separation(Q, r=args.r)
        size = result["n_selected"]
    elif args.method == "exact":
        result = exact_separation(Q, r=args.r, timeout_sec=args.timeout)
        size = result.get("optimal")
    else:
        raise ValueError(args.method)

    print(f"method={args.method} r={args.r} size={size} status={result.get('status', 'OK')}")
    if args.json:
        print(json.dumps(
            {k: v for k, v in result.items() if k != "selected_pattern_multiplicity"},
            indent=2,
        ))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mintest", description="Minimum test design over a Q-matrix")
    sub = p.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="Print Q-matrix statistics")
    p_stats.add_argument("--npz", required=True, help="Path to Q-matrix .npz file")
    p_stats.add_argument("--json", action="store_true", help="Also print stats as JSON")
    p_stats.set_defaults(func=_cmd_stats)

    p_cover = sub.add_parser("cover", help="Minimum r-cover test design")
    p_cover.add_argument("--npz", required=True)
    p_cover.add_argument("--method", choices=["greedy", "ilp", "maxsat"], default="greedy")
    p_cover.add_argument("--r", type=int, default=1)
    p_cover.add_argument("--timeout", type=int, default=300)
    p_cover.add_argument("--verbose", action="store_true")
    p_cover.add_argument("--json", action="store_true")
    p_cover.set_defaults(func=_cmd_cover)

    p_sep = sub.add_parser("separate", help="r-robust diagnostic separation test design")
    p_sep.add_argument("--npz", required=True)
    p_sep.add_argument("--method", choices=["greedy", "exact"], default="greedy")
    p_sep.add_argument("--r", type=int, default=1)
    p_sep.add_argument("--timeout", type=int, default=1200)
    p_sep.add_argument("--json", action="store_true")
    p_sep.set_defaults(func=_cmd_separate)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
