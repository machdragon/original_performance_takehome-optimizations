#!/usr/bin/env python3
"""
Grid-search block_size/lookahead to minimize worst-case cycles
across submission parameter ranges.
"""

import argparse
import csv
from statistics import mean

from test_submission_sweep import parse_int_list, parse_rounds, run_sweep


def parse_ranges(value):
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return parse_int_list(value)


def main():
    parser = argparse.ArgumentParser(description="Grid-search block_size/lookahead")
    parser.add_argument("--heights", default="8,9,10", help="Comma-separated forest heights")
    parser.add_argument("--rounds", default="8-20", help="Comma-separated or range (e.g. 8-20)")
    parser.add_argument("--batches", default="128,256", help="Comma-separated batch sizes")
    parser.add_argument(
        "--block-sizes",
        default="4,6,8,10,12,14,16,18",
        help="Comma-separated block sizes or range (e.g. 4-18)",
    )
    parser.add_argument(
        "--lookaheads",
        default="512,1024,2048,4096",
        help="Comma-separated lookahead values or range (e.g. 512-4096)",
    )
    parser.add_argument("--max-arith-level", type=int, default=-1)
    parser.add_argument("--enable-prefetch", action="store_true")
    parser.add_argument("--assume-zero", action="store_true", default=True)
    parser.add_argument("--no-assume-zero", dest="assume_zero", action="store_false")
    parser.add_argument("--output", default="worstcase_grid.csv")
    args = parser.parse_args()

    heights = parse_int_list(args.heights)
    rounds = parse_rounds(args.rounds)
    batches = parse_int_list(args.batches)
    block_sizes = parse_ranges(args.block_sizes)
    lookaheads = parse_ranges(args.lookaheads)

    print("Worst-case grid search settings:")
    print(f"  heights: {heights}")
    print(f"  rounds: {rounds[0]}..{rounds[-1]} ({len(rounds)} values)")
    print(f"  batches: {batches}")
    print(f"  block_sizes: {block_sizes}")
    print(f"  lookaheads: {lookaheads}")
    print("  kernel:")
    print(f"    max_arith_level={args.max_arith_level}, enable_prefetch={args.enable_prefetch}")
    print(f"    assume_zero_indices={args.assume_zero}")
    print()

    results = []
    for block_size in block_sizes:
        for lookahead in lookaheads:
            _, summary = run_sweep(
                heights,
                rounds,
                batches,
                block_size=block_size,
                lookahead=lookahead,
                max_arith_level=args.max_arith_level,
                enable_prefetch=args.enable_prefetch,
                assume_zero=args.assume_zero,
                verbose=False,
            )
            if not summary:
                worst_cycles = None
                best_cycles = None
                avg_cycles = None
                runs = 0
                total = 0
            else:
                worst_cycles = summary["worst"][3]
                best_cycles = summary["best"][3]
                avg_cycles = summary["avg"]
                runs = summary["runs"]
                total = summary["total"]

            results.append(
                {
                    "block_size": block_size,
                    "lookahead": lookahead,
                    "worst_cycles": worst_cycles,
                    "best_cycles": best_cycles,
                    "avg_cycles": avg_cycles,
                    "runs": runs,
                    "total": total,
                }
            )
            status = "OK" if worst_cycles is not None else "ERROR"
            print(
                f"block_size={block_size:2d} lookahead={lookahead:5d} "
                f"worst={worst_cycles if worst_cycles is not None else 'N/A':>5} "
                f"avg={avg_cycles if avg_cycles is not None else 'N/A':>6} {status}"
            )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "block_size",
                "lookahead",
                "worst_cycles",
                "best_cycles",
                "avg_cycles",
                "runs",
                "total",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    valid = [row for row in results if row["worst_cycles"] is not None]
    if not valid:
        print("\nNo valid results.")
        return

    best = min(valid, key=lambda row: row["worst_cycles"])
    worst = max(valid, key=lambda row: row["worst_cycles"])
    print("\nSummary:")
    print(
        "  best worst-case: "
        f"block_size={best['block_size']} lookahead={best['lookahead']} "
        f"worst={best['worst_cycles']} avg={best['avg_cycles']:.1f}"
    )
    print(
        "  worst worst-case: "
        f"block_size={worst['block_size']} lookahead={worst['lookahead']} "
        f"worst={worst['worst_cycles']} avg={worst['avg_cycles']:.1f}"
    )

    top5 = sorted(valid, key=lambda row: row["worst_cycles"])[:5]
    print("\nTop 5 by worst-case:")
    for row in top5:
        print(
            "  block_size={block_size} lookahead={lookahead} "
            "worst={worst_cycles} avg={avg_cycles:.1f}".format(**row)
        )


if __name__ == "__main__":
    main()
