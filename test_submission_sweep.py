#!/usr/bin/env python3
"""
Submission-range sweep harness.

Sweeps forest_height (8-10), rounds (8-20), batch_size (128/256)
with configurable kernel settings to highlight worst-case cycles.
"""

import argparse
import contextlib
import io
from statistics import mean

from perf_takehome import do_kernel_test


def parse_int_list(value):
    return [int(v) for v in value.split(",") if v]


def parse_rounds(value):
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return parse_int_list(value)


def run_sweep(
    heights,
    rounds,
    batches,
    *,
    block_size,
    lookahead,
    max_arith_level,
    enable_prefetch,
    assume_zero,
    verbose=True,
):
    results = []
    if verbose:
        print("Submission sweep settings:")
        print(f"  heights: {heights}")
        print(f"  rounds: {rounds[0]}..{rounds[-1]} ({len(rounds)} values)")
        print(f"  batches: {batches}")
        print("  kernel:")
        print(f"    block_size={block_size}, lookahead={lookahead}")
        print(f"    max_arith_level={max_arith_level}, enable_prefetch={enable_prefetch}")
        print(f"    assume_zero_indices={assume_zero}")
        print()

    for h in heights:
        for r in rounds:
            for b in batches:
                try:
                    if verbose:
                        cycles = do_kernel_test(
                            h,
                            r,
                            b,
                            block_size=block_size,
                            lookahead=lookahead,
                            max_arith_level=max_arith_level,
                            enable_prefetch=enable_prefetch,
                            assume_zero_indices=assume_zero,
                        )
                    else:
                        with contextlib.redirect_stdout(io.StringIO()):
                            with contextlib.redirect_stderr(io.StringIO()):
                                cycles = do_kernel_test(
                                    h,
                                    r,
                                    b,
                                    block_size=block_size,
                                    lookahead=lookahead,
                                    max_arith_level=max_arith_level,
                                    enable_prefetch=enable_prefetch,
                                    assume_zero_indices=assume_zero,
                                )
                except Exception as exc:
                    if verbose:
                        print(f"ERROR h={h} r={r} b={b}: {exc}")
                    cycles = None
                results.append((h, r, b, cycles))
                if verbose and cycles is not None:
                    print(f"h={h} r={r:2d} b={b:3d} -> {cycles}")

    valid = [row for row in results if row[3] is not None]
    summary = None
    if valid:
        cycles_list = [row[3] for row in valid]
        worst = max(valid, key=lambda row: row[3])
        best = min(valid, key=lambda row: row[3])
        summary = {
            "runs": len(valid),
            "total": len(results),
            "best": best,
            "worst": worst,
            "avg": mean(cycles_list),
        }

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Sweep submission parameter ranges")
    parser.add_argument("--heights", default="8,9,10", help="Comma-separated forest heights")
    parser.add_argument("--rounds", default="8-20", help="Comma-separated or range (e.g. 8-20)")
    parser.add_argument("--batches", default="128,256", help="Comma-separated batch sizes")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--lookahead", type=int, default=1024)
    parser.add_argument("--max-arith-level", type=int, default=-1)
    parser.add_argument("--enable-prefetch", action="store_true")
    parser.add_argument("--assume-zero", action="store_true", default=True)
    parser.add_argument("--no-assume-zero", dest="assume_zero", action="store_false")
    args = parser.parse_args()

    heights = parse_int_list(args.heights)
    rounds = parse_rounds(args.rounds)
    batches = parse_int_list(args.batches)

    results, summary = run_sweep(
        heights,
        rounds,
        batches,
        block_size=args.block_size,
        lookahead=args.lookahead,
        max_arith_level=args.max_arith_level,
        enable_prefetch=args.enable_prefetch,
        assume_zero=args.assume_zero,
        verbose=True,
    )

    if not summary:
        print("No valid results.")
        return

    print("\nSummary:")
    print(f"  runs: {summary['runs']} / {summary['total']}")
    best = summary["best"]
    worst = summary["worst"]
    print(f"  best:  h={best[0]} r={best[1]} b={best[2]} -> {best[3]}")
    print(f"  worst: h={worst[0]} r={worst[1]} b={worst[2]} -> {worst[3]}")
    print(f"  avg:   {summary['avg']:.1f}")

    print("\nTop 5 worst:")
    valid = [row for row in results if row[3] is not None]
    for h, r, b, c in sorted(valid, key=lambda row: row[3], reverse=True)[:5]:
        print(f"  h={h} r={r} b={b} -> {c}")


if __name__ == "__main__":
    main()
