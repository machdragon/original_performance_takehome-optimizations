#!/usr/bin/env python3
"""
Test combinations of optimizations with block_size=16 baseline.
"""

from perf_takehome import do_kernel_test

def test(name, **kwargs):
    try:
        return do_kernel_test(10, 16, 256, **kwargs)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

baseline = test("Baseline (block_size=16)", block_size=16)
print(f"\nBaseline: {baseline} cycles\n")

# Test combinations with block_size=16
tests = [
    ("Block=16 + Lookahead=1024", {"block_size": 16, "lookahead": 1024}),
    ("Block=16 + Second-pass", {"block_size": 16, "enable_second_pass": True}),
    ("Block=16 + Latency-aware", {"block_size": 16, "enable_latency_aware": True}),
    ("Block=16 + All bundler", {"block_size": 16, "lookahead": 1024, "enable_second_pass": True, "enable_latency_aware": True}),
]

wins = []
for name, kwargs in tests:
    cycles = test(name, **kwargs)
    if cycles:
        diff = cycles - baseline
        status = "✓" if diff < 0 else "✗" if diff > 0 else "="
        print(f"{status} {name:40s} {cycles:5d} ({diff:+4d})")
        if diff < 0:
            wins.append((name, cycles, diff))

if wins:
    print(f"\nBest: {wins[0][0]} - {wins[0][1]} cycles (saved {abs(wins[0][2])})")
    print(f"Remaining to <1487: {wins[0][1] - 1487} cycles")
