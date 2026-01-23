#!/usr/bin/env python3
"""
Test advanced VLIW bundling techniques.
"""

from perf_takehome import do_kernel_test

def test(name, **kwargs):
    try:
        return do_kernel_test(10, 16, 256, **kwargs)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

baseline = test("Baseline (block_size=16, lookahead=1024)", block_size=16, lookahead=1024)
print(f"\nBaseline: {baseline} cycles\n")

tests = [
    ("Combining", {"block_size": 16, "lookahead": 1024, "enable_combining": True}),
    ("Combining + Second-pass", {"block_size": 16, "lookahead": 1024, "enable_combining": True, "enable_second_pass": True}),
    ("Combining + Latency-aware", {"block_size": 16, "lookahead": 1024, "enable_combining": True, "enable_latency_aware": True}),
    ("All bundler opts", {"block_size": 16, "lookahead": 1024, "enable_combining": True, "enable_second_pass": True, "enable_latency_aware": True}),
    ("Lookahead=2048 + Combining", {"block_size": 16, "lookahead": 2048, "enable_combining": True}),
    ("Lookahead=4096 + Combining", {"block_size": 16, "lookahead": 4096, "enable_combining": True}),
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
else:
    print("\nNo improvements found")
