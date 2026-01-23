#!/usr/bin/env python3
"""
Comprehensive systematic optimization test harness.
Tests all configurable optimizations.
"""

import sys
from perf_takehome import do_kernel_test

def test_optimization(name, **kwargs):
    """Test a single optimization and return cycle count."""
    try:
        result = do_kernel_test(10, 16, 256, **kwargs)
        return result
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    print("="*70)
    print("COMPREHENSIVE SYSTEMATIC OPTIMIZATION TEST")
    print("="*70)
    
    # Baseline with current best settings
    baseline = test_optimization("Baseline (current best)", 
                                 enable_debug=False,
                                 assume_zero_indices=True,
                                 max_special_level=-1,
                                 max_arith_level=-1,
                                 enable_prefetch=False,
                                 lookahead=512,
                                 block_size=8,
                                 enable_second_pass=False,
                                 enable_latency_aware=False)
    
    if baseline is None:
        print("Failed to get baseline!")
        return
    
    print(f"\n{'='*70}")
    print(f"Baseline: {baseline} cycles")
    print(f"Target: <1487 cycles (need to save {baseline - 1487} cycles)")
    print(f"{'='*70}\n")
    
    optimizations = []
    
    # Category 1: Lookahead values
    print("Category 1: Testing lookahead values...")
    for lookahead in [256, 512, 1024, 2048]:
        optimizations.append((f"Lookahead={lookahead}", {"lookahead": lookahead}))
    
    # Category 2: Block sizes
    print("Category 2: Testing block sizes...")
    for block_size in [4, 6, 8, 10, 12, 16]:
        optimizations.append((f"Block size={block_size}", {"block_size": block_size}))
    
    # Category 3: Bundler optimizations
    print("Category 3: Testing bundler optimizations...")
    optimizations.extend([
        ("Second-pass reordering", {"enable_second_pass": True}),
        ("Latency-aware scheduling", {"enable_latency_aware": True}),
        ("Both bundler opts", {"enable_second_pass": True, "enable_latency_aware": True}),
    ])
    
    # Category 4: Combinations
    print("Category 4: Testing combinations...")
    optimizations.extend([
        ("Lookahead=1024 + Second-pass", {"lookahead": 1024, "enable_second_pass": True}),
        ("Lookahead=1024 + Latency-aware", {"lookahead": 1024, "enable_latency_aware": True}),
        ("Block=10 + Second-pass", {"block_size": 10, "enable_second_pass": True}),
        ("Block=12 + Latency-aware", {"block_size": 12, "enable_latency_aware": True}),
        ("All bundler opts", {"lookahead": 1024, "enable_second_pass": True, "enable_latency_aware": True}),
    ])
    
    print(f"\nTesting {len(optimizations)} optimizations...\n")
    results = []
    
    for name, kwargs in optimizations:
        print(f"Testing: {name}")
        cycles = test_optimization(name, **kwargs)
        if cycles is not None:
            diff = cycles - baseline
            status = "✓ WIN" if diff < 0 else "✗ REGRESS" if diff > 0 else "= SAME"
            print(f"  Result: {cycles} cycles ({diff:+d}) {status}")
            results.append((name, cycles, diff))
        print()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    wins = [r for r in results if r[2] < 0]
    regressions = [r for r in results if r[2] > 0]
    same = [r for r in results if r[2] == 0]
    
    print(f"\nTotal tests: {len(results)}")
    print(f"  Wins: {len(wins)}")
    print(f"  Regressions: {len(regressions)}")
    print(f"  Same: {len(same)}")
    
    if wins:
        print(f"\n{'='*70}")
        print("BEST IMPROVEMENTS (top 15):")
        print(f"{'='*70}")
        wins.sort(key=lambda x: x[2])
        for i, (name, cycles, diff) in enumerate(wins[:15], 1):
            print(f"{i:2d}. {name:45s} {cycles:5d} cycles (saved {abs(diff):4d})")
        
        best = wins[0]
        print(f"\n{'='*70}")
        print(f"BEST OPTIMIZATION: {best[0]}")
        print(f"  Cycles: {best[1]} (saved {abs(best[2])} from baseline)")
        print(f"  Remaining to target: {best[1] - 1487} cycles")
        if best[1] < 1487:
            print(f"  ✓ TARGET ACHIEVED!")
        print(f"{'='*70}")
    
    if regressions:
        print(f"\n{'='*70}")
        print("WORST REGRESSIONS (top 5):")
        print(f"{'='*70}")
        regressions.sort(key=lambda x: x[2], reverse=True)
        for i, (name, cycles, diff) in enumerate(regressions[:5], 1):
            print(f"{i:2d}. {name:45s} {cycles:5d} cycles (+{diff:4d})")
    
    print()

if __name__ == "__main__":
    main()
