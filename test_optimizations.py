#!/usr/bin/env python3
"""
Systematic optimization test harness.
Tests different optimization combinations and measures impact.
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
    baseline = test_optimization("Baseline (current best)", 
                                 enable_debug=False,
                                 assume_zero_indices=True,
                                 max_special_level=-1,
                                 max_arith_level=-1,
                                 enable_prefetch=False)
    
    if baseline is None:
        print("Failed to get baseline!")
        return
    
    print(f"\n{'='*60}")
    print(f"Baseline: {baseline} cycles")
    print(f"Target: <1487 cycles (need to save {baseline - 1487} cycles)")
    print(f"{'='*60}\n")
    
    optimizations = [
        # Test level 2 arith (was +21, but might work with better integration)
        ("Level 2 arith selection", {
            "max_arith_level": 2,
            "enable_prefetch": False,
        }),
        
        # Test level 2 arith with prefetch
        ("Level 2 arith + prefetch", {
            "max_arith_level": 2,
            "enable_prefetch": True,
        }),
    ]
    
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
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    wins = [r for r in results if r[2] < 0]
    if wins:
        print("\nBest improvements:")
        wins.sort(key=lambda x: x[2])
        for name, cycles, diff in wins[:5]:
            print(f"  {name}: {cycles} cycles (saved {abs(diff)} cycles)")
    else:
        print("\nNo improvements found in tested optimizations.")
    print()

if __name__ == "__main__":
    main()
