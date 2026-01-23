#!/usr/bin/env python3
"""
Systematic optimization test harness.
Tests different optimization combinations and measures impact.
"""

import sys
import os
from perf_takehome import do_kernel_test

# Store original lookahead value
ORIGINAL_LOOKAHEAD = 128

def test_optimization(name, **kwargs):
    """Test a single optimization and return cycle count."""
    try:
        result = do_kernel_test(10, 16, 256, **kwargs)
        return result
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def test_lookahead(lookahead_value):
    """Test different lookahead values by temporarily modifying the code."""
    import perf_takehome
    # Save original
    original = perf_takehome.KernelBuilder.build
    original_lookahead = None
    
    # Find the lookahead value in the build method
    import inspect
    source = inspect.getsource(perf_takehome.KernelBuilder.build)
    
    # Create modified version
    modified_source = source.replace(
        f"lookahead={ORIGINAL_LOOKAHEAD}",
        f"lookahead={lookahead_value}"
    )
    
    # This is complex - let's just test what we can with flags
    # For lookahead, we'll need to manually test by editing the file
    return None

def main():
    print("="*70)
    print("SYSTEMATIC OPTIMIZATION TEST HARNESS")
    print("="*70)
    
    baseline = test_optimization("Baseline (current best)", 
                                 enable_debug=False,
                                 assume_zero_indices=True,
                                 max_special_level=-1,
                                 max_arith_level=-1,
                                 enable_prefetch=False)
    
    if baseline is None:
        print("Failed to get baseline!")
        return
    
    print(f"\n{'='*70}")
    print(f"Baseline: {baseline} cycles")
    print(f"Target: <1487 cycles (need to save {baseline - 1487} cycles)")
    print(f"{'='*70}\n")
    
    # Group optimizations by category
    optimizations = []
    
    # Category 1: Arith selection (previously tested, but retest with current bundler)
    optimizations.extend([
        ("Level 2 arith selection", {
            "max_arith_level": 2,
            "enable_prefetch": False,
        }),
        ("Level 2 arith + prefetch", {
            "max_arith_level": 2,
            "enable_prefetch": True,
        }),
        ("Level 3 arith selection", {
            "max_arith_level": 3,
            "enable_prefetch": False,
        }),
    ])
    
    # Category 2: Special levels (vselect tree - previously regressed)
    optimizations.extend([
        ("Level 2 special (vselect)", {
            "max_special_level": 2,
        }),
        ("Level 3 special (vselect)", {
            "max_special_level": 3,
        }),
    ])
    
    # Category 3: Combinations
    optimizations.extend([
        ("Level 2 arith + Level 2 special", {
            "max_arith_level": 2,
            "max_special_level": 2,
            "enable_prefetch": False,
        }),
    ])
    
    # Category 4: Edge cases
    optimizations.extend([
        ("No assume_zero_indices", {
            "assume_zero_indices": False,
        }),
    ])
    
    print("Testing optimizations...\n")
    results = []
    category_wins = {}
    category_regressions = {}
    
    for name, kwargs in optimizations:
        print(f"Testing: {name}")
        cycles = test_optimization(name, **kwargs)
        if cycles is not None:
            diff = cycles - baseline
            status = "✓ WIN" if diff < 0 else "✗ REGRESS" if diff > 0 else "= SAME"
            print(f"  Result: {cycles} cycles ({diff:+d}) {status}")
            results.append((name, cycles, diff))
            
            # Categorize
            category = "Unknown"
            if "arith" in name.lower():
                category = "Arith Selection"
            elif "special" in name.lower():
                category = "VSelect Tree"
            elif "assume" in name.lower():
                category = "Flags"
            else:
                category = "Combinations"
            
            if diff < 0:
                if category not in category_wins:
                    category_wins[category] = []
                category_wins[category].append((name, cycles, diff))
            elif diff > 0:
                if category not in category_regressions:
                    category_regressions[category] = []
                category_regressions[category].append((name, cycles, diff))
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
        print("BEST IMPROVEMENTS (sorted by cycles saved):")
        print(f"{'='*70}")
        wins.sort(key=lambda x: x[2])
        for i, (name, cycles, diff) in enumerate(wins[:10], 1):
            print(f"{i:2d}. {name:40s} {cycles:5d} cycles (saved {abs(diff):4d})")
    
    if category_wins:
        print(f"\n{'='*70}")
        print("WINS BY CATEGORY:")
        print(f"{'='*70}")
        for category, wins_list in category_wins.items():
            print(f"\n{category}:")
            wins_list.sort(key=lambda x: x[2])
            for name, cycles, diff in wins_list:
                print(f"  {name:40s} {cycles:5d} cycles (saved {abs(diff):4d})")
    
    if regressions:
        print(f"\n{'='*70}")
        print("REGRESSIONS (sorted by impact):")
        print(f"{'='*70}")
        regressions.sort(key=lambda x: x[2], reverse=True)
        for i, (name, cycles, diff) in enumerate(regressions[:5], 1):
            print(f"{i:2d}. {name:40s} {cycles:5d} cycles (+{diff:4d})")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    if wins:
        best = wins[0]
        print(f"✓ Best optimization found: {best[0]} ({best[1]} cycles, saved {abs(best[2])})")
        print(f"  Remaining to target: {best[1] - 1487} cycles")
    else:
        print("✗ No improvements found in tested optimizations.")
        print("  Consider:")
        print("    - Testing different lookahead values (256, 512, 1024)")
        print("    - Testing different block sizes (requires code modification)")
        print("    - Manual schedule optimization")
        print("    - Trace analysis for bottlenecks")
    print()

if __name__ == "__main__":
    main()
