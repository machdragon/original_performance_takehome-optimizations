#!/usr/bin/env python3
"""
Comprehensive systematic optimization test suite.
Tests block sizes, lookahead combinations, VLIW techniques, and more.
"""

from perf_takehome import do_kernel_test

def test(name, **kwargs):
    """Test a single optimization and return cycle count."""
    try:
        result = do_kernel_test(10, 16, 256, **kwargs)
        return result
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    print("="*80)
    print("COMPREHENSIVE SYSTEMATIC OPTIMIZATION TEST SUITE")
    print("="*80)
    
    # Baseline
    baseline = test("Baseline (block_size=16, lookahead=1024)", 
                    block_size=16, lookahead=1024)
    
    if baseline is None:
        print("Failed to get baseline!")
        return
    
    print(f"\n{'='*80}")
    print(f"Baseline: {baseline} cycles")
    print(f"Target: <1487 cycles (need to save {baseline - 1487} cycles)")
    print(f"{'='*80}\n")
    
    all_results = []
    
    # Category 1: Block size variations
    print("="*80)
    print("CATEGORY 1: Block Size Variations")
    print("="*80)
    block_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 32]
    for bs in block_sizes:
        name = f"Block size={bs}"
        cycles = test(name, block_size=bs, lookahead=1024)
        if cycles:
            diff = cycles - baseline
            status = "✓" if diff < 0 else "✗" if diff > 0 else "="
            print(f"{status} {name:30s} {cycles:5d} ({diff:+5d})")
            all_results.append((name, cycles, diff))
    print()
    
    # Category 2: Lookahead variations with block_size=16
    print("="*80)
    print("CATEGORY 2: Lookahead Variations (with block_size=16)")
    print("="*80)
    lookaheads = [512, 1024, 2048, 4096, 8192, 16384]
    for la in lookaheads:
        name = f"Lookahead={la}"
        cycles = test(name, block_size=16, lookahead=la)
        if cycles:
            diff = cycles - baseline
            status = "✓" if diff < 0 else "✗" if diff > 0 else "="
            print(f"{status} {name:30s} {cycles:5d} ({diff:+5d})")
            all_results.append((name, cycles, diff))
    print()
    
    # Category 3: Combined block_size + lookahead
    print("="*80)
    print("CATEGORY 3: Combined Block Size + Lookahead")
    print("="*80)
    combinations = [
        (8, 2048), (10, 2048), (12, 2048), (14, 2048),
        (16, 2048), (16, 4096), (18, 2048), (20, 2048),
        (24, 2048), (32, 2048),
    ]
    for bs, la in combinations:
        name = f"Block={bs} + Lookahead={la}"
        cycles = test(name, block_size=bs, lookahead=la)
        if cycles:
            diff = cycles - baseline
            status = "✓" if diff < 0 else "✗" if diff > 0 else "="
            print(f"{status} {name:30s} {cycles:5d} ({diff:+5d})")
            all_results.append((name, cycles, diff))
    print()
    
    # Category 4: VLIW bundler techniques
    print("="*80)
    print("CATEGORY 4: VLIW Bundler Techniques")
    print("="*80)
    vliw_tests = [
        ("Second-pass reordering", {"enable_second_pass": True}),
        ("Latency-aware", {"enable_latency_aware": True}),
        ("Second-pass + Latency-aware", {"enable_second_pass": True, "enable_latency_aware": True}),
        ("Combining (disabled)", {"enable_combining": True}),
    ]
    for name, kwargs in vliw_tests:
        full_kwargs = {"block_size": 16, "lookahead": 1024, **kwargs}
        cycles = test(name, **full_kwargs)
        if cycles:
            diff = cycles - baseline
            status = "✓" if diff < 0 else "✗" if diff > 0 else "="
            print(f"{status} {name:30s} {cycles:5d} ({diff:+5d})")
            all_results.append((name, cycles, diff))
    print()
    
    # Category 5: Triple combinations
    print("="*80)
    print("CATEGORY 5: Triple Combinations (Best Block + Best Lookahead + VLIW)")
    print("="*80)
    # Find best block size and lookahead from previous tests
    best_block = 16  # Known good
    best_lookahead = 1024  # Known good
    triple_tests = [
        ("Best + Second-pass", {"enable_second_pass": True}),
        ("Best + Latency-aware", {"enable_latency_aware": True}),
        ("Best + Both VLIW", {"enable_second_pass": True, "enable_latency_aware": True}),
        ("Best + Lookahead=2048", {"lookahead": 2048}),
        ("Best + Lookahead=4096", {"lookahead": 4096}),
    ]
    for name, kwargs in triple_tests:
        full_kwargs = {"block_size": best_block, "lookahead": best_lookahead, **kwargs}
        cycles = test(name, **full_kwargs)
        if cycles:
            diff = cycles - baseline
            status = "✓" if diff < 0 else "✗" if diff > 0 else "="
            print(f"{status} {name:30s} {cycles:5d} ({diff:+5d})")
            all_results.append((name, cycles, diff))
    print()

    # Category 6: Fusion + where-tree
    print("="*80)
    print("CATEGORY 6: Fused Rounds and Where-Tree")
    print("="*80)
    fusion_tests = [
        ("Level2 where + Prefetch", {"enable_level2_where": True, "enable_prefetch": True}),
        ("Level3 VALU + Prefetch", {"enable_level3_valu": True, "enable_prefetch": True}),
        ("Level2 arith (max=2)", {"max_arith_level": 2}),
        ("Level2 arith + Prefetch", {"max_arith_level": 2, "enable_prefetch": True}),
        ("Two-round fusion", {"enable_two_round_fusion": True}),
        ("Two-round fusion + Prefetch", {"enable_two_round_fusion": True, "enable_prefetch": True}),
        ("Fusion + Level2 where", {"enable_two_round_fusion": True, "enable_level2_where": True, "enable_prefetch": True}),
    ]
    for name, kwargs in fusion_tests:
        full_kwargs = {"block_size": best_block, "lookahead": best_lookahead, **kwargs}
        cycles = test(name, **full_kwargs)
        if cycles:
            diff = cycles - baseline
            status = "✓" if diff < 0 else "✗" if diff > 0 else "="
            print(f"{status} {name:30s} {cycles:5d} ({diff:+5d})")
            all_results.append((name, cycles, diff))
    print()
    
    # Summary
    print("="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    wins = [r for r in all_results if r[2] < 0]
    regressions = [r for r in all_results if r[2] > 0]
    same = [r for r in all_results if r[2] == 0]
    
    print(f"\nTotal tests: {len(all_results)}")
    print(f"  Wins: {len(wins)}")
    print(f"  Regressions: {len(regressions)}")
    print(f"  Same: {len(same)}")
    
    if wins:
        print(f"\n{'='*80}")
        print("TOP 20 IMPROVEMENTS:")
        print(f"{'='*80}")
        wins.sort(key=lambda x: x[2])
        for i, (name, cycles, diff) in enumerate(wins[:20], 1):
            print(f"{i:2d}. {name:45s} {cycles:5d} cycles (saved {abs(diff):4d})")
        
        best = wins[0]
        print(f"\n{'='*80}")
        print(f"🏆 BEST OPTIMIZATION: {best[0]}")
        print(f"{'='*80}")
        print(f"  Cycles: {best[1]} (saved {abs(best[2])} from baseline {baseline})")
        print(f"  Remaining to target: {best[1] - 1487} cycles")
        if best[1] < 1487:
            print(f"  ✅ TARGET ACHIEVED! (<1487)")
        elif best[1] < 1548:
            print(f"  🎯 Very close! (<1548 threshold)")
        print(f"{'='*80}")
    
    if regressions:
        print(f"\n{'='*80}")
        print("WORST REGRESSIONS (top 5):")
        print(f"{'='*80}")
        regressions.sort(key=lambda x: x[2], reverse=True)
        for i, (name, cycles, diff) in enumerate(regressions[:5], 1):
            print(f"{i:2d}. {name:45s} {cycles:5d} cycles (+{diff:4d})")
    
    print()

if __name__ == "__main__":
    main()
