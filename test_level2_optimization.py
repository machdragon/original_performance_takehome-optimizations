#!/usr/bin/env python3
"""
Test level 2 arithmetic selection optimization with different block sizes.
"""

import sys
from perf_takehome import do_kernel_test

def test_level2_with_blocksizes():
    """Test level 2 optimization with different block sizes to find scratch-efficient approach."""
    block_sizes = [12, 14, 15, 16]
    lookahead = 1024
    
    print("Testing level 2 optimization with different block sizes:")
    print("=" * 60)
    
    results = []
    for bs in block_sizes:
        try:
            print(f"\nblock_size={bs}:")
            cycles = do_kernel_test(10, 16, 256, block_size=bs, lookahead=lookahead)
            results.append((bs, cycles, "OK"))
            print(f"  {cycles} cycles")
        except AssertionError as e:
            if "Out of scratch space" in str(e):
                results.append((bs, None, "SCRATCH_OVERFLOW"))
                print(f"  SCRATCH_OVERFLOW")
            else:
                results.append((bs, None, "ERROR"))
                print(f"  ERROR: {e}")
        except Exception as e:
            results.append((bs, None, f"ERROR: {e}"))
            print(f"  ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    valid = [(bs, cyc) for bs, cyc, status in results if status == "OK" and cyc is not None]
    if valid:
        best = min(valid, key=lambda x: x[1])
        print(f"  Best: block_size={best[0]}, {best[1]} cycles")
    
    return results

if __name__ == "__main__":
    test_level2_with_blocksizes()
