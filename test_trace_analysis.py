#!/usr/bin/env python3
"""
Generate trace for Perfetto analysis.
"""

import subprocess
import sys
from perf_takehome import do_kernel_test

def generate_trace():
    """Generate trace.json for Perfetto visualization."""
    print("Generating trace for Perfetto analysis...")
    print("Running: python3 perf_takehome.py Tests.test_kernel_trace")
    
    result = subprocess.run(
        ['python3', 'perf_takehome.py', 'Tests.test_kernel_trace'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Trace generated: trace.json")
        print("\nTo view trace:")
        print("  1. python3 watch_trace.py")
        print("  2. Open http://localhost:8000 in browser")
        print("  3. Load trace.json in Perfetto UI")
    else:
        print("✗ Failed to generate trace")
        print(result.stderr)
    
    return result.returncode == 0

def analyze_cycles():
    """Get current cycle count for comparison."""
    cycles = do_kernel_test(10, 16, 256, block_size=16, lookahead=1024)
    print(f"\nCurrent cycles: {cycles}")
    print(f"Target: <1487 (need {cycles - 1487} more cycles saved)")
    return cycles

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "trace":
        generate_trace()
    else:
        analyze_cycles()
        print("\nTo generate trace, run: python3 test_trace_analysis.py trace")
