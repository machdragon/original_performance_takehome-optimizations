#!/usr/bin/env python3
"""
Test different lookahead values for the bundler.
Temporarily modifies perf_takehome.py to test different values.
"""

import subprocess
import re
import shutil

def test_lookahead(lookahead_value):
    """Test a specific lookahead value."""
    # This edits perf_takehome.py in-place; avoid concurrent runs.
    # Read the file
    with open('perf_takehome.py', 'r') as f:
        content = f.read()
    
    # Find and replace lookahead value
    # Look for: lookahead=128, or lookahead=XXX
    pattern = r'lookahead=\d+'
    new_content = re.sub(pattern, f'lookahead={lookahead_value}', content)
    
    # Write to temp file
    with open('perf_takehome.py.tmp', 'w') as f:
        f.write(new_content)
    
    # Backup original
    shutil.copy('perf_takehome.py', 'perf_takehome.py.bak')
    
    # Replace with modified
    shutil.copy('perf_takehome.py.tmp', 'perf_takehome.py')
    
    try:
        # Run test
        result = subprocess.run(
            ['python3', 'perf_takehome.py', 'Tests.test_kernel_cycles'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Extract cycle count
        match = re.search(r'CYCLES:\s+(\d+)', result.stdout)
        if match:
            cycles = int(match.group(1))
            return cycles
        else:
            print(f"  ERROR: Could not parse cycles from output")
            if result.stderr:
                print(f"  stderr: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Timeout")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    finally:
        # Restore original
        shutil.copy('perf_takehome.py.bak', 'perf_takehome.py')
        # Clean up
        import os
        if os.path.exists('perf_takehome.py.bak'):
            os.remove('perf_takehome.py.bak')
        if os.path.exists('perf_takehome.py.tmp'):
            os.remove('perf_takehome.py.tmp')

def main():
    print("="*70)
    print("TESTING DIFFERENT LOOKAHEAD VALUES")
    print("="*70)
    
    # Get baseline (current lookahead=128)
    print("\nGetting baseline (lookahead=128)...")
    baseline = test_lookahead(128)
    if baseline is None:
        print("Failed to get baseline!")
        return
    
    print(f"Baseline: {baseline} cycles (lookahead=128)")
    print(f"Target: <1487 cycles (need to save {baseline - 1487} cycles)\n")
    
    # Test different lookahead values
    lookahead_values = [16, 32, 64, 128, 256, 512, 1024, 2048]
    
    print("Testing lookahead values...\n")
    results = []
    
    for lookahead in lookahead_values:
        print(f"Testing lookahead={lookahead}...", end=' ', flush=True)
        cycles = test_lookahead(lookahead)
        if cycles is not None:
            diff = cycles - baseline
            status = "✓ WIN" if diff < 0 else "✗ REGRESS" if diff > 0 else "= SAME"
            print(f"{cycles} cycles ({diff:+d}) {status}")
            results.append((lookahead, cycles, diff))
        else:
            print("FAILED")
        print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    if results:
        results.sort(key=lambda x: x[1])  # Sort by cycles
        print("\nBest lookahead values (sorted by cycles):")
        for lookahead, cycles, diff in results[:5]:
            print(f"  lookahead={lookahead:4d}: {cycles:5d} cycles ({diff:+5d})")
        
        best = results[0]
        print(f"\nBest: lookahead={best[0]} with {best[1]} cycles")
        if best[1] < baseline:
            print(f"  Improvement: {baseline - best[1]} cycles saved")
        print(f"  Remaining to target: {best[1] - 1487} cycles")

if __name__ == "__main__":
    main()
