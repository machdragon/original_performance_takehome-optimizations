#!/usr/bin/env python3
"""
Multi-Parameter Regression Heatmap Test

Generates a heatmap matrix over (block_size, lookahead) combinations
to visualize parameter cliffs and interaction effects.
"""

import argparse
import csv
from perf_takehome import do_kernel_test

def parse_int_list(value):
    return [int(v) for v in value.split(",") if v]


def test_heatmap(block_sizes, lookahead_values, output_path):
    """Generate heatmap data for block_size x lookahead combinations."""
    results = []
    baseline = None
    
    print("Generating heatmap data...")
    print("Format: block_size, lookahead, cycles, status")
    print("-" * 60)
    
    for block_size in block_sizes:
        for lookahead in lookahead_values:
            try:
                # Run test and capture cycles
                cycles = do_kernel_test(
                    10, 16, 256,
                    block_size=block_size,
                    lookahead=lookahead,
                    enable_debug=False,
                    assume_zero_indices=True,
                )
                
                if baseline is None:
                    # Baseline is the first successful run in iteration order.
                    baseline = cycles
                
                status = "OK"
                results.append({
                    'block_size': block_size,
                    'lookahead': lookahead,
                    'cycles': cycles,
                    'status': status,
                    'vs_baseline': cycles - baseline,
                })
                
                print(f"{block_size:3d}, {lookahead:5d}, {cycles:5d}, {status}")
                
            except AssertionError as e:
                if "Out of scratch space" in str(e):
                    status = "SCRATCH_OVERFLOW"
                    results.append({
                        'block_size': block_size,
                        'lookahead': lookahead,
                        'cycles': None,
                        'status': status,
                        'vs_baseline': None,
                    })
                    print(f"{block_size:3d}, {lookahead:5d}, {'N/A':>5s}, {status}")
                else:
                    status = "ERROR"
                    results.append({
                        'block_size': block_size,
                        'lookahead': lookahead,
                        'cycles': None,
                        'status': status,
                        'vs_baseline': None,
                    })
                    print(f"{block_size:3d}, {lookahead:5d}, {'N/A':>5s}, {status}")
            except Exception as e:
                status = "ERROR"
                results.append({
                    'block_size': block_size,
                    'lookahead': lookahead,
                    'cycles': None,
                    'status': status,
                    'vs_baseline': None,
                })
                print(f"{block_size:3d}, {lookahead:5d}, {'N/A':>5s}, {status}: {e}")
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['block_size', 'lookahead', 'cycles', 'status', 'vs_baseline'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_path}")
    
    # Generate summary statistics
    valid_results = [r for r in results if r['cycles'] is not None]
    if valid_results:
        best = min(valid_results, key=lambda x: x['cycles'])
        worst = max(valid_results, key=lambda x: x['cycles'])
        avg = sum(r['cycles'] for r in valid_results) / len(valid_results)
        
        print("\nSummary:")
        print(f"  Best:  block_size={best['block_size']}, lookahead={best['lookahead']}, cycles={best['cycles']}")
        print(f"  Worst: block_size={worst['block_size']}, lookahead={worst['lookahead']}, cycles={worst['cycles']}")
        print(f"  Average: {avg:.1f} cycles")
        print(f"  Range: {worst['cycles'] - best['cycles']} cycles")
    
    # Identify parameter cliffs
    print("\nParameter Cliffs (large jumps):")
    for block_size in block_sizes:
        cycles_by_lookahead = {}
        for r in results:
            if r['block_size'] == block_size and r['cycles'] is not None:
                cycles_by_lookahead[r['lookahead']] = r['cycles']
        
        if len(cycles_by_lookahead) > 1:
            sorted_lookaheads = sorted(cycles_by_lookahead.keys())
            for i in range(len(sorted_lookaheads) - 1):
                la1, la2 = sorted_lookaheads[i], sorted_lookaheads[i+1]
                diff = cycles_by_lookahead[la2] - cycles_by_lookahead[la1]
                if abs(diff) > 10:  # Significant change
                    print(f"  block_size={block_size}: {la1}→{la2} lookahead: {diff:+d} cycles")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmap for block_size/lookahead combos")
    parser.add_argument("--block-sizes", default="4,6,8,10,12,14,16,18")
    parser.add_argument("--lookaheads", default="512,1024,2048,4096")
    parser.add_argument("--output", default="heatmap_results.csv")
    args = parser.parse_args()

    block_sizes = parse_int_list(args.block_sizes)
    lookahead_values = parse_int_list(args.lookaheads)
    test_heatmap(block_sizes, lookahead_values, args.output)
