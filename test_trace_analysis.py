#!/usr/bin/env python3
"""
Generate trace for Perfetto analysis and compute saturation diagnostics.
"""

import subprocess
import sys
import json
from perf_takehome import do_kernel_test, KernelBuilder
from problem import Machine, build_mem_image, Tree, Input
import random

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

def analyze_saturation():
    """Analyze load/VALU saturation from instruction stream."""
    print("Analyzing load/VALU saturation...")
    
    random.seed(123)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)
    
    kb = KernelBuilder(
        enable_debug=False,
        assume_zero_indices=True,
        lookahead=1024,
        block_size=16,
    )
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)
    
    # Analyze instruction bundles
    SLOT_LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}
    
    total_bundles = len(kb.instrs)
    load_slots_used = 0
    valu_slots_used = 0
    alu_slots_used = 0
    flow_slots_used = 0
    
    bundles_at_capacity = {"load": 0, "valu": 0, "alu": 0}
    bundles_empty = {"load": 0, "valu": 0, "alu": 0}
    bundles_partial = {"load": 0, "valu": 0, "alu": 0}
    
    load_valu_both_used = 0
    cycles_with_valu_idle = 0
    
    for bundle in kb.instrs:
        loads = len(bundle.get("load", []))
        valus = len(bundle.get("valu", []))
        alus = len(bundle.get("alu", []))
        flows = len(bundle.get("flow", []))
        
        load_slots_used += loads
        valu_slots_used += valus
        alu_slots_used += alus
        flow_slots_used += flows
        
        # Track capacity
        if loads == SLOT_LIMITS["load"]:
            bundles_at_capacity["load"] += 1
        elif loads == 0:
            bundles_empty["load"] += 1
        else:
            bundles_partial["load"] += 1
        
        if valus == SLOT_LIMITS["valu"]:
            bundles_at_capacity["valu"] += 1
        elif valus == 0:
            bundles_empty["valu"] += 1
            cycles_with_valu_idle += 1
        else:
            bundles_partial["valu"] += 1
        
        if alus == SLOT_LIMITS["alu"]:
            bundles_at_capacity["alu"] += 1
        elif alus == 0:
            bundles_empty["alu"] += 1
        else:
            bundles_partial["alu"] += 1
        
        # Track dual-engine usage
        if loads > 0 and valus > 0:
            load_valu_both_used += 1
    
    # Calculate statistics
    max_load_slots = total_bundles * SLOT_LIMITS["load"]
    max_valu_slots = total_bundles * SLOT_LIMITS["valu"]
    max_alu_slots = total_bundles * SLOT_LIMITS["alu"]
    
    load_util = (load_slots_used / max_load_slots * 100) if max_load_slots > 0 else 0
    valu_util = (valu_slots_used / max_valu_slots * 100) if max_valu_slots > 0 else 0
    alu_util = (alu_slots_used / max_alu_slots * 100) if max_alu_slots > 0 else 0
    
    print("\n" + "=" * 70)
    print("LOAD/VALU SATURATION DIAGNOSTICS")
    print("=" * 70)
    print(f"\nTotal bundles: {total_bundles}")
    print()
    
    print("Engine Utilization:")
    print(f"  Load:  {load_slots_used:5d} / {max_load_slots:5d} slots ({load_util:5.1f}%)")
    print(f"  VALU:  {valu_slots_used:5d} / {max_valu_slots:5d} slots ({valu_util:5.1f}%)")
    print(f"  ALU:   {alu_slots_used:5d} / {max_alu_slots:5d} slots ({alu_util:5.1f}%)")
    print(f"  Flow:  {flow_slots_used:5d} slots")
    print()
    
    print("Bundle Capacity Distribution:")
    print("  Load:")
    print(f"    At capacity ({SLOT_LIMITS['load']} slots): {bundles_at_capacity['load']:5d} ({bundles_at_capacity['load']/total_bundles*100:5.1f}%)")
    print(f"    Empty (0 slots):           {bundles_empty['load']:5d} ({bundles_empty['load']/total_bundles*100:5.1f}%)")
    print(f"    Partial:                   {bundles_partial['load']:5d} ({bundles_partial['load']/total_bundles*100:5.1f}%)")
    print("  VALU:")
    print(f"    At capacity ({SLOT_LIMITS['valu']} slots): {bundles_at_capacity['valu']:5d} ({bundles_at_capacity['valu']/total_bundles*100:5.1f}%)")
    print(f"    Empty (0 slots):           {bundles_empty['valu']:5d} ({bundles_empty['valu']/total_bundles*100:5.1f}%)")
    print(f"    Partial:                   {bundles_partial['valu']:5d} ({bundles_partial['valu']/total_bundles*100:5.1f}%)")
    print()
    
    print("Key Metrics:")
    print(f"  Cycles with VALU idle:        {cycles_with_valu_idle:5d} ({cycles_with_valu_idle/total_bundles*100:5.1f}%)")
    print(f"  Cycles with both load+VALU:   {load_valu_both_used:5d} ({load_valu_both_used/total_bundles*100:5.1f}%)")
    print()
    
    # Diagnose bottleneck
    print("Bottleneck Analysis:")
    if load_util > 80:
        print("  ⚠️  LOAD-BOUND: High load utilization suggests memory-bound workload")
    elif valu_util > 80:
        print("  ⚠️  VALU-BOUND: High VALU utilization suggests compute-bound workload")
    else:
        print("  ℹ️  MIXED: Neither engine fully saturated")
    
    if cycles_with_valu_idle > total_bundles * 0.3:
        print(f"  ⚠️  VALU IDLE: {cycles_with_valu_idle/total_bundles*100:.1f}% of cycles have no VALU ops")
        print("     → Opportunity: Insert VALU work (preloads, index prep) during idle cycles")
    
    if load_valu_both_used < total_bundles * 0.5:
        print(f"  ⚠️  LOW OVERLAP: Only {load_valu_both_used/total_bundles*100:.1f}% of cycles use both load+VALU")
        print("     → Opportunity: Better interleaving of loads and VALU ops")
    
    print()
    print("=" * 70)
    
    return {
        'total_bundles': total_bundles,
        'load_util': load_util,
        'valu_util': valu_util,
        'alu_util': alu_util,
        'cycles_with_valu_idle': cycles_with_valu_idle,
        'load_valu_both_used': load_valu_both_used,
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "trace":
            generate_trace()
        elif sys.argv[1] == "saturation":
            analyze_saturation()
        else:
            print("Usage: python3 test_trace_analysis.py [trace|saturation]")
    else:
        analyze_cycles()
        print("\nOptions:")
        print("  python3 test_trace_analysis.py trace      - Generate trace.json")
        print("  python3 test_trace_analysis.py saturation - Analyze load/VALU saturation")
