#!/usr/bin/env python3
"""
Cross-Round Performance Tracking

Tracks cycle count per round to identify high-cost rounds and bottlenecks.
"""

import sys
from problem import Machine, build_mem_image, Tree, Input
from perf_takehome import KernelBuilder, reference_kernel2
import random

def track_rounds(forest_height=10, rounds=16, batch_size=256, seed=123):
    """Track cycle count per round by inserting pause after each round."""
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    
    # Create kernel builder
    kb = KernelBuilder(
        enable_debug=False,
        assume_zero_indices=True,
        lookahead=1024,
        block_size=16,
    )
    
    # Modify build_kernel_general to insert pause after each round
    # We'll need to patch the machine to track cycles at each pause
    
    # Build kernel normally
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    
    # Create machine with round tracking
    round_cycles = []
    current_round = 0
    
    # We need to modify the program to track cycles at each pause
    # For now, let's run and manually track by counting pauses
    
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=1,
        trace=False,
    )
    
    # Track cycles before each pause
    pause_count = 0
    cycles_at_pause = []
    
    # Run until first pause (after setup)
    while machine.cores[0].state.name != "PAUSED" and machine.cores[0].state.name != "STOPPED":
        machine.run()
        if machine.cores[0].state.name == "PAUSED":
            cycles_at_pause.append(machine.cycle)
            pause_count += 1
            machine.cores[0].state = machine.cores[0].state.__class__(1)  # RUNNING
            if pause_count > rounds:
                break
    
    # Calculate per-round cycles
    if len(cycles_at_pause) >= 2:
        # First pause is after setup, subsequent are after each round
        setup_cycles = cycles_at_pause[0] if cycles_at_pause else 0
        round_deltas = []
        
        for i in range(1, min(len(cycles_at_pause), rounds + 1)):
            prev_cycles = cycles_at_pause[i-1] if i > 0 else setup_cycles
            curr_cycles = cycles_at_pause[i]
            round_deltas.append(curr_cycles - prev_cycles)
        
        print("Round-by-Round Cycle Breakdown:")
        print("-" * 50)
        print(f"Setup: {setup_cycles} cycles")
        print()
        
        total_round_cycles = 0
        for round_num, delta in enumerate(round_deltas):
            total_round_cycles += delta
            print(f"Round {round_num:2d}: {delta:4d} cycles")
        
        print("-" * 50)
        print(f"Total rounds: {total_round_cycles} cycles")
        print(f"Total (setup + rounds): {cycles_at_pause[-1] if cycles_at_pause else 0} cycles")
        
        # Identify outliers
        if round_deltas:
            avg = sum(round_deltas) / len(round_deltas)
            max_round = max(enumerate(round_deltas), key=lambda x: x[1])
            min_round = min(enumerate(round_deltas), key=lambda x: x[1])
            
            print()
            print("Outliers:")
            print(f"  Slowest: Round {max_round[0]} ({max_round[1]} cycles, +{max_round[1]-avg:.1f} vs avg)")
            print(f"  Fastest: Round {min_round[0]} ({min_round[1]} cycles, {min_round[1]-avg:.1f} vs avg)")
            print(f"  Average: {avg:.1f} cycles/round")
        
        return round_deltas
    else:
        print("Not enough pause points detected for round tracking")
        return []

def track_rounds_simple():
    """Simpler approach: run full kernel and estimate per-round from trace."""
    from perf_takehome import do_kernel_test
    
    # Run with trace enabled to get detailed breakdown
    print("Running with trace to analyze round-by-round performance...")
    print("(This is a simplified version - full tracking requires pause insertion)")
    print()
    
    # Just report total for now
    cycles = do_kernel_test(10, 16, 256, trace=False)
    print(f"Total cycles: {cycles}")
    print(f"Average per round: {cycles / 16:.1f} cycles")
    print()
    print("For detailed round-by-round tracking, use:")
    print("  python3 perf_takehome.py Tests.test_kernel_trace")
    print("  python3 watch_trace.py")
    print("  (Then analyze trace.json in Perfetto)")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        track_rounds_simple()
    else:
        track_rounds()
