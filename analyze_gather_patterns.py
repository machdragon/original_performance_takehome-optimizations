#!/usr/bin/env python3
"""
Analyze gather load patterns to identify deduplication opportunities.

This script simulates the tree traversal to count unique indices per round,
helping identify which rounds have the most duplicate loads.
"""

import random
from problem import Tree, Input, build_mem_image, myhash

def analyze_index_uniqueness(forest_height=10, batch_size=256, rounds=16):
    """Analyze unique indices per round by simulating traversal."""
    random.seed(42)  # Reproducible results
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    
    # Simulate traversal manually to track indices per round
    indices = inp.indices.copy()
    values = inp.values.copy()
    n_nodes = len(forest.values)
    
    round_stats = []
    
    for round_num in range(rounds):
        # Get current indices for this round (BEFORE processing)
        current_indices = indices.copy()
        
        # Count unique indices
        unique_indices = set(current_indices)
        num_unique = len(unique_indices)
        num_duplicates = batch_size - num_unique
        
        # Calculate potential savings
        # Each duplicate load could be eliminated
        # At 2 loads/cycle, eliminating N loads saves N/2 cycles
        potential_cycle_savings = num_duplicates / 2
        
        # Determine level (tree level based on index range)
        # Level 0: indices 0
        # Level 1: indices 1-2
        # Level 2: indices 3-6
        # Level 3: indices 7-14
        # Level 4: indices 15-30
        # etc.
        max_idx = max(current_indices) if current_indices else 0
        if max_idx == 0:
            level = 0
        elif max_idx <= 2:
            level = 1
        elif max_idx <= 6:
            level = 2
        elif max_idx <= 14:
            level = 3
        elif max_idx <= 30:
            level = 4
        elif max_idx <= 62:
            level = 5
        else:
            level = ">5"
        
        round_stats.append({
            'round': round_num,
            'level': level,
            'unique_indices': num_unique,
            'duplicate_loads': num_duplicates,
            'potential_savings': potential_cycle_savings,
            'max_idx': max_idx,
            'sample_indices': sorted(list(unique_indices))[:10],  # First 10 for display
        })
        
        # Advance to next round: simulate one round of traversal
        for i in range(batch_size):
            idx = indices[i]
            val = values[i]
            val = myhash(val ^ forest.values[idx])
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= n_nodes else idx
            values[i] = val
            indices[i] = idx
    
    return round_stats

def print_analysis(round_stats):
    """Print analysis results."""
    print("=" * 80)
    print("Gather Load Pattern Analysis")
    print("=" * 80)
    print(f"\n{'Round':<8} {'Level':<8} {'Unique':<10} {'Duplicates':<12} {'Max Idx':<10} {'Potential Savings':<18}")
    print("-" * 90)
    
    total_potential = 0
    high_potential_rounds = []
    
    for stat in round_stats:
        round_num = stat['round']
        level = stat['level']
        unique = stat['unique_indices']
        duplicates = stat['duplicate_loads']
        savings = stat['potential_savings']
        total_potential += savings
        
        max_idx = stat['max_idx']
        print(f"{round_num:<8} {str(level):<8} {unique:<10} {duplicates:<12} {max_idx:<10} {savings:<18.1f}")
        
        if savings > 50:  # High potential rounds
            high_potential_rounds.append((round_num, level, savings))
    
    print("-" * 80)
    print(f"{'TOTAL POTENTIAL SAVINGS':<50} {total_potential:.1f} cycles")
    print("=" * 80)
    
    print("\nHigh Potential Rounds for Deduplication (>50 cycle savings):")
    print("-" * 80)
    for round_num, level, savings in high_potential_rounds:
        print(f"  Round {round_num} (Level {level}): {savings:.1f} cycles")
    
    # Group by level
    print("\nSummary by Level:")
    print("-" * 80)
    level_totals = {}
    for stat in round_stats:
        level = stat['level']
        if level not in level_totals:
            level_totals[level] = {'rounds': 0, 'savings': 0}
        level_totals[level]['rounds'] += 1
        level_totals[level]['savings'] += stat['potential_savings']
    
    # Sort levels: numeric first, then strings
    sorted_levels = sorted([k for k in level_totals.keys() if isinstance(k, int)]) + \
                    sorted([k for k in level_totals.keys() if isinstance(k, str)])
    for level in sorted_levels:
        totals = level_totals[level]
        print(f"  Level {level}: {totals['rounds']} rounds, {totals['savings']:.1f} cycle savings potential")

if __name__ == "__main__":
    stats = analyze_index_uniqueness()
    print_analysis(stats)
