# Phase-30: Bundler Refinements

## Overview
Improved latency-aware scheduling and second-pass reordering in the VLIW bundler to better exploit instruction-level parallelism and reduce pipeline stalls.

## Changes Made

### 1. Improved Latency-Aware Scheduling
**Location**: `submission_perf_takehome.py:267-269, 325-330`

**Improvements**:
- **Previous**: Only tracked loads from current bundle
- **New**: Tracks loads from previous bundles using `recent_load_history` with `LOAD_LATENCY=1`
- **Rationale**: Loads take effect at end of cycle, so ALU/VALU ops in the NEXT cycle should avoid using them

**Implementation**:
- Maintains a history of load writes from previous bundles
- Prevents ALU/VALU ops from using values that were loaded in recent bundles
- Properly updates history on flush and barrier boundaries

### 2. Improved Second-Pass Reordering
**Location**: `submission_perf_takehome.py:410-490`

**Improvements**:
- **Previous**: Simple priority-based reordering (load > VALU > ALU) without dependency awareness
- **New**: Dependency-aware topological sort within bundles
- **Rationale**: Respects data dependencies while prioritizing critical engines (loads)

**Implementation**:
- Builds dependency graph for all slots in bundle
- Performs topological sort respecting dependencies
- Prioritizes by engine (load > VALU > ALU > store > flow) when multiple slots are schedulable
- Ensures all dependencies are satisfied before scheduling a slot

## Performance Impact

**Test Results** (submission test: 10, 16, 256):
- Baseline: 1865 cycles
- With latency-aware: 1865 cycles (no change)
- With second-pass: 1865 cycles (no change)
- With both: 1865 cycles (no change)

**Analysis**:
- Current bundler already does a good job scheduling
- Load/VALU overlap is already well-optimized (79.9% from previous analysis)
- Improvements may help with different workloads or future optimizations

## Future Work

1. **Multi-cycle latency modeling**: Currently assumes 1-cycle latency; could model longer latencies
2. **Load prefetch hints**: Use latency-aware info to schedule prefetches earlier
3. **VALU idle cycle filling**: Use second-pass to identify and fill idle VALU slots
4. **Adaptive scheduling**: Adjust priorities based on current bundle state

## Notes

- Both features remain disabled by default (`enable_latency_aware=False`, `enable_second_pass=False`)
- Code is correct and tested, but doesn't improve current benchmark performance
- Infrastructure improvements may benefit future optimizations
