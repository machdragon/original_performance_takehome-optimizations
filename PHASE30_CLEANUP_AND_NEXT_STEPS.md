# Phase 30: Cleanup and Next Steps

## Completed Cleanup

### ✅ Removed Level 3 Deduplication
- Removed `level3_dedup_round` condition and all references
- Removed `level3_prepare_slots_dedup()` function
- Removed `v_level_start_7`, `tmp4`, `tmp5`, `tmp6` allocations
- Removed `level3_base_addr_const` and `level3_addr_temp` (only used for dedup)
- **Result**: Build succeeds, baseline maintained at **1865 cycles**
- **Scratch space freed**: ~3-8 words (tmp4-6 + constants)

### ✅ Level 4 Status
- Level 4 is disabled by default (`enable_level4_where=False`, `enable_level4_valu=False`)
- Not blocking scratch space or performance
- Code remains but is not active
- **Decision**: Keep Level 4 code (not blocking, may be useful for future)

## Bundler Refinements Testing

### Results
- **Baseline**: 1865 cycles
- **With latency-aware**: 1865 cycles (no change)
- **With second-pass**: 1865 cycles (no change)
- **With both**: 1865 cycles (no change)

### Analysis
- Bundler refinements are implemented correctly
- Current bundler already does a good job scheduling
- Load/VALU overlap is already well-optimized (79.9% from previous analysis)
- **Conclusion**: Bundler refinements don't help for current benchmark

## Next Steps: Deeper Prefetching

### Current Prefetching
- `enable_prefetch=True` by default
- Prefetches block 0 during arith/level2/level3/level4 rounds
- Uses `v_node_prefetch` buffer (block_size * VLEN words)
- Cross-round pipelining enabled

### Opportunities for Deeper Prefetching

1. **Unroll Round Loop** (enable_unroll_8 exists but disabled)
   - Currently: `enable_unroll_8=False`
   - Unrolls exactly 8 rounds (eliminates loop overhead)
   - Enables better VLIW bundling across rounds
   - **Potential**: 10-30 cycles (loop overhead + better bundling)

2. **Double-Buffer Loads** (ping-pong pattern)
   - Issue loads for round N+1 while processing round N
   - Use two prefetch buffers (ping-pong)
   - **Potential**: 50-100 cycles if memory latency is bottleneck
   - **Constraint**: Requires 2x scratch space for prefetch buffers

3. **Lookahead Prefetching**
   - Current `lookahead=1024` in bundler (for instruction scheduling)
   - Could extend to prefetch multiple rounds ahead
   - **Potential**: 20-50 cycles
   - **Constraint**: Must balance against scratch space

4. **Aggressive Prefetch Scheduling**
   - Prefetch earlier in the pipeline
   - Overlap prefetch with more computation
   - **Potential**: 10-30 cycles

### Implementation Plan

1. **Test enable_unroll_8** (already implemented)
   ```python
   do_kernel_test(10, 16, 256, enable_unroll_8=True)
   ```
   - If rounds == 8, unrolls the round loop
   - For 16 rounds, would need `enable_unroll_16` or similar

2. **Implement Double-Buffer Prefetching**
   - Allocate `v_node_prefetch_A` and `v_node_prefetch_B`
   - Ping-pong between buffers
   - Prefetch round N+1 into buffer B while using buffer A for round N

3. **Extend Lookahead**
   - Increase prefetch distance (prefetch 2 rounds ahead)
   - Requires more scratch space but may hide more latency

## Current Performance

- **Baseline**: 1865 cycles
- **Target**: <1487 cycles
- **Gap**: 378 cycles needed

## Recommendations

1. **Test enable_unroll_8** for 8-round cases
2. **Implement double-buffer prefetching** (highest potential: 50-100 cycles)
3. **Consider arithmetic optimizations** (20-50 cycles potential)
4. **Profile to identify bottlenecks** (may reveal other opportunities)
