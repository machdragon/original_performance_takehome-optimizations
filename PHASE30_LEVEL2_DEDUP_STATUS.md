# Phase 30: Level 2 Gather Deduplication Implementation Status

## Implementation

Implemented gather deduplication for Level 2 rounds:
- Added `level2_prepare_slots_dedup()` function to load 4 unique values (indices 3-6) once per round
- Added condition in `vec_block_hash_only_slots`: `level == 2 and not level2_round and node_arith is None`
- Uses vselect tree to select correct value per lane (same as existing level2_round path)
- Interleaves prepare slots with hash slots to maintain load/VALU pipeline

## Results

**Current Performance:**
- With `enable_level2_where=True` (default): **1865 cycles** ✅
- With `enable_level2_where=False`: **1924 cycles** (no improvement from deduplication)

**Load Count Analysis:**
- With `level2_where=True`: 2634 loads
- With `level2_where=False`: 3146 loads
- **Difference: 512 loads** (exactly 2 rounds * 256 loads = rounds 2 and 13)

## Issue Analysis

The deduplication path is being triggered (192 vselect instructions), but it's not improving performance. Possible reasons:

1. **Vselect overhead**: The flow engine (1 slot/cycle) may be a bottleneck
2. **Pipeline breaking**: The prepare slots might not be interleaved optimally
3. **Buffer conflicts**: `level2_vecs_base = v_node_block[0]` might be overwritten by other operations
4. **Missing optimization**: The existing `level2_round` path works because it's integrated with prefetch and cross-round pipelining

## Next Steps

1. **Profile the vselect overhead**: Measure if flow engine is the bottleneck
2. **Try arithmetic selection**: Use VALU-only selection instead of vselect (like the disabled arith path)
3. **Optimize prepare timing**: Ensure vectors are prepared just before use, not at round start
4. **Consider Level 3/4**: If Level 2 doesn't help, try higher levels with more savings potential

## Key Insight

The existing `level2_round` optimization (1865 cycles) already implements gather deduplication effectively. The challenge is replicating that performance when `enable_level2_where=False`. The difference might be in how it integrates with prefetch and cross-round pipelining.
