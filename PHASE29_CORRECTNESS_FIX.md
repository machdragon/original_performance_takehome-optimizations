# Phase-29: Level 4 Precompute Correctness Fix

## Problem
Level 4 precompute with vselect tree fails correctness for large batches (batch_size=256) due to bundler reordering conflicts when processing multiple block items with shared temp space.

## Root Cause
- All block items use the same `tree_temp_base` (v_tmp2_block) for intermediate vselect tree operations
- VLIW bundler can interleave instructions from different block items
- This causes one item's writes to overwrite another's intermediate temps before they're used
- Result: 252-255/256 incorrect output values

## Solution Implemented
Disable level4_precompute for block processing by adding condition in `vec_block_hash_only_slots`:
```python
if level4_precompute_round and level is not None and level in upper_levels and len(upper_levels) > 0 and len(block_vecs) <= VLEN:
    # Use vselect tree for level4 precompute (only for per-vector processing)
```

This ensures:
- ✅ **Correctness**: Always passes (falls back to normal gather loads for blocks)
- ✅ **Small batches**: Level4 precompute still works (when `len(block_vecs) <= VLEN`, i.e., per-vector processing)
- ⚠️ **Large batches**: Level4 precompute disabled for block processing (uses normal gather loads, but correctness maintained)

## Why This Works
- Per-vector processing (when `len(block_vecs) <= VLEN`) doesn't have the bundler reordering issue because each vector has isolated scratch space
- Block processing (when `len(block_vecs) > VLEN`) would require per-item isolated temps (1664 words) which exceeds scratch limit (1536 words)
- By disabling for blocks, we maintain correctness while preserving the optimization for cases where it works safely

## Ideal Solution (Not Feasible)
Per-item isolated tree temps:
- Need: 16 items * 13*VLEN = 1664 words
- Have: 1536 words (SCRATCH_SIZE)
- Shortage: 128 words

## Future Work
1. **Scratch optimization**: Free 128+ words to enable per-item temps
2. **Alternative approach**: Use different vselect tree implementation that needs less temp space
3. **Bundler control**: If bundler supports explicit barriers/dependencies, use those
