# Phase-29: Level 4 Precompute Correctness Issue

## Problem Summary

Level 4 precompute implementation works correctly for small batches (batch_size=32, no blocks) but fails catastrophically for large batches (batch_size=256, with blocks), producing 255-256/256 incorrect output values.

## Root Cause (Per Diagnosis)

The issue is **bundler reordering conflicts** when processing multiple block items:

1. **Shared tree_temp_base**: All block items use the same `tree_temp_base` (v_tmp2_block or v_node_block[1]) for intermediate vselect tree operations
2. **Bundler reordering**: The VLIW bundler can interleave instructions from different block items, causing one item's writes to overwrite another's intermediate temps before they're used
3. **Data corruption**: This corrupts the vselect tree computation, leading to incorrect node selection and wrong final results

## Diagnosis Details

Per the detailed diagnosis provided:

- **Aliasing conflict**: When `v_tmp2_block` and `v_tmp3_block` are aliased (to save scratch), writes from one block item can overwrite another's data
- **Shared temp space**: Using `tree_temp = level2_vecs_base + 4*VLEN` (or equivalent) for all block items means all items write to the same scratch region
- **Bundler independence**: The bundler sees instructions from different block items as independent and can reorder them, causing conflicts

## Proposed Fix (Per Diagnosis)

The proper fix is to use **per-item isolated tree temps**:

```python
# Per diagnosis: Use per-item offsets
tree_temp = level2_vecs_base + 4*VLEN + bi*2*VLEN  # For level2 (4 nodes)
# For level4 (16 nodes), would need:
tree_temp = v_node_block[0] + bi * 13*VLEN  # 13*VLEN = 104 words per item
```

However, this requires:
- **16 items * 13*VLEN = 1664 words** for tree temps
- **Current scratch limit: 1536 words**
- **Shortage: 128 words**

## Current Implementation (Compromise)

Current approach uses:
- **tree_temp_base**: Shared `v_node_block[1]` (sequential processing should be safe)
- **final_temp**: Per-item `v_node_block[0] + bi*VLEN` (isolated, prevents final conflicts)

This prevents final output conflicts but still has risk of intermediate temp conflicts if bundler reorders.

## Why It Works for Small Batches

Small batches (batch_size=32) don't use block processing - they use per-vector processing where each vector has its own isolated scratch space. This avoids the bundler reordering conflict entirely.

## Potential Solutions

### Option 1: Per-Item Tree Temps (Ideal, but exceeds scratch)
- Allocate 1664 words for per-item tree temps
- Would require freeing ~128 words from elsewhere
- **Status**: Not feasible with current scratch usage

### Option 2: Ensure Sequential Processing
- Add explicit dependencies or barriers to prevent bundler reordering
- Process items one at a time with explicit sequencing
- **Status**: Hard to implement at Python instruction level

### Option 3: Disable Bundling for VSelect Tree Section
- If bundler supports disabling for specific sections, use that
- **Status**: Unknown if bundler supports this

### Option 4: Process Items in Smaller Groups
- Process 1-2 items at a time, ensuring completion before next group
- **Status**: Would require restructuring the loop

### Option 5: Use Different Buffer Strategy
- Use v_node_block[0] and v_node_block[1] more cleverly
- Alternate or use ping-pong for tree temps
- **Status**: Still limited by buffer sizes

## Current Status

- ✅ **Build**: Successful (scratch space sufficient)
- ✅ **Small batches**: Correctness passes (no blocks)
- ❌ **Large batches**: Correctness fails (block processing, bundler conflicts)
- ⏳ **Performance**: Not measured (blocked by correctness)

## Next Steps

1. **Investigate bundler behavior**: Check if bundler actually reorders instructions from different block items
2. **Try explicit dependencies**: Add fake dependencies to force sequential processing
3. **Consider alternative approaches**: Process items differently or use different scratch strategy
4. **Measure performance once fixed**: Once correctness is resolved, measure cycle improvement

## Notes

- The vselect tree logic itself is correct (works for small batches)
- The issue is specifically in block processing with bundler reordering
- Per-item tree temps would solve it but exceed scratch limits
- Current compromise may work if bundler respects instruction order, but testing shows it doesn't
