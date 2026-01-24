# Phase-32: Scratch Optimization Status

## Completed

### Scratch Space Optimizations Implemented
1. **Aggressive Temporary Block Aliasing**: 
   - When `enable_level2_where=False` and `enable_level3_where=False`, alias `v_tmp2_block` and `v_tmp3_block`
   - **Savings**: 128 words (block_size * VLEN = 16 * 8 = 128)

2. **Enhanced Scratch Allocation Error Messages**:
   - Added detailed error messages showing scratch usage and what allocation failed
   - Helps debug scratch space issues

3. **VSelect Tree Reuse Method**:
   - Created `build_vselect_tree_reuse()` that uses pre-allocated temp space instead of allocating during execution
   - Prevents scratch allocation failures in vselect tree operations

## Current Status

### Scratch Usage
- **Before optimizations**: Would exceed 1536 words (build failed)
- **After optimizations**: 1464/1536 words (95% usage, 72 words headroom)
- **Level 4 precompute enabled**: Builds successfully

### Level 4 Precompute Status
- **Build**: ✅ Successful (scratch space sufficient)
- **Correctness**: ❌ Failing (logic issue in vselect tree or relative index calculation)
- **Performance**: Not yet measured (blocked by correctness)

## Issues to Resolve

### 1. Correctness Failure
**Problem**: Level 4 precompute produces incorrect output values
**Symptoms**: All output values mismatch reference
**Possible Causes**:
- Relative index calculation incorrect
- VSelect tree selecting wrong nodes
- Level base address calculation wrong
- Temp space reuse causing data corruption

**Next Steps**:
- Add debug logging to trace vselect tree operations
- Compare intermediate values with reference
- Verify level 4 precompute data is loaded correctly
- Check if relative index calculation matches expected behavior

### 2. Temp Space Reuse in VSelect Tree
**Problem**: Processing multiple vectors in a loop reuses same temp space
**Risk**: If VLIW scheduling overlaps operations, temp space could be corrupted
**Solution**: Either process vectors sequentially (no overlap) or allocate separate temp space per vector

## Remaining Phase-32 Tasks

### High Priority (Required for Level 4 Precompute)
- [ ] Fix correctness issue in level 4 precompute logic
- [ ] Verify vselect tree selects correct nodes
- [ ] Test with enable_debug=True to trace intermediate values
- [ ] Measure performance improvement once correctness is fixed

### Medium Priority (Additional Optimizations)
- [ ] Implement buffer pooling for where-tree operations
- [ ] Eliminate overshoot buffers (if they exist)
- [ ] Optimize block buffer usage further
- [ ] Consider deeper prefetch if space allows

### Low Priority (Future Work)
- [ ] Triple-buffering for block prefetch
- [ ] More aggressive scratch space optimizations
- [ ] Perfetto trace analysis for overlap verification

## Notes

- Scratch optimizations are working and provide sufficient space
- Level 4 precompute builds successfully but has correctness issues
- Need to debug vselect tree logic before measuring performance
- Once correctness is fixed, should see performance improvement from eliminating level 4 gathers
