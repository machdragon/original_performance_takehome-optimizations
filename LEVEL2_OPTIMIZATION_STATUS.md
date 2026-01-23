# Level 2 Arithmetic Selection Optimization Status

## Current State

**Status**: Structure implemented, temporarily disabled due to IndexError  
**Current Performance**: 1923 cycles (level 2 disabled)  
**Target**: <1487 cycles (need 436 more)  
**Potential Savings**: 300-500 cycles (theoretical)

## Implementation Details

### Strategy
- **VALU-only arithmetic selection** (no vselect/flow bottleneck)
- **Maintains pipelining** by doing selection DURING load phase (unlike Phase 17 which broke pipelining)
- **Scratch reuse**: Uses existing temps (v_tmp1-3 for scalars, v_node_block[0] for vectors)
- **No new allocations**: Reuses existing scratch space

### Code Structure
- Level 2 detection: `level2_round = fast_wrap and level == 2`
- Load 4 scalars to v_tmp1, v_tmp2, v_tmp3, v_tmp3 (first word of each vector)
- Broadcast to vectors in v_node_block[0]
- Two-level arithmetic selection using multiply_add
- Reuses v_tmp1_block, v_tmp2_block, v_tmp3_block for computations

### Key Insight
Unlike Phase 17's arith selection (which regressed), this approach:
- Does arithmetic selection **during** the load phase
- Maintains cross-round pipelining (hash VALU overlaps with next round's loads)
- Avoids the "pipeline bubble" that caused Phase 17's regression

## Current Issue

**IndexError**: `list index out of range` in load operation
- Occurs when level 2 optimization is enabled
- Likely causes:
  1. Memory address calculation issue (forest_values_p + 3)
  2. Scratch address access issue (v_tmp1, v_tmp2, v_tmp3)
  3. Closure/variable scope issue with nested function

## Debugging Steps Needed

1. **Verify memory address calculation**:
   - Check that `forest_values_p + 3` is valid
   - Verify level 2 values exist at indices 3-6

2. **Verify scratch addresses**:
   - Confirm v_tmp1, v_tmp2, v_tmp3 are accessible in nested function
   - Check that they're valid scratch addresses

3. **Test with smaller block_size**:
   - Test with block_size=14-15 first to verify approach
   - Then optimize for block_size=16

4. **Alternative approaches**:
   - Use scalar temps instead of vector temps
   - Load to different scratch locations
   - Verify load instruction format is correct

## Test Infrastructure

Created `test_level2_optimization.py` to test with different block sizes:
- Tests block_size 12, 14, 15, 16
- Helps identify scratch-efficient approach
- Current: All block sizes work with level 2 disabled

## Next Steps

1. **Debug IndexError**:
   - Add print statements or assertions to identify exact failure point
   - Verify memory and scratch addresses are valid
   - Check if issue is with load instruction format

2. **Test incrementally**:
   - Enable level 2 for block_size=14 first
   - Verify correctness and performance
   - Then enable for block_size=16

3. **Alternative implementation**:
   - If current approach has fundamental issues, try different scratch reuse strategy
   - Consider using scalar temps (tmp1, tmp2, tmp3) instead of vector temps

## Expected Impact

If successful:
- **Level 2**: Eliminates 256 loads → ~128 cycles saved (at 2 loads/cycle)
- **Level 3**: Could extend to level 3 for additional savings
- **Total potential**: 300-500 cycles (would bring us to ~1400-1600 cycles, close to <1487 target)

## Notes

- The structure is complete and ready
- The approach is sound (maintains pipelining unlike Phase 17)
- Just needs debugging of the IndexError
- Once fixed, this is the highest-impact optimization remaining
