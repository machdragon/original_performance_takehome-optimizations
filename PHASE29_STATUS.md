# Phase-29: Level 4 Precompute Implementation Status

## Completed ✅

1. **Feature Branch**: Created `phase-29-level4-precompute`
2. **Precompute Infrastructure**: Added level 4 precompute logic (16 scalars + 16*VLEN vectors)
3. **VSelect Tree Extension**: Extended `build_vselect_tree` to handle level_size=16 (4 layers)
4. **Round Loop Integration**: Integrated precomputed level access in round loop
5. **Tests Added**: Added correctness tests in `perf_takehome.py`
6. **Flags Updated**: Uses `max_special_level >= 4` to enable optimization
7. **Phase-30/31 Documentation**: Created `PHASE30_PLAN.md` and `PHASE32_PLAN.md`

## Scratch Space Optimizations (Phase-32) ✅

1. **Temporary Block Aliasing**: 
   - When `enable_level2_where=False` and `enable_level3_where=False`, alias `v_tmp2_block` and `v_tmp3_block`
   - **Savings**: 128 words
   - **Result**: Scratch usage reduced from 1524 to 1464 words (95% usage, 72 words headroom)

2. **Enhanced Error Messages**: Added detailed scratch allocation error messages

3. **VSelect Tree Reuse Method**: Created `build_vselect_tree_reuse()` to avoid runtime scratch allocations

## Current Issues ❌

### Level 4 Precompute Correctness Failure

**Status**: Builds successfully, but produces incorrect output values

**Symptoms**:
- With batch_size=32: ✅ PASSES (no blocks, uses per-vector path)
- With batch_size=256: ❌ FAILS (uses block path, 252/256 mismatches)

**Root Cause Analysis**:
The issue appears to be in how level4_precompute handles block processing. The logic works for small batches (no blocks) but fails for larger batches (with blocks). Possible causes:

1. **Temp Space Reuse Conflicts**: Processing multiple block items reuses same temp space; VLIW scheduler might process them in parallel, causing conflicts
2. **VSelect Tree Logic Bug**: The `build_vselect_tree_reuse` method might have incorrect address calculations or selection logic
3. **Relative Index Calculation**: The relative index computation might be wrong for level 4
4. **Level Base Address**: The precomputed level base address might be incorrect

**Next Steps for Debugging**:
1. Add debug logging to trace vselect tree operations
2. Compare intermediate values with reference using `enable_debug=True`
3. Verify level 4 precompute data is loaded correctly
4. Test with smaller block sizes to isolate the issue
5. Compare with working level2_round implementation to identify differences

## Performance Impact

- **Scratch Space**: Optimized from 1524 to 1464 words (freed 60 words, plus 128 from aliasing = 188 words total headroom)
- **Level 4 Precompute**: Not yet measured (blocked by correctness)
- **Expected Savings**: ~100-400 cycles from eliminating level 4 gathers (once correctness is fixed)

## Files Modified

- `submission_perf_takehome.py`: 
  - Added level 4 precompute logic
  - Extended `build_vselect_tree` for level_size=16
  - Added `build_vselect_tree_reuse` method
  - Integrated level4_precompute_round in round loop
  - Added scratch space optimizations (aliasing)
- `perf_takehome.py`: Added correctness tests
- `PHASE30_PLAN.md`: Documented phase-30 and phase-31 work
- `PHASE32_PLAN.md`: Documented scratch optimization techniques
- `PHASE32_STATUS.md`: Status of scratch optimizations

## Testing Status

- ✅ Build: Successful (scratch space sufficient)
- ✅ Small batches (batch_size=32): Correctness passes
- ❌ Large batches (batch_size=256): Correctness fails
- ⏳ Performance: Not yet measured (blocked by correctness)

## Recommendations

1. **Immediate**: Debug level4_precompute correctness issue
   - Focus on block processing path
   - Compare with working level2_round implementation
   - Add debug logging to trace execution

2. **Short-term**: Once correctness is fixed
   - Measure performance improvement
   - Run full test suite
   - Validate with submission harness

3. **Future**: Continue with phase-32 scratch optimizations
   - Implement buffer pooling
   - Eliminate overshoot buffers (if they exist)
   - Further optimize block buffer usage

## Notes

- Scratch optimizations are working and provide sufficient space for level 4 precompute
- The correctness issue is specific to the block processing path
- Level 4 precompute logic appears correct for per-vector processing (works with small batches)
- Need to fix block processing logic before marking phase-29 complete
