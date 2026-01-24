# Phase-29: Level 4 Precompute Integration - PR Summary

## Overview
Implements level 4 tree precomputation with vselect tree integration to eliminate gather loads for level 4 rounds. Includes scratch space optimizations to enable the feature.

## Changes

### Core Implementation
1. **Level 4 Precompute Logic** (`submission_perf_takehome.py`):
   - Added precomputation of level 4 (16 nodes) into scratch vectors
   - Precomputes 16 scalars + 16*VLEN vectors = 144 words
   - Enabled when `max_special_level >= 4` and `assume_zero_indices=True`

2. **VSelect Tree Extension**:
   - Extended `build_vselect_tree` to handle `level_size=16` (4 layers)
   - Added `build_vselect_tree_reuse()` method to avoid runtime scratch allocations
   - Supports selecting from 16 precomputed nodes using binary tree selection

3. **Round Loop Integration**:
   - Added `level4_precompute_round` flag to round_info
   - Integrated vselect tree usage in `vec_block_hash_only_slots`
   - Replaces gather loads with vselect operations for level 4 rounds

### Scratch Space Optimizations (Phase-32)
1. **Temporary Block Aliasing**:
   - When `enable_level2_where=False` and `enable_level3_where=False`, alias `v_tmp2_block` and `v_tmp3_block`
   - **Savings**: 128 words (enables level 4 precompute to fit)
   - **Result**: Scratch usage 1464/1536 words (95%, 72 words headroom)

2. **Enhanced Error Messages**:
   - Added detailed scratch allocation error messages for debugging

### Testing
- Added correctness tests in `perf_takehome.py`
- Tests pass for small batches (batch_size=32, no blocks)
- **Known Issue**: Correctness fails for large batches (batch_size=256, with blocks) - 252/256 mismatches

## Performance Impact

- **Scratch Space**: Optimized from 1524 to 1464 words (freed 60 words + 128 from aliasing = 188 words total)
- **Level 4 Precompute**: Builds successfully, correctness issue blocks performance measurement
- **Expected Savings**: ~100-400 cycles from eliminating level 4 gathers (once correctness is fixed)

## Known Issues

### Correctness Failure in Block Processing
**Status**: ❌ Failing for batch_size=256 (block processing path)
**Status**: ✅ Passing for batch_size=32 (per-vector path)

**Symptoms**:
- 252/256 output values mismatch reference
- Issue is specific to block processing (works for small batches without blocks)

**Root Cause Analysis**:
The vselect tree logic works correctly for per-vector processing (small batches), but fails when processing multiple block items. Possible causes:
1. Temp space reuse conflicts when processing multiple block items
2. Final output space conflicts when `v_tmp3_block` is aliased to `v_tmp2_block`
3. Instruction bundling reordering causing dependency violations

**Current Workaround**:
- Uses `relative_idx_vec` space for final output when aliased (per-item, should be safe)
- Processes items sequentially in instruction stream
- May need explicit dependencies or different temp space strategy

**Next Steps**:
1. Debug block processing path with Perfetto traces
2. Verify temp space usage doesn't conflict across block items
3. Consider alternative final output strategy when aliased
4. Add explicit dependencies if bundler reordering is the issue

## Files Modified

- `submission_perf_takehome.py`: 
  - Level 4 precompute logic (lines ~1160-1195)
  - VSelect tree extension (lines ~512-623, 625-756)
  - Round loop integration (lines ~1797-1845, 2168-2193)
  - Scratch optimizations (lines ~1086-1095, 427-434)
- `perf_takehome.py`: Added correctness tests
- `PHASE30_PLAN.md`: Documented phase-30 and phase-31 work
- `PHASE32_PLAN.md`: Documented scratch optimization techniques
- `PHASE29_STATUS.md`: Status documentation

## Testing

### Correctness
- ✅ Small batches (batch_size=32): PASSES
- ❌ Large batches (batch_size=256): FAILS (252/256 mismatches)
- ✅ Build: Successful (scratch space sufficient)

### Performance
- ⏳ Not yet measured (blocked by correctness)

## Recommendations

1. **Immediate**: Debug block processing correctness issue
   - Focus on temp space reuse across block items
   - Verify final output space doesn't conflict
   - Check for instruction bundling issues

2. **Short-term**: Once correctness is fixed
   - Measure performance improvement
   - Run full test suite
   - Validate with submission harness

3. **Future**: Continue phase-32 scratch optimizations
   - Implement buffer pooling
   - Eliminate overshoot buffers (if they exist)
   - Further optimize block buffer usage

## Notes

- Scratch optimizations are working and provide sufficient space
- Level 4 precompute logic is correct for per-vector processing
- Block processing path needs debugging before marking complete
- All infrastructure is in place; correctness fix needed
