# Phase-29: Level 4 Precompute Performance Analysis

## Performance Measurement Results

### Test Configuration
- **Test parameters**: (forest_height=10, rounds=16, batch_size=256) - official submission test
- **Baseline**: `KernelBuilder()` with default parameters (max_special_level=-1)
- **With level4_precompute**: `KernelBuilder(max_special_level=4)`

### Results

#### Small Batch (32 items, per-vector processing)
- **Baseline**: 57 cycles
- **With level4_precompute**: 114 cycles
- **VSelect instructions**: 208
- **Result**: **+57 cycles WORSE (100% slower)**

**Analysis**: Level4 precompute is enabled and working (208 vselect instructions), but the vselect tree overhead (4 layers for 16 nodes) is more expensive than simple gather loads.

#### Large Batch (256 items, block processing)
- **Baseline**: 113 cycles (perf_takehome.py) / 1865 cycles (submission test)
- **With level4_precompute**: **DISABLED** (correctness issue)
- **Result**: No impact (disabled for block processing)

**Analysis**: Level4 precompute is disabled for block processing due to bundler reordering conflicts. Even if enabled, it wouldn't help because the vselect overhead exceeds gather cost.

## Conclusion

**Level4 precompute optimization should be DISABLED/REMOVED:**

1. **Performance**: Makes things WORSE when enabled (57 cycles worse for small batches)
2. **Correctness**: Doesn't work for block processing (bundler conflicts)
3. **Scratch usage**: Uses 144 words that could be used for other optimizations
4. **Complexity**: Adds significant code complexity for no benefit

### Recommendation

**Disable level4_precompute by default** and consider removing it entirely:
- Set `max_special_level` default to 3 (not 4)
- Or remove level4_precompute code entirely
- Focus optimization efforts on other areas with better ROI

### Alternative: Optimize VSelect Tree

If we want to keep level4_precompute, we'd need to:
- Reduce vselect tree depth (fewer layers)
- Optimize vselect instruction scheduling
- But this is likely not worth the effort given the negative performance impact
