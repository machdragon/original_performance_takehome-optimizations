# Double-Buffer Prefetching Implementation

## Summary

Implemented double-buffer prefetching with ping-pong pattern to hide memory latency. Currently **optional** (disabled by default) due to scratch space constraints.

## Implementation

### Changes Made

1. **Added `enable_double_buffer_prefetch` flag** (default: `False`)
   - When `True`: Allocates two prefetch buffers (`v_node_prefetch_A` and `v_node_prefetch_B`)
   - When `False`: Uses single buffer (original implementation, backward compatible)

2. **Ping-Pong Pattern**
   - Round N: Uses buffer A (if prefetched), prefetches round N+1 into buffer B
   - Round N+1: Uses buffer B (prefetched), prefetches round N+2 into buffer A
   - Alternates between buffers to hide memory latency

3. **Buffer Tracking**
   - `prefetch_buffer[round]` tracks which buffer ('A' or 'B') each round should use
   - `prefetch_target` determines which buffer to prefetch into

### Code Locations

- **Allocation**: Lines ~2891-2908 in `build_kernel_general`
- **Buffer Assignment**: Lines ~2962-2980 (ping-pong logic)
- **Usage**: Updated in unrolled and general loop paths
- **Prefetch Target**: Updated in all `vec_block_prefetch_slots` calls

## Scratch Space Impact

- **Single Buffer**: `block_size * VLEN = 16 * 8 = 128 words`
- **Double Buffer**: `2 * block_size * VLEN = 256 words`
- **Additional Cost**: 128 words (causes scratch overflow in current configuration)

## Current Status

- ✅ **Implementation Complete**: Double-buffer prefetching fully implemented
- ✅ **Backward Compatible**: Default behavior unchanged (single buffer)
- ❌ **Scratch Space**: Causes overflow when enabled (needs 128 more words)
- ⚠️ **Optional**: Can be enabled when scratch space allows

## Usage

```python
# Enable double-buffer prefetching (requires sufficient scratch space)
do_kernel_test(10, 16, 256, enable_double_buffer_prefetch=True)
```

## Next Steps

To enable double-buffer prefetching, need to free ~128 words of scratch space:

1. **Reduce block_size** when double-buffering (e.g., block_size=8 instead of 16)
2. **Optimize other scratch allocations** (alias more buffers, reuse temps)
3. **Make block_size adaptive** based on available scratch space
4. **Test with smaller configurations** where scratch space allows

## Performance Results

### Test Results
- **Baseline (block_size=16, single buffer)**: 1865 cycles
- **block_size=8 (single buffer)**: 1941 cycles (76 cycles worse)
- **block_size=8 + double-buffer**: 1941 cycles (no improvement)

### Analysis
- **Smaller block_size hurts**: Reducing from 16 to 8 costs 76 cycles
- **Double-buffer doesn't help**: Same performance as single buffer with block_size=8
- **Conclusion**: Double-buffer prefetching doesn't provide benefit for this workload
  - Memory latency may not be the bottleneck
  - Current single-buffer prefetching is already effective
  - The cost of smaller block_size outweighs any double-buffer benefit

### Performance Potential

- **Theoretical**: 50-100 cycles (hides memory latency better)
- **Actual**: **No improvement** (tested with block_size=8)
- **Trade-off**: 128 words scratch space for no performance gain
- **Recommendation**: **Do not enable** - current single-buffer prefetching is optimal
