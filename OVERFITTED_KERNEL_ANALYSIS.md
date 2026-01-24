# Overfitted Kernel Analysis

## Problem

The `build_kernel_overfitted` method produces **correct** results but is **much slower** than the general kernel:

- **General kernel**: 1864 cycles, 1864 instructions
- **Overfitted kernel**: 3660 cycles, 3660 instructions
- **Correctness**: ✅ Both produce identical results

## Root Cause

The overfitted kernel is missing all the key optimizations that make the general kernel fast:

### Missing Optimizations

1. **No Block-Based Processing**
   - Overfitted: Processes vectors sequentially (`for vec in range(vec_count)`)
   - General: Uses block-based processing (block_size=16, num_blocks=2 for 256 batch)

2. **No Cross-Round Pipelining**
   - Overfitted: Processes rounds sequentially, no overlap
   - General: Overlaps hash of previous block with load of next block

3. **No Prefetching**
   - Overfitted: No prefetch mechanism
   - General: Prefetches next round's loads during arith/level2 rounds

4. **No Level-2 Where-Tree**
   - Overfitted: Loads nodes individually per vector
   - General: Loads 4 nodes once per round, uses vselect tree

5. **No VLIW Bundling Optimizations**
   - Overfitted: Basic bundling
   - General: Lookahead=1024, load pulling, VALU pulling

## Current Overfitted Implementation

The overfitted kernel (lines 882-1057) does:
- Sequential round processing: `for r in range(rounds)`
- Sequential vector processing: `for vec in range(vec_count)` (32 vectors)
- Individual node loads per vector: `load_offset` for each lane
- No block structure
- No pipelining between rounds

## Solution

The overfitted kernel needs to be rewritten to use the same optimization structure as `build_kernel_general`, but with hardcoded constants for (10, 16, 256):

### Required Changes

1. **Use block-based processing**
   - Hardcode: `block_size=16`, `num_blocks=2`, `vec_count=32`
   - Process blocks instead of individual vectors

2. **Implement cross-round pipelining**
   - Double-buffered block loads
   - Overlap hash of block N with load of block N+1
   - Overlap hash of last block of round R with load of first block of round R+1

3. **Add prefetching**
   - Prefetch block 0 of next round during arith/level2 rounds
   - Use `v_node_prefetch` buffer

4. **Add level-2 where-tree**
   - For level==2 rounds, load 4 nodes once
   - Use vselect tree instead of individual loads

5. **Use optimized bundler settings**
   - `lookahead=1024`
   - Load pulling, VALU pulling

## Expected Performance

With these optimizations, the overfitted kernel should achieve:
- **Target**: < 1864 cycles (matching or beating general kernel)
- **Potential advantage**: Can hardcode more constants, eliminate some conditionals

## Next Steps

1. Rewrite `build_kernel_overfitted` to use block-based structure
2. Add cross-round pipelining
3. Add prefetching for level-2 rounds
4. Add level-2 where-tree optimization
5. Test correctness and performance
