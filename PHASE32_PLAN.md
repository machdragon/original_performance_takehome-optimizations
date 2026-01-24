# Phase-32: Scratch Space Optimization for Level 4 Precompute

## Objective
Optimize scratch space usage to enable level 4 precompute + vselect integration. Free at least 128 words (ideally 200-400 words) to accommodate level 4 precompute buffers while maintaining all existing functionality.

## Current Scratch Usage Analysis

### Fixed Allocations (Must Persist)
- **Batch Indices/Values**: 256 + 256 = 512 words (must persist across rounds)
- **Hash Constants/Pointers**: ~16 words (small, can't meaningfully reduce)
- **Block Buffers**: Currently double-buffered (v_node_block_A/B) = 2 * block_size * VLEN = 2 * 16 * 8 = 256 words
- **Temporary Block Buffers**: v_tmp1_block, v_tmp2_block, v_tmp3_block = 3 * 128 = 384 words

### Where-Tree Temporary Buffers (Candidate for Optimization)
- **Level-2 Where-Tree**: Uses vselect operations, may use temporary buffers
- **Level-3 Where-Tree**: Similar temporary buffers
- **Parity/Overshoot Buffers**: If implemented, these could be large (up to 512 words each)

### Level 4 Precompute Requirements
- **Level 4 Values**: 16 scalars = 16 words
- **Level 4 Vectors**: 16 * VLEN = 128 words
- **Total**: ~144 words needed

## Scratch Optimization Techniques

### 1. Merge Overshoot Handling into Parity Buffers ✅ TO IMPLEMENT
**Goal**: Eliminate dedicated overshoot buffers by handling wrap-around in-place
- **Current**: If overshoot buffers exist, they use up to 512 words
- **Optimization**: Handle overshoot directly in parity buffers using vectorized mask operations
- **Savings**: Up to 512 words
- **Implementation**: Modify where-tree logic to zero out-of-range indices in-place rather than extracting to separate list

### 2. Sequential Reuse of Parity Buffers ✅ TO IMPLEMENT
**Goal**: Reuse same buffer for even/odd parity groups sequentially
- **Current**: May allocate separate buffers for even/odd (256 words each = 512 total)
- **Optimization**: Use single 256-word buffer, process even group first, then reuse for odd group
- **Savings**: Up to 256 words
- **Trade-off**: Slightly more complex control flow, but maintains correctness

### 3. Delay Allocation Until Needed ✅ TO IMPLEMENT
**Goal**: Allocate scratch only when features are actually enabled
- **Current**: May pre-allocate buffers even when features are disabled
- **Optimization**: Conditional allocation based on enable flags
- **Savings**: Variable, depends on disabled features
- **Implementation**: Check enable_level2_where, enable_level3_where before allocating

### 4. Round-Local Temporal Reuse ✅ TO IMPLEMENT
**Goal**: Ensure buffers are reused across rounds, not duplicated
- **Current**: Verify buffers are allocated once and reused
- **Optimization**: If any duplication exists, eliminate it
- **Savings**: Variable
- **Status**: Likely already implemented, but verify

### 5. In-Place Updates vs Separate Buffers ✅ TO IMPLEMENT
**Goal**: Use in-place updates instead of separate temporary buffers where possible
- **Current**: May use separate buffers for intermediate results
- **Optimization**: Update indices/values in-place using vectorized operations
- **Savings**: Variable, depends on current implementation
- **Implementation**: Review all temporary buffer usage, convert to in-place where safe

### 6. Buffer Pooling Strategy ✅ TO IMPLEMENT
**Goal**: Create shared pool for all temporary where-tree operations
- **Current**: May have separate allocations for each stage
- **Optimization**: Single shared pool (e.g., 512 words) used by parity, overshoot, and other temp operations
- **Savings**: Up to 256-512 words (depending on current allocation)
- **Implementation**: Allocate one large temp pool, use different regions for different stages

### 7. Optimize Block Buffer Usage ✅ PARTIALLY DONE
**Goal**: Ensure block buffers are efficiently used
- **Current**: Double-buffered (A/B) already implemented
- **Optimization**: Verify no unnecessary duplication, consider if v_tmp blocks can share space with block buffers
- **Savings**: Variable
- **Status**: Double buffering exists, but v_tmp blocks may be redundant

### 8. Alias Temporary Blocks ✅ PARTIALLY DONE
**Goal**: Reuse v_tmp blocks when features don't conflict
- **Current**: alias_tmp3_block flag exists but may not be fully utilized
- **Optimization**: More aggressive aliasing when features are mutually exclusive
- **Savings**: Up to 128 words per aliased block
- **Implementation**: Expand alias_tmp3_block logic, add more aliasing opportunities

### 9. Deeper Prefetch (Lookahead) ✅ FUTURE
**Goal**: Consider triple-buffering if space allows
- **Current**: Double-buffering (block-0 and block-1)
- **Optimization**: If space freed, add third buffer for block-2 prefetch
- **Savings**: Negative (uses more space, but improves performance)
- **Status**: Only if significant space is freed and performance analysis suggests benefit

## Implementation Plan

### Phase 32.1: Analyze Current Scratch Usage
- [ ] Add scratch usage tracking/debugging
- [ ] Measure actual scratch usage with current configuration
- [ ] Identify largest allocations
- [ ] Document which buffers are used when

### Phase 32.2: Implement Buffer Pooling
- [ ] Create shared temp pool for where-tree operations
- [ ] Replace separate parity/overshoot buffers with pool regions
- [ ] Verify correctness with tests

### Phase 32.3: Eliminate Overshoot Buffers
- [ ] Modify where-tree logic to handle overshoot in-place
- [ ] Use vectorized mask operations for wrap-around
- [ ] Remove dedicated overshoot buffer allocations

### Phase 32.4: Optimize Temporary Block Buffers
- [ ] Review v_tmp1_block, v_tmp2_block, v_tmp3_block usage
- [ ] Identify opportunities for aliasing or reuse
- [ ] Implement more aggressive aliasing

### Phase 32.5: Enable Level 4 Precompute
- [ ] Verify scratch space is sufficient
- [ ] Enable level 4 precompute
- [ ] Test correctness
- [ ] Measure performance improvement

### Phase 32.6: Validate with Perfetto Traces
- [ ] Generate trace with level 4 precompute enabled
- [ ] Verify overlap and pipeline utilization
- [ ] Confirm scratch usage is within limits

## Success Criteria

- [ ] At least 128 words of scratch space freed
- [ ] Level 4 precompute enabled and working
- [ ] All correctness tests pass
- [ ] Performance improvement measured
- [ ] Scratch usage documented and verified

## Notes

- Must maintain correctness - all optimizations must preserve algorithm behavior
- Test thoroughly with enable_debug=True to catch any buffer conflicts
- Use Perfetto traces to verify no overlapping buffer usage
- Consider making optimizations configurable via flags for safety
