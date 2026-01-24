# Phase-30 Optimization Plan

## Current State
- **Current cycles**: 1865 (from submission test)
- **Target**: <1487 cycles (need to save 378 cycles)
- **Level 4 precompute**: Disabled (performance analysis showed it makes things worse)

## Phase-30 Objectives

### Primary Focus: Gather Pressure Reduction
The highest-potential optimization is reducing gather loads through deduplication for early rounds where index uniqueness is low.

### Planned Work

1. **Gather Load Analysis**
   - Analyze index uniqueness per round (especially rounds 1-3, 12-14)
   - Measure current gather load counts
   - Identify rounds with highest deduplication potential
   - Estimate cycle savings from eliminating duplicate loads

2. **Gather Deduplication Implementation**
   - For rounds with low uniqueness (e.g., level 2-4):
     - Load unique values once into scratch cache
     - Reuse cached values for duplicate indices
     - Avoid redundant `load_offset` instructions
   - Key insight: Early rounds (1-3) and wrap rounds (12-14) have low uniqueness
   - Potential savings: 300-500 cycles (from PHASE19_PLAN analysis)

3. **Bundler Refinements** (if time permits)
   - Add latency-aware scheduling rules
   - Second-pass reorder within bundles to keep load limits saturated
   - Small unroll factors to increase ILP

4. **Structure-Aware Optimizations** (future)
   - 2-round jump composition (combine two traversal steps)
   - Memoization at upper levels where lanes converge

## Implementation Strategy

### Step 1: Analyze Gather Patterns
- Add instrumentation to count unique indices per round
- Measure gather load frequency
- Identify hot rounds for optimization

### Step 2: Implement Deduplication Cache
- Allocate scratch space for unique value cache (per round)
- For each round, collect unique indices
- Load unique values once, broadcast to matching indices
- Reuse cached values instead of redundant loads

### Step 3: Test and Measure
- Verify correctness with submission test harness
- Measure cycle count improvement
- Profile to ensure no regressions

## Success Criteria

- [ ] Gather load analysis completed
- [ ] Gather deduplication implemented for at least rounds 1-3
- [ ] Correctness tests pass
- [ ] Performance improvement measured (target: save 100+ cycles)
- [ ] Code documented and maintainable

## Notes

- Previous attempt at gather deduplication (arith selection) regressed due to pipeline breaking
- Need to ensure deduplication doesn't break load/VALU overlap
- Focus on rounds with highest deduplication potential first
- Keep implementation simple and testable
