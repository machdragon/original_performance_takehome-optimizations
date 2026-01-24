# Phase-30 Optimization Plan

## Current State
- **Current cycles**: 1865 (from submission test)
- **Target**: <1487 cycles (need to save 378 cycles)
- **Level 4 precompute**: Disabled (performance analysis showed it makes things worse)

## Phase-30 Objectives

Phase 30 aims to fine-tune inner loop scheduling and memory access patterns to squeeze out remaining cycles. Focus areas:

1. **Load/VALU Scheduling** - Pack 1-2 loads per bundle and interleave with arithmetic
2. **Deeper Prefetching** - Unroll/look ahead to hide memory latency
3. **VLIW Bundling Enhancements** - Second-pass reorder, latency-aware scheduling
4. **Arithmetic Fusions** - Replace expensive ops (div/mod) with shifts/bitwise
5. **Structure-Aware Tricks** - Exploit tree invariants to reduce redundant work

### Planned Work

1. **Gather Load Analysis** ✅ COMPLETED
   - Analyzed index uniqueness per round
   - Identified high-potential rounds: 0-5 and 11-15 (levels 0-5)
   - Total theoretical savings: 1611.5 cycles
   - Best targets: Rounds 0-4 and 11-14 (120-127.5 cycles each)

2. **VLIW Bundling Enhancements** (IN PROGRESS - user added latency-aware & second-pass)
   - ✅ Latency-aware scheduling: Track loads from previous bundles
   - ✅ Second-pass reorder: Dependency-aware reordering within bundles
   - Test and measure impact on cycle count
   - Potential: 50-100 cycles (from HARNESS_IMPROVEMENTS_SUMMARY)

3. **Gather Deduplication Implementation** (NEXT)
   - For rounds with low uniqueness (levels 2-4):
     - Load unique values once into scratch cache
     - Reuse cached values for duplicate indices
     - **Critical**: Maintain load/VALU overlap to avoid pipeline breaking
   - Key insight: Early rounds (0-4) and wrap rounds (11-14) have low uniqueness
   - Potential savings: 300-500 cycles (theoretical), but must avoid regression
   - Start with Level 2 (Rounds 2, 13) - 126 cycles potential each

4. **Arithmetic Optimizations**
   - Identify expensive operations (divisions, modulos)
   - Replace with shifts/bitwise operations where semantics allow
   - Look for constant multiplications that can use shifts
   - Potential: Small but consistent savings per operation

5. **Deeper Prefetching** (if time permits)
   - Unroll round loop to issue loads further in advance
   - Double-buffer loads (ping-pong pattern)
   - Balance against scratch space pressure
   - Potential: 50-100 cycles if memory latency is bottleneck

6. **Structure-Aware Optimizations** (future)
   - 2-round jump composition (combine two traversal steps)
   - Memoization at upper levels where lanes converge
   - Reuse partial results across rounds

## Implementation Strategy

### Step 1: Analyze Gather Patterns ✅ COMPLETED
- ✅ Added instrumentation to count unique indices per round
- ✅ Measured gather load frequency
- ✅ Identified hot rounds for optimization (rounds 0-5, 11-15)

### Step 2: Test Bundler Enhancements (IN PROGRESS)
- ✅ Latency-aware scheduling implemented
- ✅ Second-pass reordering implemented
- Test performance impact of these features
- Enable by default if beneficial

### Step 3: Implement Gather Deduplication (NEXT)
- Allocate scratch space for unique value cache (per round)
- For each round, collect unique indices
- Load unique values once, broadcast to matching indices
- **Critical**: Maintain load/VALU overlap to avoid pipeline breaking
- Start with Level 2 (Rounds 2, 13) - 126 cycles potential each

### Step 4: Arithmetic Optimizations
- Identify expensive operations (divisions, modulos)
- Replace with shifts/bitwise operations where semantics allow
- Look for constant multiplications that can use shifts

### Step 5: Test and Measure
- Verify correctness with submission test harness
- Measure cycle count improvement
- Profile to ensure no regressions

## Success Criteria

- [x] Gather load analysis completed
- [ ] Bundler enhancements tested and optimized (latency-aware, second-pass)
- [ ] Gather deduplication implemented for at least rounds 2-4 (levels 2-4)
- [ ] Arithmetic optimizations (replace expensive ops with shifts/bitwise)
- [ ] Correctness tests pass
- [ ] Performance improvement measured (target: save 200+ cycles to reach <1487)
- [ ] Code documented and maintainable

## Notes

- Previous attempt at gather deduplication (arith selection) regressed due to pipeline breaking
- Need to ensure deduplication doesn't break load/VALU overlap
- Focus on rounds with highest deduplication potential first (rounds 0-4, 11-14)
- Keep implementation simple and testable
- Bundler enhancements (latency-aware, second-pass) currently show no improvement - may need tuning
- Must balance optimizations against scratch space pressure (Phase 32 constraints)
- Follow GPU optimization principles: hide memory latency, maximize on-chip memory use, use cheap arithmetic

## Optimization Priorities (Based on Analysis)

1. **Gather Deduplication** (Highest potential: 300-500 cycles theoretical)
   - Rounds 0-4 and 11-14: 120-127.5 cycles each
   - Must maintain pipeline to avoid regression
   
2. **Bundler Enhancements** (Medium potential: 50-100 cycles)
   - Latency-aware scheduling: Track loads across bundles
   - Second-pass reorder: Dependency-aware within bundles
   - Currently implemented but not showing improvement - needs investigation

3. **Arithmetic Optimizations** (Low-medium potential: 20-50 cycles)
   - Replace expensive operations with shifts/bitwise
   - Constant multiplications → shifts
   - Modulo operations → bitwise masks where possible

4. **Deeper Prefetching** (Medium potential: 50-100 cycles if memory-bound)
   - Unroll round loop for lookahead
   - Double-buffer loads (ping-pong)
   - Balance against scratch space
