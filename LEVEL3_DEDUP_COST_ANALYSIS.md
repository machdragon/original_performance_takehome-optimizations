# Level 3 Deduplication Cost-Benefit Analysis

## Executive Summary

**Recommendation: Level 3 dedup is NOT worth the scratch cost.**

The analysis shows that Level 3 deduplication will likely make performance WORSE, not better, based on the Level 2 dedup results. Additionally, it requires scratch space that's already at capacity and has implementation bugs.

## Current State

- **Baseline Performance**: 1865 cycles
- **Target**: <1487 cycles (need 378 more cycles)
- **Scratch Space**: Already tight (near 1536-word limit)

## Level 2 Dedup Results (Reference)

From `PHASE30_LEVEL2_DEDUP_FINAL.md`:

- **With `enable_level2_where=True` (default)**: 1865 cycles ✅
- **With deduplication (`enable_level2_where=False`)**: 1924 cycles ❌
- **Result**: **59 cycles WORSE** despite eliminating 512 loads

### Why Level 2 Dedup Failed

1. **VALU Instruction Overhead**: 13 VALU ops per vector offset load savings
2. **Pipeline Efficiency**: Existing `level2_round` path is highly optimized with prefetch integration
3. **VALU Saturation**: More instructions = more cycles, even with 6 VALU slots vs 1 flow slot

## Level 3 Dedup Analysis

### Theoretical Benefits

- **Potential Savings**: 124 cycles per round × 2 rounds = **248 cycles theoretical**
- **Load Reduction**: 512 loads eliminated (2 rounds × 256 loads)
- **Appears in**: Rounds 3 and 14

### Actual Costs

1. **Scratch Space**: 
   - Requires 3 additional scalar registers (`tmp4`, `tmp5`, `tmp6`) = 3 words
   - Already causing scratch overflow (current implementation fails)
   - Would need to free up scratch elsewhere or make allocation conditional

2. **VALU Operations**:
   - **20 VALU ops per vector** (vs 13 for Level 2)
   - Exceeds user's <10 VALU ops target
   - More overhead than Level 2, which already failed

3. **Implementation Issues**:
   - Path not triggering (0 `LEVEL3_DEDUP_TRIGGERED` markers)
   - Has correctness bugs (`IndexError: list index out of range`)
   - Requires fixing before it can even be tested

4. **Performance Prediction**:
   - Based on Level 2 results, Level 3 will likely be **WORSE** than baseline
   - More VALU ops (20 vs 13) = even more overhead
   - Same fundamental problem: VALU overhead offsets load savings

## Comparison: Level 2 vs Level 3

| Metric | Level 2 Dedup | Level 3 Dedup |
|--------|---------------|---------------|
| VALU ops per vector | 13 | 20 |
| Loads eliminated | 512 | 512 |
| Theoretical savings | 252 cycles | 248 cycles |
| Actual result | **-59 cycles** (worse) | **Predicted: worse** |
| Scratch cost | Low | **3 words** (causes overflow) |
| Implementation status | Working | **Broken** (bugs, not triggering) |

## Root Cause

The fundamental issue is that **gather deduplication doesn't improve performance** because:

1. **VALU Overhead > Load Savings**: More VALU instructions take more cycles than saved loads
2. **Existing Path is Optimal**: The `level2_round`/`level3_round` paths are already highly optimized
3. **Pipeline Integration**: Existing paths have better prefetch integration than dedup paths

## Recommendations

### ❌ Do NOT pursue Level 3 dedup because:

1. **Level 2 already showed it doesn't work** (59 cycles worse)
2. **Level 3 has more overhead** (20 VALU ops vs 13)
3. **Scratch space is too tight** (would need to free up 3+ words)
4. **Implementation is broken** (bugs, not triggering)
5. **Predicted to make performance worse**, not better

### ✅ Better alternatives:

1. **Focus on other optimizations** from Phase 30 plan:
   - Bundler refinements (latency-aware, second-pass)
   - Deeper prefetching
   - Arithmetic optimizations
   - Structure-aware tricks

2. **Optimize existing paths** rather than replacing them:
   - Fine-tune `level2_round`/`level3_round` scheduling
   - Improve load/VALU overlap
   - Better VLIW bundling

3. **Remove Level 3 dedup code** to:
   - Reduce code complexity
   - Free up scratch space
   - Avoid maintenance burden

## Conclusion

Level 3 deduplication is **not worth the scratch cost** because:

- **Performance**: Will likely make things worse (based on Level 2 results)
- **Scratch**: Requires space that's already at capacity
- **Implementation**: Has bugs and isn't working
- **ROI**: Negative - costs more than it saves

**Recommendation**: Remove Level 3 dedup implementation and focus on other optimizations that have better ROI.
