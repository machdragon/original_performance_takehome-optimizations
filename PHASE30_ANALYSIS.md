# Phase 30: Gather Load Pattern Analysis Results

## Analysis Summary

Analyzed index uniqueness per round to identify gather deduplication opportunities.

## Key Findings

### High-Potential Rounds

**Early Rounds (0-5):**
- Round 0 (Level 0): 1 unique index, 255 duplicates → **127.5 cycle potential**
- Round 1 (Level 1): 2 unique indices, 254 duplicates → **127.0 cycle potential**
- Round 2 (Level 2): 4 unique indices, 252 duplicates → **126.0 cycle potential**
- Round 3 (Level 3): 8 unique indices, 248 duplicates → **124.0 cycle potential**
- Round 4 (Level 4): 16 unique indices, 240 duplicates → **120.0 cycle potential**
- Round 5 (Level 5): 32 unique indices, 224 duplicates → **112.0 cycle potential**

**Wrap Rounds (11-15):**
- Round 11 (Level 0): 1 unique index, 255 duplicates → **127.5 cycle potential**
- Round 12 (Level 1): 2 unique indices, 254 duplicates → **127.0 cycle potential**
- Round 13 (Level 2): 4 unique indices, 252 duplicates → **126.0 cycle potential**
- Round 14 (Level 3): 8 unique indices, 248 duplicates → **124.0 cycle potential**
- Round 15 (Level 4): 16 unique indices, 240 duplicates → **120.0 cycle potential**

### Summary by Level

| Level | Rounds | Total Potential Savings |
|-------|--------|-------------------------|
| Level 0 | 2 | 255.0 cycles |
| Level 1 | 2 | 254.0 cycles |
| Level 2 | 2 | 252.0 cycles |
| Level 3 | 2 | 248.0 cycles |
| Level 4 | 2 | 240.0 cycles |
| Level 5 | 1 | 112.0 cycles |
| Level >5 | 5 | 250.5 cycles |

**Total Theoretical Potential: 1611.5 cycles**

## Implementation Strategy

### Priority Targets
1. **Rounds 0-4 and 11-14** (Levels 0-4): Highest potential, 120-127.5 cycles each
2. **Round 5** (Level 5): Good potential, 112.0 cycles
3. **Rounds 6-10** (Level >5): Lower priority, 13.5-97.0 cycles

### Key Constraints
- Previous arith selection attempts regressed due to **pipeline breaking**
- Must maintain **load/VALU overlap** to avoid regression
- Need to ensure deduplication doesn't eliminate loads entirely (which breaks pipelining)

### Recommended Approach
1. **Cache-based deduplication**: Load unique values once, broadcast to matching indices
2. **Maintain load pipeline**: Still emit some loads to maintain cross-round overlap
3. **Start with Level 2-4**: Moderate complexity, high savings (120-126 cycles each)
4. **Test incrementally**: Verify no pipeline regression at each step

## Next Steps

1. Implement gather deduplication for Level 2 (Round 2, 13)
2. Test for correctness and performance
3. Extend to Level 3 and 4 if successful
4. Measure actual cycle savings vs. theoretical
