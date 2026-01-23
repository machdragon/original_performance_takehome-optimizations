# Phase 19 Optimization Exploration Plan

## Current State
- **Cycles**: 2095
- **Target**: < 1790 (need to save 305 cycles)
- **Theoretical minimum**: ~1536 cycles (load-bound)

---

## Investigation 1: Gather Deduplication Analysis

### What We're Checking
The hypothesis is that early rounds have low index uniqueness, meaning we load duplicate values.

### Method
Simulated the traversal to count unique indices at each round:

```
Round  0 (level  0):   1 unique (already optimized - uniform broadcast)
Round  1 (level  1):   2 unique (already optimized - binary select)
Round  2 (level  2):   4 unique <- POTENTIAL: 256 loads → 4 values
Round  3 (level  3):   8 unique <- POTENTIAL: 256 loads → 8 values
Round  4 (level  4):  16 unique <- POTENTIAL: 256 loads → 16 values
Round  5+: Higher uniqueness, diminishing returns
```

### Findings
- Levels 2-4 have huge dedup potential: 1480 loads could be eliminated
- At 2 loads/cycle, that's ~740 cycles potential savings
- BUT: previous arith selection approach REGRESSED

---

## Investigation 2: Why Arith Selection Regressed

### What We're Checking
Understanding why the VALU-based selection (which should save cycles) actually hurts performance.

### Method
1. Calculate theoretical savings:
   - Level 2: 256 load cycles → 85 VALU cycles = 171 cycles saved (theory)

2. Actual measurement:
   - Baseline: 2095 cycles
   - With level 2 arith: 2116 cycles
   - **Regression: +21 cycles** (not 171 saved!)

3. Analyzed bundle utilization during load-heavy cycles:
   - 872 bundles have 3 free VALU slots
   - 2616+ free VALU ops available
   - Arith selection adds ~512 VALU ops
   - Should easily fit in free slots!

### Root Cause Found
**The pipelining breaks when arith selection is enabled.**

Current pipeline:
- Round N-1 hash VALU overlaps with Round N load_offset
- When Round N uses arith (no loads), nothing overlaps with N-1's hash
- Creates a "bubble" that adds cycles

The arith VALU doesn't use the free slots in load-heavy bundles because:
1. Arith rounds have NO loads (intentionally)
2. So there are no load-heavy bundles to share VALU slots with
3. The arith VALU runs in its own bundles, adding to total cycle count

---

## Investigation 3: Potential Fix - Restructured Pipelining

### Hypothesis
If we can overlap arith selection VALU with the NEXT round's loads, we get the savings.

### Method (To Implement)
Instead of:
```
Round N-1 (load-based):  hash_VALU + load_offset
Round N   (arith-based): select_VALU (no overlap!)
Round N+1 (load-based):  hash_VALU + load_offset
```

Do:
```
Round N-1 (load-based):  hash_VALU + load_offset
Round N   (arith-based): select_VALU + load_offset_of_N+1
Round N+1 (load-based):  hash_VALU + load_offset_of_N+2
```

### Challenges
- Need to look ahead to determine which rounds need loads
- Load buffer management becomes more complex
- Index calculations for N+1 depend on N's results (data dependency!)

### Status: NOT YET IMPLEMENTED

---

## Investigation 4: Alternative - Partial Selection

### Hypothesis
Even if full arith selection regresses, partial selection (just level 2) loses only 21 cycles.
What if we fix just the level 2 case more carefully?

### Method (To Try)
1. For level 2 only, try different approaches:
   - Option A: Load all 4 values DURING the round (not in setup)
   - Option B: Use comparison-based selection instead of multiply_add
   - Option C: Interleave level 2 selection with level 3's loads

---

## Next Steps

1. **Profile the pipeline bubble** - Exactly how many cycles are lost at round boundaries?

2. **Prototype lookahead pipelining** - Overlap arith VALU with future round loads

3. **Try comparison-based selection** - May be faster than multiply_add for small levels

4. **Measure combined effect** - If we fix pipelining, what's the actual gain?

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Current cycles | 2095 |
| Target cycles | 1790 |
| Gap to close | 305 cycles |
| Theoretical load minimum | 1536 cycles |
| Current overhead | 559 cycles |
| Free VALU slots during loads | 2616+ |
| Level 2 regression | +21 cycles |
| Level 3 regression | +349 cycles |
| Level 4 regression | +1295 cycles |

---

## Key Insights Gained

### 1. The Kernel is Load-Bound (Near Theoretical Minimum)
- 3072 load_offset operations at 2/cycle = 1536 cycles minimum
- We're at 2095 cycles, only 559 cycles above minimum
- 70.5x speedup from baseline 147734 cycles already achieved

### 2. vselect is NOT in VALU Engine
- vselect instruction is under "flow" engine (1 slot/cycle)
- NOT under "valu" engine (6 slots/cycle)
- This makes vselect-based selection unviable for performance

### 3. Free VALU Slots Exist But Aren't Used
- 872 bundles have 3+ free VALU slots during loads
- 2616+ total free VALU ops available
- Arith selection doesn't use these because it eliminates loads entirely

### 4. Pipelining is the Real Bottleneck
- Cross-round pipelining overlaps hash VALU with next round's loads
- When a round has no loads (arith selection), the overlap breaks
- This creates "pipeline bubbles" that add cycles

### 5. Level 2 Has Smallest Regression
- Level 2 arith: +21 cycles (manageable)
- Level 3 arith: +349 cycles (severe)
- Level 4 arith: +1295 cycles (catastrophic)
- Smaller levels have less VALU overhead relative to pipeline cost

### 6. The Path to 1790 is Clear But Complex
- Need to fix pipelining so arith VALU overlaps with future loads
- OR find a way to reduce VALU ops to fit within existing pipeline slack
- The 305-cycle gap requires careful restructuring, not just flag changes

### 7. Index Uniqueness Pattern is Predictable
- Level N has exactly 2^N unique indices
- This means we KNOW exactly how many values are needed per level
- But exploiting this requires efficient per-lane selection
