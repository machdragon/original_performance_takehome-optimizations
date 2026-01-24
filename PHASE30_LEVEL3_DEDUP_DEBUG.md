# Phase 30: Level 3 Dedup Path Debugging

## Status

**Path is triggering in `submission_perf_takehome.py`** but failing due to scratch space overflow.

## Findings

### 1. submission_perf_takehome.py
- ✅ Has `level3_dedup_round` implementation
- ✅ Path IS being triggered (confirmed by error at `v_level_start = self.scratch_vconst(7)`)
- ❌ Fails with scratch space overflow: `1539/1536 words`
- **Root cause**: Allocating scratch constants (`v_one`, `v_zero`, `v_level_start`, etc.) pushes us over the 1536-word limit

### 2. perf_takehome.py
- ❌ Does NOT have `level3_dedup_round` implementation
- ✅ Only has `level3_round` (requires `enable_level3_where=True`)
- ⚠️ **Tests use `perf_takehome.py`**, so level3_dedup will NOT trigger in tests

## Issues to Fix

1. **Scratch space overflow**: Need to optimize scratch usage or reuse existing constants
2. **Missing implementation in perf_takehome.py**: Need to add Level 3 dedup code there
3. **v_tmp4_block allocation**: Currently disabled due to scratch pressure

## Next Steps

1. **Optimize scratch usage**: Reuse existing constants, avoid redundant allocations
2. **Add to perf_takehome.py**: Copy Level 3 dedup implementation to `perf_takehome.py`
3. **Fix v_one/v_zero references**: Use lazy initialization or reuse from outer scope

## Condition Analysis

The `level3_dedup_round` condition should be True when:
- `fast_wrap=True` (assume_zero_indices=True, enable_debug=False)
- `enable_level3_where=False` (default)
- `enable_prefetch=True` (default, requires enable_level2_where=True OR enable_level3_dedup=True)
- `level == 3` (rounds 3 and 14)
- Not uniform_round, binary_round, arith_round, level2_round, level2_dedup_round

All conditions are met, so the path should trigger (and it does, but fails on scratch allocation).
