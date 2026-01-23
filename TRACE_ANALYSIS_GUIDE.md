# Perfetto Trace Analysis Guide

## Current Status
- **Cycles**: 1923 (need 436 more to reach <1487)
- **Trace file**: `trace.json` (16MB, already generated)
- **Trace cycles**: 7910 (includes debug overhead)

## How to View Trace

### Method 1: Local Server (Recommended)
```bash
# Terminal 1: Start server
python3 watch_trace.py

# Terminal 2: Open browser
# Navigate to: http://localhost:8000
# Click "Open trace file" and select trace.json
```

### Method 2: Perfetto UI (Online)
1. Go to https://ui.perfetto.dev/
2. Click "Open trace file"
3. Upload `trace.json`

## What to Look For in Perfetto

### 1. Bundle Utilization
- **View**: CPU timeline
- **Look for**: Underutilized bundles (gaps in execution)
- **Goal**: Identify cycles with NOPs or idle engines
- **Action**: If bundles are <80% full, there's room for optimization

### 2. Load Stalls
- **View**: Memory/load operations timeline
- **Look for**: Load operations that could be parallelized
- **Goal**: Identify if we're load-bound (2 loads/cycle limit)
- **Action**: If loads are serialized, try to interleave better

### 3. VALU/ALU Idle Cycles
- **View**: Engine utilization
- **Look for**: VALU/ALU engines sitting idle while loads execute
- **Goal**: Fill VALU/ALU slots during load-heavy cycles
- **Action**: Better instruction scheduling to overlap compute with loads

### 4. Pipeline Bubbles
- **View**: Instruction flow timeline
- **Look for**: Gaps between dependent instructions
- **Goal**: Identify dependency chains causing stalls
- **Action**: Reorder instructions or use delayed issue techniques

### 5. Round Boundaries
- **View**: Round-by-round breakdown
- **Look for**: Overhead at round transitions
- **Goal**: Minimize round boundary overhead
- **Action**: Better cross-round pipelining

## Key Metrics to Check

1. **Bundle Fill Rate**: Average slots used per bundle
   - Target: >90% utilization
   - Current: Check in trace

2. **Load Utilization**: Loads per cycle
   - Target: Close to 2 loads/cycle (limit)
   - Current: Check if we're saturating load engine

3. **VALU Utilization**: VALU ops per cycle
   - Target: Close to 6 VALU/cycle (limit)
   - Current: Check if VALU is idle during loads

4. **Dependency Stalls**: Cycles waiting on dependencies
   - Target: Minimize
   - Current: Identify longest dependency chains

## Perfetto UI Tips

### Useful Views
- **CPU Timeline**: See bundle execution
- **Thread State**: See idle vs. busy cycles
- **Counter Track**: See cycle counts per round
- **Slice Details**: Click bundles to see instruction details

### Filters
- Filter by engine type (load, valu, alu)
- Filter by round number
- Filter by instruction type

### Analysis Queries
- Count bundles per round
- Measure average bundle utilization
- Find longest dependency chains
- Identify most idle cycles

## Expected Findings

Based on current performance (1923 cycles):

1. **Load-bound**: Likely hitting 2 loads/cycle limit frequently
2. **Bundle utilization**: Probably good (80-90%) due to our bundler
3. **VALU idle**: May see VALU idle during load-heavy rounds
4. **Round overhead**: Some overhead at round boundaries

## Next Steps After Analysis

1. **If load-bound**: Focus on level deduplication (where-tree)
2. **If VALU idle**: Better instruction reordering
3. **If bundle gaps**: Manual schedule optimization
4. **If round overhead**: Better cross-round pipelining

## Trace Generation

To regenerate trace with current optimizations:
```bash
python3 perf_takehome.py Tests.test_kernel_trace
```

Note: Trace includes debug overhead, so cycle count will be higher (7910 vs 1923).
