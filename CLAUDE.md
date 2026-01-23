# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Performance take-home: optimize a tree traversal kernel for a simulated VLIW/SIMD accelerator. The goal is to minimize cycle count in `KernelBuilder.build_kernel`.

## Commands

```bash
# Run cycle benchmark (primary metric)
python3 perf_takehome.py Tests.test_kernel_cycles

# Run official submission tests (correctness + all speed thresholds)
python3 tests/submission_tests.py

# Debug with trace visualization
python3 perf_takehome.py Tests.test_kernel_trace
# Then in another terminal: python3 watch_trace.py (opens browser for Perfetto)

# Debug specific parameters
python3 -c "from perf_takehome import do_kernel_test; do_kernel_test(4, 5, 10, enable_debug=True)"
```

## Architecture

### Files
- `perf_takehome.py`: Entry point with `KernelBuilder.build_kernel` (optimization target), local tests, and trace helpers
- `problem.py`: Simulator (`Machine`), ISA definitions, reference kernels, data structures (`Tree`, `Input`)
- `tests/submission_tests.py`: Official correctness + speed harness (**do not modify**)
- `tests/frozen_problem.py`: Frozen copy of simulator for submission testing
- `OPTIMIZATION_SUMMARY.txt`: Phased optimization history and remaining plan

### Simulator Model (VLIW/SIMD)
- **VLIW**: Multiple engines execute slots in parallel per cycle. Slot limits in `SLOT_LIMITS` (alu:12, valu:6, load:2, store:2, flow:1)
- **SIMD**: Vector instructions operate on `VLEN=8` elements. Use `vload`/`vstore` for contiguous memory, `vbroadcast` for scalar-to-vector
- **Memory model**: Effects don't take place until end of cycle (reads before writes)
- **Scratch space**: `SCRATCH_SIZE=1536` words, acts as registers/cache

### Key Constants
- `VLEN = 8` (vector width)
- `N_CORES = 1` (single-core only)
- `SCRATCH_SIZE = 1536`
- `BASELINE = 147734` cycles
- Current best: **4269 cycles** (see OPTIMIZATION_SUMMARY.txt for history)

## Constraints

- Submission harness checks **values only** (not indices), but avoid index-skipping unless explicitly requested
- `Input.generate` always starts indices at 0 (structural invariant used by `assume_zero_indices`)
- Wrap periodicity: after `forest_height + 1` rounds from root, index exceeds bounds
- Debug mode (`enable_debug=True`) adds comparison instructions that match `reference_kernel2` yields

## KernelBuilder Flags
- `enable_debug`: Insert debug compare instructions (slows down execution)
- `assume_zero_indices`: Enable fast wrap optimization (default: True)
- `max_special_level`: Enable early-level gather bypass (-1 = disabled, regressed in testing)

## Guidelines

- Keep changes localized and explain performance impact and correctness risks
- Validate with `test_kernel_cycles` after changes
- Update `OPTIMIZATION_SUMMARY.txt` when adding optimizations
- Focus on real optimization (scheduling, VLIW packing, load/VALU overlap, structure-aware shortcuts)
- Avoid speculative "cheats" (hardcoding outputs, RNG seeding tricks)
- Propose gray-area experiments only behind flags with clear tradeoffs
