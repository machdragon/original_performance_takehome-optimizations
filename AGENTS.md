# Repository Guidelines

## Project Structure & Module Organization
- `perf_takehome.py` is the main entry point. `KernelBuilder.build_kernel` is the performance target, and the file also includes local tests and trace helpers.
- `problem.py` defines the simulator, ISA, and reference kernels. Treat it as the source of truth for correctness.
- `tests/` contains the scoring and correctness harness (`submission_tests.py`) plus a frozen copy of the problem. Do not modify this directory.
- `watch_trace.py` and `watch_trace.html` serve a hot‑reloading Perfetto trace. Running trace tests generates `trace.json` in the repo root.

## Build, Test, and Development Commands
- `python3 perf_takehome.py` runs the local `unittest` suite in `perf_takehome.py`.
- `python3 perf_takehome.py Tests.test_kernel_cycles` runs the cycle-count test only.
- `python3 perf_takehome.py Tests.test_kernel_trace` emits a trace (`trace.json`) for Perfetto.
- `python3 watch_trace.py` serves `watch_trace.html` and proxies Perfetto for live viewing.
- `python3 tests/submission_tests.py` runs the official correctness + speed thresholds.
- `git diff upstream/main tests/` (or `origin/main` if no upstream) should be empty before submitting.

## Coding Style & Naming Conventions
- Python, 4‑space indentation, PEP 8–style naming (`snake_case` functions, `CapWords` classes).
- Prefer small, focused helpers and clear scratch/register naming in `KernelBuilder`.
- No formatter or linter is configured—match existing style and keep changes localized.

## Testing Guidelines
- Framework: `unittest`. Tests use `test_*` methods and are grouped in `Tests` classes.
- Use `tests/submission_tests.py` for the definitive score; `perf_takehome.py` is for local iteration.
- Keep the `tests/` folder intact; the harness assumes the frozen problem and tests are unmodified.

## Optimization Context & Constraints
- This is the original performance take-home: start from a fully serial kernel and progressively exploit accelerator parallelism.
- The simulated machine mirrors TPU-style constraints: scratchpad-managed memory, VLIW slot packing, SIMD vector ops, and Perfetto traces for instruction-level analysis. Multicore exists in the simulator but is intentionally disabled in this version.
- Benchmarks in `Readme.md` indicate target thresholds (e.g., 2164 → 1790 → 1579 → 1548 → 1487 → 1363 cycles); report your cycle count alongside the command used.
- Tooling matters: building and using trace-driven feedback is part of the expected workflow.

## Commit & Pull Request Guidelines
- Commit history favors short, imperative summaries (e.g., “Update readme text”). Follow that pattern.
- PRs should include: a brief perf summary, cycle count achieved, commands run, and confirmation that `tests/` is unchanged. Attach trace instructions or screenshots only if they help explain performance changes.
