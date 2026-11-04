# status.md — Progress Log

## Phase checklist
- [x] Prompt 1 — Scaffolding + read the original Fortran (2026-11-03)
- [x] Prompt 2 — Toolchain + first-pass Python port + per-routine speed comparison (2026-11-03)
- [x] Prompt 3 — Full-run validation + first hot spot (fused xint) (2026-11-04)

## Log

### 2026-11-03
- Created `CLAUDE.md`, `architecture.md`, `status.md`, `PROMPTS.md`, `README.md`, `.gitignore`
  (pattern from `Documents/oist_stuff/gnn_try`), empty `src/ scripts/ tests/ data/ results/`.
- Downloaded `sd.tar.gz` (parallel) and `etas8pr.rar` (serial) into `original/archives/`,
  extracted to `original/sd_parallel/` and `original/etas8pr_serial/`.
- Read all Fortran; full walkthrough + cost model in `architecture.md`.
- Key findings:
  - Dominant cost = numerical polygon integrals in `xint` (4 `polyint` × 27000 kernel evals
    per event per likelihood call, 443 calls) ≈ 4.8e11 evals, ~30× the O(N²) pair sums.
  - Fixed 11-outer-iteration schedule; MLE already converged at outer iteration 4.
  - Memory is O(N) (static 29000 cap); O(N²) only in `pmatr.dat` output.
  - `probs.dat`/`pmatr.dat` only written by ranks 2/1 -> need ≥ 3 MPI ranks.
- Shipped reference outputs available for validation: `jap.out`, `probs.dat`, `rates.dat`,
  `para`, `fort.46` (no `pmatr.dat`). Final MLE in `para`.
- Env: no gfortran, no MPI, WSL installed but no distro.

**Next steps**: Prompt 2 — set up a Fortran toolchain and reproduce the shipped outputs.

### 2026-11-03 (cont.) — Python port, first pass
- User decision: port to Python, keep roughly the same speed and memory.
- Conda env `sdecl` (py3.12, numpy, numba, gfortran 16.2). Fortran built single-process with
  an MPI stub (`scripts/build_fortran.sh`, `scripts/fortran/`). Data copied to `data/jma/`.
- Python port in `src/etas8py/` (Numba kernels, davidn, driver). Benchmarks:
  `build/fortran/bench.exe` (from `scripts/fortran/bench_main.f`) vs `scripts/bench_python.py`.
- Result (table in `architecture.md`): speed on par (func15 79 s Python vs 95 s Fortran;
  bkgdcalc 18 vs 16; bandwcalc 0.85 vs 0.24). Outputs bit-identical for one pass of
  bandw/bkgd/prob/func15/outrates. Memory 90 MB vs 7 MB, all fixed runtime overhead.
- `tests/test_davidn.py`: Python davidn == Fortran davidn on chained Rosenbrock (357 evals,
  identical x). Passes.
- Not yet run: the full schedule (≈ 12 h each, single process), so `run.py` end-to-end
  (schedule, dav loop, para/probs output) is untested on real data.
- Shipped `jap.out` came from 10 MPI ranks -> different summation order, so expect
  agreement to ~1e-10 relative, not bit-exact, vs shipped `probs.dat`/`para`.

**Next steps**: smoke-test `python -m etas8py.run --schedule short`; then full-run
validation (overnight, Fortran + Python) vs shipped outputs; then first optimisation (`xint`).

### 2026-11-03 (cont.) — full runs launched, xint optimisation
- Full-schedule runs started in background: Fortran single-process (`build/fortran/full_run/`)
  and Python 1:1 port (`results/python_full/`). Python smoke check OK: first -logL
  15380.697195234 vs shipped 15380.6971952371.
- Optimisation 1 — fused xint polygon integrals (`--fast-xint`), details in `architecture.md`.
  func15 8.7 s vs 33.5 s (Python port) vs 42.4 s (Fortran), same session. f bit-identical.
  `tests/test_xint_fast.py` passes (max rel diff 2e-12).
- Found: machine speed drifts ~2× between sessions -> first benchmark table is "slow state".
- Third background run: full schedule with `--fast-xint` -> `results/python_full_fastxint/`.
- When all done: `python -I scripts/compare_runs.py` (python vs fortran vs shipped, fastxint
  vs python), then record results + wall times here.

- Fast-xint full run done: wall 5545 s (1.54 h; dav 4699 s, bkgdcalc 536, probcalc 209,
  outrates 99), 529 func15 calls. Final -logL 15310.95186130678 vs shipped jap.out
  15310.9518614449 (diff 1.4e-7, ~1e-11 rel); per-iteration -logL and MLE track shipped
  jap.out at every outer iteration. Shipped has 443 func15 calls (line search diverges at
  roundoff level once converged).
- Found: shipped `para`/`probs.dat`/`rates.dat` (2011) are from an older run than `jap.out`
  (2016): final mu 0.2857 vs 0.5506. Only `jap.out` is a valid shipped reference; probs/rates
  must be checked against our own Fortran run. `compare_runs.py` now reads MLE from the log.
- Python 1:1 port full run done: wall 18605 s (5.17 h; dav 17656 s), 521 func15 calls,
  final -logL 15310.95185705 (4.4e-6 from shipped jap.out). Fast-xint is 3.4x faster end to
  end (dav 3.8x).
- Fast-xint vs port, full schedule: -logL 4.3e-6 abs (3e-10 rel), para 3e-7 rel, zprob 5e-7
  max abs (sum 5489.89996 vs 5489.90023), zbkgd 3e-9, rates lambda 1.7e-5 abs on max 55.8
  (3e-7 rel; the 1e-4 "max rel" is at near-zero cells). Matches the davidn stopping-tolerance
  scale: different roundoff -> different stopping point, not a bias.
- Fortran single-process full run done: wall 22245 s (6.18 h), 521 func15 calls.
  Python port vs Fortran, full schedule: bit-identical (same 521 calls, para file identical,
  probs.dat and rates.dat identical, -logL equal to print precision). Fortran -logL is 4.4e-6
  from shipped jap.out (10-rank MPI summation order).
- End to end: Fortran 6.18 h, Python port 5.17 h, fast-xint 1.54 h (4.0x vs Fortran).

### 2026-11-04 — parallel comparison, OOP API
- MS-MPI (conda-forge msmpi 10.1.1) installed; real MPI build in `build/fortran_mpi/`.
  Single-call sweep vs Python prange (`--threads`): table in `architecture.md`. func15:
  Python fast ~4.5x faster than MPI Fortran at every core count; bkgd/prob at parity.
- Python parallel kernels (`*_par`, incl. outrates) bit-identical to serial
  (`tests/test_parallel.py`). Full run fast-xint + 10 threads: 1002 s (outrates serial 87 s
  of it; outrates now parallel, not yet timed).
- OOP API `etas8py/declustering.py` (Catalog, Region, ETASParams, Settings,
  StochasticDeclustering); hyperparameter table in `architecture.md`. Exposed `ndiv` and
  davidn `tol`/`max_sweeps` (defaults = original; bench output still bit-identical to
  Fortran). `tests/test_declustering.py` OOP == run.Model bit for bit.
- Full runs at 10 cores: Fortran MPI np=10 4075 s (inflated: tests ran alongside for
  ~25 min; per-call estimate ~3500 s), 512 func15 calls, -logL 1.6e-6 from shipped jap.out
  (also 10 ranks; not bit-identical: different compiler/hardware). Python fast-xint 10
  threads 1002 s, bit-identical to the serial fast-xint run. ~3.5-4x end to end.
- Parallel outrates at 10 threads: 17.7 s (vs ~90 s serial) -> Python full run ~930 s.
- Clean rerun of everything (`scripts/rerun_all.sh`): tests pass; Python 1:1 bench ==
  Fortran bench bit for bit; full runs at 10 cores: Fortran MPI 3965 s, Python 940 s, OOP API
  961 s (API == run.py byte for byte). Table in `architecture.md`.
- Measured flat-projection error on JMA (pairs < 1 deg): flat/great-circle 0.94-1.06 (p1-p99),
  0.89-1.10 extremes. Polygon integral flips sign for clockwise vertex order (verified).
- Grill of "expose all hyperparameters" plan: use projections not great-circle; unit
  conversion for D/bwm; polygon validation; keep bit-exact `original` preset; TOML config.

**Next steps**: next hot spots: bkgdcalc's polyint (same fusion trick:
endpoint sharing), then O(N²) xlamb sum (space/time cutoffs), convergence test on schedule.
