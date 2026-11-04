# architecture.md — Original Code Walkthrough & Design

## Source layout (`original/`, read-only)
- `archives/` — `sd.tar.gz` (parallel MPI version + JMA example), `etas8pr.rar` (serial port
  by Qi Wang, Windows Compaq Visual Fortran project).
- `sd_parallel/` — the code we study. Prebuilt `etas8p` is a Linux x86-64 ELF.
- `etas8pr_serial/etas8p/` — serial port. Prebuilt `etas8p` is an SGI MIPS binary; `Debug/etas8p.exe`
  is a Windows build. `jpsh.dat` is empty (0 lines), so its `jap.in` cannot run as shipped.

| File | Role |
|---|---|
| `etas8p.f` | main program, input (`readata`), work split (`para`), bandwidths, `probcalc`, likelihood `func15`, `dav` (calls optimiser), `outpmat` |
| `etas8fun.f` | `bkgdcalc`, kernels `dgauss/pgauss` (+unused Cauchy), intensity `xlamb`, integral `xint`, radial antiderivatives `fr, dd_fr, dq_fr, dgamma_fr` |
| `frint.f` | `polyint`: integral of a radial function over the polygon (Green's theorem along edges) |
| `poly.f` | `polyse` point-in-polygon, `ycentroid` |
| `davidn.f` | Davidon (DFP-type) quasi-Newton minimiser + line search (ISM legacy code) |
| `etas8out.f` | `outrates` (grid of background/total rate), `outprob` |
| `common.inc` | shared COMMON blocks; static arrays sized 29000 events |

Run: `mpirun -np K etas8p < jap.in > jap.out`. Shipped `jap.out` was made with 10 ranks.

## Model
Space-time ETAS conditional intensity, magnitudes `m = M - M0`:

    λ(t,x,y) = μ·u(x,y) + Σ_{t_j<t} A e^{α m_j} · (p-1)/c (1+(t-t_j)/c)^{-p}
                                    · (q-1)/(π D e^{γ m_j}) (1 + r²/(D e^{γ m_j}))^{-q}

8 parameters `b = (μ, A, c, α, p, D, q, γ)`. The optimiser works on `x = sqrt(b)` to keep
them positive; this is why `b(k)**2` / `2*b(k)` chain-rule factors appear everywhere.

Background `u` = variable-bandwidth Gaussian kernel estimate weighted by background probs:

    zbkgd(i) = 1/(T - T0) · Σ_j φ_j · Gauss(r_ij; h_j)        (sum includes j = i)
    h_j      = max(distance to np-th nearest event, bwm)
    φ_i      = zprob(i) = μ·zbkgd(i) / λ(t_i, x_i, y_i)        (prob. event i is background)

Log-likelihood over target events (`ind=1`) minus the integral of λ over [T0,T]×polygon:

    log L = Σ_{i target} log λ_i  -  μ·Σ_i φ_i ∫_S Gauss_i  -  Σ_i A e^{α m_i} · G_i · S_i

`G_i` = closed-form Omori time integral; `S_i` = spatial kernel integral over the polygon,
computed numerically by `polyint`. All events act as triggers; only targets enter `Σ log λ`.

## Program flow (`sd_parallel/etas8p.f`)
1. **`readata`** (rank 0 reads stdin = `jap.in`, then `mpi_bcast`):
   catalog file, format (unused), `tz` = time span (days), `nn` (ignored, recounted),
   `zmin` start time, `xmg0 tstart` = M0 and target start, polygon vertices, grid
   `mx my` + ranges, initial `b`, `npp bwm`, `ioptimise`.
   Catalog columns: `id lon lat mag t(days) depth year month (day)`. Must be time-sorted.
   Longitudes are multiplied by `cos(centroid lat)` -> flat local degrees. Keeps events with
   `M >= M0`, `zmin < t <= zmin+tz`. Targets: inside polygon and `t >= tstart`.
   JMA example: 10071 events kept, 4656 targets. Dumps all events to `fort.46`.
2. **`para`** — static work split per rank. `ista..iend` is a sqrt split (balances the
   triangular `j < i` loops in `xlamb`); `ista2..iend2` is an even split.
3. **`bandwcalc`** — brute-force np-nearest-neighbour distance per event (insertion into a
   sorted buffer). Results gathered with a per-rank `mpi_bcast` loop (= hand-rolled allgatherv).
4. **Main loop** — fixed schedule, no convergence test:

       do k = 1..11
         repeat (12-k) times: bkgdcalc ; probcalc      ! update u and φ, params fixed
         if ioptimise: dav()                           ! MLE of 8 params, u and φ fixed
   -> 66 background/prob passes, 11 MLE runs. Shipped log: **443 `func15` calls total**;
   the MLE is unchanged to 5 digits from outer iteration 4 onward (iterations 5-11 wasted).
5. **Outputs** — `rates.dat` (rank 0), `probs.dat` (rank 2), `pmatr.dat` (rank 1),
   `para` (latest MLE, rewritten after each `dav`), `fort.46`, stdout.

### Key routines
- `bkgdcalc`: O(N²) Gaussian sum for `zbkgd`; then `xint0 = Σ φ_i ∫_S Gauss_i` via `polyint`.
- `probcalc`: `zprob(i) = μ zbkgd(i)/λ_i` for **all** events. Calls `xlamb`, which also
  computes 7 unused gradient terms. Prints all N probs to stdout every pass (66×10071 lines,
  most of the 87k-line `jap.out`).
- `xlamb(i)`: λ at event i + 8-component gradient. O(i) loop over earlier events; each pair
  costs 2 `exp`, 2 `pow`, 2 `log`.
- `xint(i)`: time integral closed form; spatial integral = **4 `polyint` calls** (value + d/dD,
  d/dq, d/dγ).
- `polyint`: splits each of `npoly` edges into `ndiv=1000` pieces; per piece `frint` applies
  Simpson in angle with 3 radial evaluations (endpoints + angle-bisector point).
  Cost: `3·npoly·1000` kernel evals per call (27000 for JMA's 9-gon).
- `func15`: `-log L` + gradient. Loop 1 (targets, sqrt split): `xlamb`. Loop 2 (all events,
  even split): `xint`. Sums via `mpi_reduce` + `mpi_bcast` (= allreduce), 9 scalars each.
- `outrates`: rates on a 401×401 grid, O(G·N) with G = mx·my.
- `outpmat`: all pairs `ρ_ij > 1e-20`, O(N²) time and O(N²) disk.

## Cost model (N = 10071 events, N_t = 4656 targets, P = 9 polygon edges, D = 1000)
| Step | Time per call | Calls | Total (rough) |
|---|---|---|---|
| `bandwcalc` | O(N²·np) | 1 | 5e8 |
| `bkgdcalc` | O(N²) + O(N·3PD) | 66 | 66 × 3.7e8 ≈ 2.4e10 |
| `probcalc` | O(N²/2), 8-term grad | 66 | 66 × 5e7 ≈ 3.3e9 pairs |
| `func15` loglik part | O(N_t·N) | 443 | 443 × ~3.6e7 ≈ 1.6e10 pairs |
| **`func15` integral part** | **O(N·4·3PD) ≈ 1.1e9 kernel evals** | 443 | **≈ 4.8e11** |
| `outrates` | O(G·N) | 1 | 3.2e9 |
| `outpmat` | O(N²) + disk | 1 | 5e7 |

**Dominant cost: the numerical polygon integrals in `xint`, ~30× everything else.**
Second tier: the O(N²) pairwise sums (`xlamb`, `bkgdcalc`) — these set the asymptotic scaling.

Memory: O(N) only — no N×N matrix is kept. Hard cap = static arrays of 29000 events
(`common.inc`, `itemp`), no bounds check. O(N²) appears only on disk (`pmatr.dat`).

## Quirks / bugs noticed
- `outprob` runs only on rank 2 and `outpmat` only on rank 1 -> need ≥ 3 MPI ranks. The
  serial port sets `myrank=0`, so it never writes `probs.dat` / `pmat.dat`.
- `frint`: `if(abs(theta).gt.1d0) theta=1d0-1e-10` also maps `theta < -1` to ~+1.
- No convergence check in the main loop; schedule hard-coded (11 outer, 12-k inner).
- `fmt` and input `nn` are read and ignored. Events above 29000 overflow silently.
- `zbkgd` self-term `j = i` is included (by design in ZOV2002? — check paper).
- Serial port differs: 4 warm-up bkgd/prob passes, then 10×(3 passes + `dav`), always optimises.

## Improvement candidates (not decided yet)
- `xint`: cache θ-independent geometry (segment radii, angles), adaptive / fewer `ndiv`, or
  closed-form / far-field shortcuts for events far from the boundary (`S_i ≈ 1`).
- Pairwise sums: truncate negligible pairs (spatial cutoff via k-d tree / grid, Omori time
  cutoff) -> O(N log N)-ish instead of O(N²).
- `probcalc`: value-only `xlamb` (skip 7 gradient terms).
- Convergence test on params / φ instead of fixed schedule.
- k-d tree for `bandwcalc` nearest neighbours.
- Dynamic arrays instead of the 29000 cap; drop the per-pass stdout dump.

## Design decisions
- **2026-11-03: Python + Numba port as the baseline** (`src/etas8py/`). First pass is a
  faithful 1:1 translation (same loops, operation order, wasted work, single-precision
  literals) so it is bit-comparable with the Fortran; optimisations come after.
  - `kernels.py`: `@njit(cache=True)` ports of bandwcalc, bkgdcalc, xlamb, xint, polyint/frint
    (kernel picked by an int `kind` instead of a function pointer), probcalc, func15, outrates.
  - `davidn.py`: plain-Python goto-free port of `linear`/`davidn` (8 params, cost is in
    func15). Unreachable "label 500" path dropped; uninitialised `ig` assumed 0.
  - `run.py`: readata (list-directed input emulation, polyse, ycentroid), schedule, outputs.
    Skips the per-pass zprob stdout dump and `pmatr.dat`.
- Fortran reference built single-process (MPI stub). Patch in build copy only: `outprob`
  writes on rank 0 (was rank 2). `outpmat` (rank 1) therefore never runs.

## Single-process benchmark (2026-11-03, JMA example, one call each, `scripts/bench_*`)
| Routine | Fortran -O2 (s) | Python/Numba (s) |
|---|---|---|
| readata | 0.08 | 0.08 |
| bandwcalc | 0.24 | 0.85 |
| bkgdcalc | 15.9 | 17.9 |
| probcalc | 6.7 | 6.8 |
| func15 | 94.7 | 79.2 |
| outrates | 250.9 | 217.2 |
JIT warm-up (cached after first time) ≈ 10 s. Results bit-identical: f, gradient, xint0,
zbkgd, zprob, and all of `rates.dat`. Peak memory: Fortran ≈ 7 MB, Python ≈ 90 MB
(≈ 59 MB is numpy+numba import; model arrays 0.6 MB). Both O(N).
Full original schedule single-process ≈ 443 × func15 + 66 × (bkgd+prob) ≈ **12 h** (at
these timings; see below — the machine later ran ~2.2× faster).

**Timing caveat:** absolute timings on this laptop drift ~2× between sessions (power state).
Only compare numbers measured in the same session.

## Optimisation 1: fused `xint` polygon integrals (2026-11-03)
`kernels.polyint_etas4` / `xint_fast`, enabled by `func15(..., fast_xint=True)` /
`run.py --fast-xint`. Same quadrature as the original, but per event:
- segment geometry (sqrt, det, acos) once instead of 4× (one per integrand);
- `ssig = D e^{γm}` once per event instead of in every radial call;
- each sub-segment endpoint evaluated once (shared by neighbours): 2 radii/piece, not 3;
- all four integrands from one `t = 1 + r²/ssig`, one `pow` + one `log` per radius
  (`t^(1-q)` computed as `t·t^-q` -> rounding-level differences only).
No extra memory (a cross-call geometry cache would need N × 27k doubles ≈ 2.2 GB; rejected).

Measured (same session, full JMA data, initial params after one bkgd/prob pass):
| | func15 (s) |
|---|---|
| Fortran -O2 original | 42.4 |
| Python 1:1 port | 33.5 (xint part 31.0, xlamb part 2.0) |
| Python fused xint | **8.7** (xint part 6.6) |
-> 3.8× vs the Python port, 4.9× vs Fortran. f bit-identical, gradient |Δ| ≤ 9e-13;
per-event xint value/gradient max rel diff 2e-12 (`tests/test_xint_fast.py`).
After this, xint is still ~75% of func15; the O(N²) xlamb sum is ~25%.

## Parallel comparison: MPI Fortran vs prange Python (2026-11-04)
Hardware: i7-12650H, 10 cores (6 P + 4 E), 16 threads. Same session, one call each.
Fortran: real MS-MPI build (`build/fortran_mpi/`, `mpiexec -n P`), original sources, only
outpmat disabled. Python: `--fast-xint --threads P` (`kernels.*_par`, numba prange, chunk 16).

| P  | Fortran MPI func15 | Python fast func15 | Fortran bkgd/prob | Python bkgd/prob |
|----|-------------------:|-------------------:|------------------:|-----------------:|
| 1  | 34.4 s | 7.2 s | 6.29 / 2.61 | 6.71 / 2.62 |
| 4  | 10.1   | 2.60  | 1.88 / 0.77 | 2.17 / 1.25* |
| 6  | 7.66   | 2.02  | 1.43 / 0.61 | 1.66 / 0.95* |
| 8  | 6.88   | 1.70  | 1.34 / 0.57 | 1.43 / 0.79* |
| 10 | 6.21   | 1.36  | 1.21 / 0.52 | 1.24 / 0.49 |
| 16 | 5.30   | 1.19  | 1.03 / 0.42 | 1.06 / 0.44 |
(* = default chunking, before chunk 16; rows 10/16 use chunk 16.)

- func15: Python ~4.5x faster than MPI Fortran at every P (the xint fusion; both scale ~5.5x
  at 10 cores). bkgdcalc/probcalc: parity (not yet optimised).
- Scaling flattens past 6 (E-cores slower) and past 10 (hyperthreads).
- Python parallel results are bit-identical to serial at any thread count
  (`tests/test_parallel.py`): per-event values go to arrays, summed serially in index order.
  MPI Fortran results change in the last digits with P (reduction order).
- outrates: serial in Fortran (~97 s). Python `outrates_grid_par`: 17.7 s at P=10.
- Full runs at P=10: Fortran MPI ~3500-4075 s, Python fast-xint 1002 s (serial
  outrates) -> ~930 s with parallel outrates.

## Hyperparameters (all knobs in the method, 2026-11-04)
Exposed = settable in `etas8py.Settings` / `Catalog` / `Region` / `ETASParams`.

| What | Value (JMA example) | Where | Exposed |
|---|---|---|---|
| Catalog file, magnitude cutoff m0 | alljpM45.etas, 4.5 | jap.in | via readata |
| Catalog start time, total length T | 0, 23376 days | jap.in | via readata |
| Target window start t_start (end = T) | 10000 days | jap.in | `Region` |
| Study polygon | 9 vertices | jap.in | `Region` |
| Initial ETAS params (mu,A,c,alpha,p,D,q,gamma) | jap.in line | jap.in | `ETASParams` |
| Optimise on/off | 1 | jap.in | `Settings.optimise` |
| Bandwidth neighbour order npp | 5 | jap.in | `Settings.npp` |
| Min bandwidth bwm (deg) | 0.05 | jap.in | `Settings.bwm` |
| Rate grid (mx, my, ranges) | 401x401 | jap.in | `rates(gx, gy)` args |
| Outer/inner schedule | 11 outer, 12-k inner | hardcoded | `Settings.schedule` |
| Polygon pieces per edge NDIV | 1000 | hardcoded | `Settings.ndiv` |
| Optimiser tolerance (tau1,tau2,eps1,eps2) | 1e-6 | davidn | `Settings.tol` |
| Optimiser max sweeps | 10 | davidn | `Settings.max_sweeps` |
| Fused xint / threads | - | new | `Settings.fast_xint/threads` |
| Line search: initial step 0.05, retry 0.1, shrink x0.1, grow x2 | | davidn | no |
| Simpson rule in angle (3 points per piece) | | frint | no |
| frint guards: |det| < 1e-10, r1+r2 <= 1e-20 skip; acos clamp | float32 literals | frint | no |
| log(lambda) floor: lambda <= 1e-25 -> -100 | | func15 | no |
| Magnitude cut epsilon 1e-6 | | readata | no |
| Projection: x = lon cos(lat of polygon centroid), pi = 3.1415926535 | | readata | no |
| Static cap 29000 events (Fortran only) | | common.inc | n/a in Python |
| prange chunk size 16 | | new | no |

Model structure choices (not numbers, but swappable in principle): Gaussian background kernel
with nearest-neighbour bandwidth (self term included); Omori-Utsu time kernel; power-law
spatial kernel scaled by e^{gamma m}; productivity A e^{alpha m}; sqrt parametrisation;
rates evaluated at time T.

## OOP API (`etas8py/declustering.py`, 2026-11-04)
`Catalog`, `Region`, `ETASParams`, `Settings` (frozen dataclass, validated), and
`StochasticDeclustering` with step methods `compute_bandwidths`, `update_background`,
`update_probabilities`, `neg_log_likelihood`, `maximise`, `fit`, `rates`. Thin layer over
`kernels`; `run.Model` is kept as the validated 1:1 driver (CLI, benches) and
`tests/test_declustering.py` asserts the two agree bit for bit.
Deferred: pluggable kernels (Numba makes this a compile-time choice; would need one jitted
function per kernel family), triggering-probability matrix (pmatr) output.

## Clean rerun, all benchmarks (2026-11-04, `scripts/rerun_all.sh`, logs `build/rerun/`)
Nothing else running. Single call each (s); P = MPI ranks / Numba threads.

| P | F bkgd | F prob | F func15 | F outrates | Py bkgd | Py prob | Py func15 fast | Py func15 1:1 | Py outrates |
|---|---|---|---|---|---|---|---|---|---|
| 1 (serial) | 6.43 | 2.63 | 34.7 | 99.2 | 6.97 | 2.60 | 7.31 | 28.6 | 79.4 |
| 4  | 1.91 | 0.78 | 10.3 | 99.7 | 2.02 | 0.77 | 2.14 | | 26.6 |
| 6  | 1.48 | 0.61 | 7.59 | 99.9 | 1.51 | 0.58 | 1.61 | | 21.2 |
| 8  | 1.32 | 0.56 | 6.90 | 99.9 | 1.39 | 0.53 | 1.48 | | 19.4 |
| 10 | 1.22 | 0.50 | 6.24 | 100.0 | 1.26 | 0.49 | 1.37 | 5.06 | 17.1 |
| 16 | 1.03 | 0.42 | 5.31 | 100.0 | 1.06 | 0.43 | 1.20 | 4.57 | 14.7 |

Full runs, 10 cores: Fortran MPI np=10 3965 s; Python run.py fast-xint 10 threads 940 s
(4.2x); OOP API 961 s. Outputs: rerun == previous runs bit for bit (Python and MPI); API ==
run.py byte for byte (probs.dat, rates.dat, final para); Python 1:1 bench == Fortran bench
bit for bit; all 4 tests pass.

## Configuration and options (2026-11-04, `etas8py/config.py`, `geo.py`)
Precedence: preset < original control file (`[input] control`) < TOML < call/CLI args.
`python -m etas8py configs/jma.toml --out DIR [--threads N] [--preset original]
[--set key=value]`; API `StochasticDeclustering.from_config(path, **overrides)`. Outputs:
probs.dat, rates.dat (original formats), params.txt (D also in km^2, -logL, AIC), resolved
config.toml (re-runnable), run.log. Unknown keys are errors.
- Presets: `default` = Settings defaults; `original` = converge off, 1:1 xint, no ndiv
  check, flat projection (bit-exact; `tests/test_config.py`: config/control-file inputs ==
  readata arrays).
- Units: model coordinates always degrees of great-circle arc (1 deg = 111.195 km) for every
  projection, so D (deg^2), bwm, frint guards and optimiser conditioning keep their scale.
  `bwm` and initial `D` must carry units (`deg`/`km`, `deg2`/`km2`).
- Polygon: text file `lon lat` per line or inline list; checks: >= 3 vertices, closure,
  repeats, zero area, self-intersection, lat/lon range, antimeridian (> 180 deg span);
  clockwise rings reversed in lon/lat before projecting (Green's-theorem integral is -1
  for clockwise). Warnings: no events before t_start / outside polygon; error: no targets.
- Projections: `flat` (original) or `laea` (Lambert azimuthal equal-area at the polygon
  centroid). JMA pairs < 1 deg (3.33M), projected/great-circle (`scripts/projection_error.py`):

  | | p1 | median | p99 | min / max | 5-NN bandwidth p1-p99 |
  |---|---|---|---|---|---|
  | flat | 0.943 | 1.000 | 1.063 | 0.890 / 1.108 | 0.922-1.064 |
  | laea | 0.9975 | 1.000 | 1.0019 | 0.993 / 1.008 | 0.996-1.004 |
- Convergence (`converge = true`): outer iterations until `outer_metric` changes by
  < `outer_tol` (max `max_outer`), each with bkgd/prob passes until `inner_metric` changes by
  < `inner_tol` (max `max_inner`). Metrics: `loglik` (relative change of -log L at current
  params/background, one extra func15) or `dphi` (max |change of phi_i|, free). History in
  `sd.history` / run.log. loglik tol < 1e-9 warns (fast-xint rounding ~3e-10).
- ndiv self-check (one-off, first pass): -log L at ndiv vs 2*ndiv; JMA 1.9e-8 (flat),
  4.2e-8 (laea): ndiv=1000 is ample. Warns above `ndiv_check_tol` (1e-6).
- Advanced table: ndiv, fast_xint, optimiser tol/max_sweeps, line-search step/shrink/expand,
  frint guards, log floor/penalty, Numba chunksize (defaults = original constants).

Full JMA runs, 10 threads, default preset (dphi inner 1e-4, loglik outer 1e-8):

| run | outer its | passes | func15 calls | fit | -log L |
|---|---|---|---|---|---|
| original schedule, fast xint (`rerun_fast_t10`) | 11 | 66 | 512 | ~920 s | 15310.951861 |
| default, flat | 6 | 32 | 485 | 838 s | 15310.951879 |
| default, laea | 6 | 31 | 480 | 842 s | 15316.196908 |

- default flat vs original schedule: params <= 1.4e-6 relative, zprob max |d| 8.4e-6
  (mean 3.8e-7), background rates 6e-7. Outer loglik changes 4.9e-4, 5.7e-6, 4.6e-7,
  3.1e-8, 3.1e-9. Inner cap (11) hit in outer 1-2 (dphi still 4.7e-4 / 5.8e-5).
  Saving only ~10%: the late outer iterations it drops were already cheap.
- laea vs flat: params move 0.04-1.5% (gamma -1.5%, A +1.4%, c -1.0%, D +1.0%), zprob max
  |d| 0.088 (mean 2.2e-3), bandwidths up to 12%. -log L not comparable across projections
  (flat densities are per flat deg^2, not true area).
