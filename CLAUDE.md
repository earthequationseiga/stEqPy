# CLAUDE.md — Project Overview

## Purpose
Improve the time and space complexity of Zhuang's stochastic declustering algorithm
(space-time ETAS + variable-kernel background, Zhuang, Ogata & Vere-Jones 2002 JASA) for
statistical seismology. Original code + data: https://bemlar.ism.ac.jp/zhuang/software.html

Workflow: understand the Fortran -> build a reference baseline (same outputs) -> profile ->
replace hot spots with cheaper algorithms -> verify outputs match the baseline within tolerance.

Priority: correctness against the original first, then speed/memory. Simple, clearly-structured,
well-commented code.

## Stack
- Conda env `sdecl`: Python 3.12.16, numpy 2.5.4, numba 0.68.1, gfortran 16.2.0 (conda-forge).
  Run Python with `-I` and `PYTHONPATH=src` (or `sys.path` insert, as the scripts do).
- No MPI. Original is built single-process against `scripts/fortran/mpi_stub.f` by
  `scripts/build_fortran.sh` (into `build/fortran/`). The script works around conda's gcc
  specs breaking on the space in `C:\Users\Herbian Hokky` (8.3 short paths).
- Python port: `src/etas8py/` (`kernels.py` Numba, `davidn.py`, `run.py` driver/CLI).

## Conventions
- Comment the *why*, not the *what* — short plain-language explanations of non-obvious
  concepts (ETAS terms, kernel choices, numerical tricks) are worth keeping in code.
- `original/` is read-only reference: never edit it. New code goes in `src/`, `scripts/`, `tests/`.
- Every speedup must be checked against the original outputs (`probs.dat`, MLE params,
  log-likelihood). Report numerical differences, don't hide them.
- Simplicity first: defer extensions until the current phase works. Deferred items are
  deliberate, not oversights — note them.
- Don't add abstractions or config flexibility beyond what the current phase needs.

## Where to look
- `architecture.md` — how the original Fortran runs, cost model, hot spots, design decisions.
- `status.md` — running progress log; update at the end of every work session.
- `PROMPTS.md` — phase-by-phase execution prompts and their completion status.
