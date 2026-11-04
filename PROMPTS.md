# PROMPTS.md — Execution Prompt Sequence

One prompt per work session. Each opens by reading `CLAUDE.md`, `architecture.md`, `status.md`,
and closes by updating `status.md`.

- [x] **Prompt 1 — Scaffolding + read the original Fortran** (done 2026-11-03)
  > Download Zhuang's code/data, set up doc scaffolding, and write up how the Fortran runs
  > (flow, model, cost per routine) in `architecture.md`.

- [x] **Prompt 2 — Toolchain + first-pass Python port + speed comparison** (done 2026-11-03;
  replaced the original "reproduce with MPI" prompt after the user chose a Python port)
  > Port the Fortran to Python (keep speed/memory roughly equal), run on the sample data,
  > compare speeds against the Fortran.

- [x] **Prompt 3 — Full-run validation + pick first hot spot** (done 2026-11-04: port bit-identical to Fortran over full run; fused xint 4.0x end to end)
  > Read `CLAUDE.md`, `architecture.md`, `status.md`. Smoke-test
  > `python -m etas8py.run data/jma/jap.in --schedule short`. Then run the full schedule for
  > both Fortran (`build/fortran/etas8p.exe < jap.in`) and Python (~12 h each, in the
  > background) and compare `probs.dat`, `para`, final `-log L` with each other and with the
  > shipped outputs. Then choose the first optimisation (likely `xint` polygon integrals).
