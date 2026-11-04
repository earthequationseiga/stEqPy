#!/usr/bin/env bash
# Reruns tests, benchmarks and the 10-core full runs back to back (nothing else should use
# the CPU meanwhile). Logs in build/rerun/. Skips the 6 h serial Fortran and 5 h Python 1:1
# full runs (code paths unchanged; bench checks they still match bit for bit).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="/c/Users/Herbian Hokky/miniconda3/envs/sdecl/python.exe"
E="$(cygpath -u "$(cygpath -d "/c/Users/Herbian Hokky/miniconda3/envs/sdecl")")"
export PATH="$E/Library/bin:$PATH"
L="$ROOT/build/rerun"
mkdir -p "$L"
stamp() { echo "== $1 $(date +%H:%M:%S)" | tee -a "$L/progress.txt"; }

stamp "build"
bash scripts/build_fortran.sh > "$L/build.txt" 2>&1

stamp "tests"
for t in test_davidn test_xint_fast test_parallel test_declustering; do
    s=$(date +%s)
    "$PY" -I tests/$t.py > "$L/$t.txt" 2>&1
    echo "$t exit $? $(( $(date +%s) - s )) s" | tee -a "$L/progress.txt"
done

stamp "fortran bench sweep"
B="$L/fortran_bench"; mkdir -p "$B"; cp data/jma/* "$B"/
(cd "$B" && echo "== stub" && "$ROOT/build/fortran/bench.exe" < jap.in 2>&1 >/dev/null) \
    > "$L/fortran_sweep.txt" 2>&1
cp "$B/rates.dat" "$L/fortran_stub_rates.dat"
for n in 1 4 6 8 10 12 16; do
    echo "== np $n" >> "$L/fortran_sweep.txt"
    (cd "$B" && mpiexec -n $n "$ROOT/build/fortran_mpi/bench.exe" < jap.in 2>>"$L/fortran_sweep.txt" >/dev/null)
done

stamp "python bench (1:1 serial, same values as fortran bench)"
"$PY" -I scripts/bench_python.py data/jma/jap.in > "$L/python_bench.txt" 2>&1

stamp "python thread sweep"
"$PY" -I scripts/bench_threads.py data/jma/jap.in 0 1 4 6 8 10 12 16 > "$L/python_sweep_fast.txt" 2>&1
"$PY" -I scripts/bench_threads.py data/jma/jap.in --slow-xint 0 10 16 > "$L/python_sweep_slow.txt" 2>&1

stamp "full: run.py fast-xint 10 threads"
s=$(date +%s)
"$PY" -I -c "import sys; sys.path.insert(0, 'src'); from etas8py.run import main; main()" \
    data/jma/jap.in --out results/rerun_fast_t10 --fast-xint --threads 10 > "$L/full_run_py.txt" 2>&1
echo "exit $? wall $(( $(date +%s) - s )) s" | tee -a "$L/full_run_py.txt" "$L/progress.txt"

stamp "full: OOP API 10 threads"
s=$(date +%s)
"$PY" -I scripts/run_api.py data/jma/jap.in results/rerun_api_t10 10 > "$L/full_api.txt" 2>&1
echo "exit $? wall $(( $(date +%s) - s )) s" | tee -a "$L/full_api.txt" "$L/progress.txt"

stamp "full: fortran MPI np 10"
M="$ROOT/build/fortran_mpi/rerun_np10"; mkdir -p "$M"; cp data/jma/* "$M"/
s=$(date +%s)
(cd "$M" && mpiexec -n 10 ../etas8p.exe < jap.in > jap.out 2> stderr.txt)
echo "exit $? wall $(( $(date +%s) - s )) s" | tee "$M/wall.txt" -a "$L/progress.txt"

stamp "done"
