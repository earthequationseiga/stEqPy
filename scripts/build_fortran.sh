#!/usr/bin/env bash
# Builds the original parallel code single-process (MPI stub) into build/fortran.
# original/ is never touched; patches are applied to copies.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/original/sd_parallel"
B="$ROOT/build/fortran"
ENV_LONG="/c/Users/Herbian Hokky/miniconda3/envs/sdecl"
# 8.3 short path: the space in "Herbian Hokky" breaks ld.
ENV_SHORT="$(cygpath -u "$(cygpath -d "$ENV_LONG")")"
export PATH="$ENV_SHORT/Library/bin:$PATH"

mkdir -p "$B"
cp "$SRC"/*.f "$SRC"/*.inc "$ROOT"/scripts/fortran/* "$B"/
cd "$B"

# conda.specs hardcodes "-L <env path>" unquoted; rewrite it with the short path.
sed "s#C:/Users/Herbian Hokky/miniconda3/envs/sdecl#$(cygpath -m "$ENV_SHORT")#g" \
    "$ENV_SHORT/Library/lib/gcc/x86_64-w64-mingw32/16.2.0/conda.specs" > fixed.specs

# outprob writes on rank 2 only; with one process make rank 0 write it.
# outpmat (rank 1, O(N^2) disk) stays skipped.
sed -i 's/myrank.eq.2/myrank.eq.0/' etas8out.f
# Library = etas8p.f without its main program (lines 1-84).
sed '1,84d' etas8p.f > etas8p_lib.f

FFLAGS="-O2 -std=legacy -w -specs=fixed.specs"
COMMON_SRC="etas8fun.f etas8out.f frint.f poly.f davidn.f mpi_stub.f"
gfortran $FFLAGS -o etas8p.exe etas8p.f $COMMON_SRC
gfortran $FFLAGS -o bench.exe bench_main.f etas8p_lib.f $COMMON_SRC
echo "built: $B/etas8p.exe $B/bench.exe"
gfortran $FFLAGS -o test_davidn.exe test_davidn.f davidn.f mpi_stub.f

# Real MPI build (MS-MPI from conda-forge) into build/fortran_mpi: original sources, no stub.
# Only patch: skip outpmat (rank 1, O(N^2) pmatr.dat). outprob stays on rank 2 as shipped.
M="$ROOT/build/fortran_mpi"
mkdir -p "$M"
cp "$SRC"/*.f "$SRC"/*.inc "$ROOT"/scripts/fortran/bench_main.f "$B"/fixed.specs "$M"/
cd "$M"
sed -i 's/myrank.eq.1/myrank.eq.-1/' etas8p.f
sed '1,84d' etas8p.f > etas8p_lib.f
# MS-MPI's mpif.h uses the Intel INT_PTR_KIND() extension -> preprocess it away.
MFLAGS="$FFLAGS -cpp -DINT_PTR_KIND()=8 -fno-range-check -fallow-invalid-boz -I$ENV_SHORT/Library/include"
MLIBS="-L$ENV_SHORT/Library/lib -lmsmpifec -lmsmpi"
MPI_SRC="etas8fun.f etas8out.f frint.f poly.f davidn.f"
gfortran $MFLAGS -o etas8p.exe etas8p.f $MPI_SRC $MLIBS
gfortran $MFLAGS -o bench.exe bench_main.f etas8p_lib.f $MPI_SRC $MLIBS
echo "built: $M/etas8p.exe $M/bench.exe"
