C     Single-process stand-in for the real mpif.h (see mpi_stub.f).
      integer mpi_comm_world, mpi_double_precision, mpi_integer
      integer mpi_sum
      parameter (mpi_comm_world=0, mpi_double_precision=1)
      parameter (mpi_integer=2, mpi_sum=3)
