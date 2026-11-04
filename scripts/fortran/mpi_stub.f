C     Minimal single-process MPI stub so the original code builds
C     without an MPI install: rank 0 of 1, bcast is a no-op, reduce
C     copies (the original only reduces scalar doubles).
      subroutine mpi_init(ierr)
      ierr=0
      end
      subroutine mpi_finalize(ierr)
      ierr=0
      end
      subroutine mpi_comm_size(icomm, nprocs, ierr)
      nprocs=1
      ierr=0
      end
      subroutine mpi_comm_rank(icomm, myrank, ierr)
      myrank=0
      ierr=0
      end
      subroutine mpi_bcast(buf, n, itype, iroot, icomm, ierr)
      ierr=0
      end
      subroutine mpi_reduce(sbuf, rbuf, n, itype, iop, iroot, icomm,
     &                      ierr)
      real*8 sbuf(*), rbuf(*)
      do i=1,n
         rbuf(i)=sbuf(i)
      enddo
      ierr=0
      end
