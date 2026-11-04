C     Times one call of each expensive routine of the original code
C     (single process). Timings + likelihood go to stderr (unit 0) so
C     the per-event probability dump on stdout can be discarded.
      program bench
      implicit real*8 (a-h, o-z)
      include 'mpif.h'
      include 'common.inc'
      character *80 fn
      real*8 h(8)
      integer*8 c0, c1, ir

      call mpi_init(ierr)
      call mpi_comm_size(mpi_comm_world, nprocs, ierr)
      call mpi_comm_rank(mpi_comm_world, myrank, ierr)

      call system_clock(c0, ir)
      call readata(npp,bwm)
      call para(nn)
      call system_clock(c1)
      if(myrank.eq.0)write(0,900) 'readata', dble(c1-c0)/ir

      call system_clock(c0)
      call bandwcalc(npp, bwm)
      call system_clock(c1)
      if(myrank.eq.0)write(0,900) 'bandwcalc', dble(c1-c0)/ir

      call system_clock(c0)
      call bkgdcalc()
      call system_clock(c1)
      if(myrank.eq.0)write(0,900) 'bkgdcalc', dble(c1-c0)/ir

      call system_clock(c0)
      call probcalc()
      call system_clock(c1)
      if(myrank.eq.0)write(0,900) 'probcalc', dble(c1-c0)/ir

      call system_clock(c0)
      call func15(n,x,f,h,0)
      call system_clock(c1)
      if(myrank.eq.0)write(0,900) 'func15', dble(c1-c0)/ir
      if(myrank.eq.0)write(0,910) f, (h(i),i=1,8)
      if(myrank.eq.0)write(0,910) xint0, zbkgd(1), zbkgd(nn),
     &  zprob(2), zprob(nn)

      call system_clock(c0)
      fn='rates.dat'
      call outrates(fn)
      call system_clock(c1)
      if(myrank.eq.0)write(0,900) 'outrates', dble(c1-c0)/ir

      call mpi_finalize(ierr)
 900  format(1x, a12, f12.3, ' s')
 910  format(1x, 9es24.16)
      end
