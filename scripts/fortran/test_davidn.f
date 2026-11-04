C     Runs the original davidn on a chained Rosenbrock function (n=8)
C     and prints the evaluation count and final x, for comparison with
C     the Python port (tests/test_davidn.py).
      program tdav
      implicit real*8 (a-h, o-z)
      external rosen
      real*8 x(50)
      common /cnt/ nev
      nev=0
      do i=1,8
         x(i)=-1.2d0+0.1d0*i
      enddo
      call davidn(x,8,0,rosen)
      write(0,'(i6)') nev
      write(0,'(8es24.16)') (x(i),i=1,8)
      end

      subroutine rosen(n,x,f,g,ig)
      implicit real*8 (a-h, o-z)
      real*8 x(50), g(50)
      common /cnt/ nev
      nev=nev+1
      f=0d0
      do i=1,n
         g(i)=0d0
      enddo
      do i=1,n-1
         t1=x(i+1)-x(i)*x(i)
         t2=1d0-x(i)
         f=f+100d0*t1*t1+t2*t2
         g(i)=g(i)-400d0*x(i)*t1-2d0*t2
         g(i+1)=g(i+1)+200d0*t1
      enddo
      end
