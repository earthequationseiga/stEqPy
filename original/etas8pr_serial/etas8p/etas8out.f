CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C   The subroutine outback outputs the background rate on a lattice of
C      mx*my grids
C
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      subroutine outrates(file1)
      implicit real*8 (a-h,o-z)
      character*80 file1
c      include 'mpif.h'
      include 'common.inc'

      real*8 w(10)
   
      if(myrank.eq.0) then
         open(34,file=file1)
         write(34,*)mx
         write(34,*)my
  
         tx1=1000000.0
         tx2=-10000000.0
         ty1=100000000.0
         ty2=-1000000000
  
         do i=1, npoly
            if(tx1.gt.tx(i))tx1=tx(i)
            if(tx2.lt.tx(i))tx2=tx(i)
            if(ty1.gt.ty(i))ty1=ty(i)
            if(ty2.lt.ty(i))ty2=ty(i)
         enddo
         do i=1,mx
            x0=tx1+(tx2-tx1)/(mx-1)*(i-1)
            do j=1,my
               y0=ty1+(ty2-ty1)/(my-1)*(j-1)
               s=0d0     
               s1=0d0
               do ii=1,nn
                  r0=sqrt((xx(ii)-x0)**2+(yy(ii)-y0)**2)
                  w(1)=zbandw(ii)
                  s=s+zprob(ii)*dgauss(r0,w)
                  s1=s1+dgauss(r0,w)
               enddo
         
               write(34,991)s,s1
             enddo
         enddo
      endif
      return 
 991  format(1x,2f20.10)
      end


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C   The subroutine outprob outputs the probability of each event
C      being a background event or not
C
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

        subroutine outprob(file1)
        implicit real*8 (a-h,o-z)
        character*80 file1
c        include 'mpif.h'
        include 'common.inc'

        
        if(myrank.eq.2)then
             open(35,file=file1)
             do i=1,nn
                write(35,991)i,zprob(i),zbandw(i),zbkgd(i)
             enddo
        endif

   
       return 

 990  format(6(1x,i5,f7.4))

 991  format(1x,i8,3f20.10)
        end





