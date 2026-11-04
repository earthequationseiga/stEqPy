C      file: etas8.f
C      associated files: etas5fun.f davidn.f frint.f
       programme etas8
c-----------------------------------------------
c     space-time version of the ETAS model
c     with parallel computing
c------------------------------------------------
      implicit real*8 (a-h, o-z)
c      include 'mpif.h'
      include 'common.inc'
      real*8 buffer(100),h(8),x1(8)
 
c      call mpi_init(ierr)
c     call mpi_comm_size(mpi_comm_world, nprocs=1, ierr)
      nprocs=1
c      call mpi_comm_rank(mpi_comm_world, myrank=0, ierr)
      myrank=0
C      print *,  'hello from task', myrank, 'of ', nprocs

      call readata(npp,bwm)

      call para(nn)
      call bandwcalc(npp, bwm)
  
      do i=1,4
         call bkgdcalc() 
         call probcalc()
      enddo

      do iterative=1, 10

         do i=1, 3
           call bkgdcalc() 
           call probcalc()
         enddo
         
         call dav()
 
      enddo
      
      call outrates('rates.dat')
      call outprob('probs.dat')
      call outpmat('pmat.dat')

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C
C   TEST
C
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c       do i=1, 8
c        x(i)=b(i)
c       enddo


c       write(*,*)"x=", (x(j), j=1,8)
c       delta=1d-8
c       call func15(n,x,f,h,0)
c       do i=1,8
c           do j=1,8
c                x1(j)=x(j)
c           enddo
c           x1(i)=x(i)+delta
c           call func15(n,x1,f1,h,0)
c           if(myrank.eq.0)write(*,*) i,f,f1,h(i),(f1-f)/delta
c       enddo

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

c      call mpi_finalize(ierr)
 991   format(1x, 8f19.12)
 994   format(1x,i5, ' fv= ', e12.4 , ' h()= ',e12.4, e12.4)
      end
   
c********************************************************************
      subroutine para(nk)
      include 'common.inc'

C      write(*,*)'para nn', nk,myrank
      ista=1
      iend=nn
      ista2=1
      iend2=nn     
  
c      print*,'in process',myrank,'data:', ista, iend, 'of ',nk
c      print*,'in process',myrank,'data:', ista2, iend2, 'of ',nk

      return
      end


c**********************************************************************
      subroutine readata(npp,bwm)
 
      implicit real*8 (a-h, o-z)
c      include 'mpif.h'
      include 'common.inc'


      character *80 hypodata, fmt, splcoef
      character *160 ftl

C----------------------------------------------------------------
C    Input data from the root process
C----------------------------------------------------------------

      if(myrank.eq.0)then
         write(*,*)'Please input the name of the data file:'
         read(*,*)hypodata
         write(*,*)hypodata
 
         write(*,*)'Please input data format:'
         read(*,*)fmt
         write(*,*)'Input the length of the time intervals:'
         read(*,*)tz
         write(*,*)'Input number of events:'
         read(*,*)nn
         write(*,*)'Input starting points of time:'
         read(*,*)  zmin
         write(*,*)'Input the magnitude threshold and tstart:'
         read(*,*) xmg0, tstart
         write(*,*)'Input the number of points for the polygon region:'
          read(*,*) npoly
 
          do i=1, npoly
            read(*,*)tx(i), ty(i)
          enddo
          tx(npoly+1)=tx(1)
          ty(npoly+1)=ty(1)
    
          ymean=ycentroid(tx,ty, npoly)
  
          do i=1, npoly+1        
               tx(i)=(tx(i))*cos((ymean)*3.1415926535d0/180d0)
          enddo

         write(*,*)'Input number of grids for background'
         read(*,*)mx,my


         open(11, file=hypodata)
         read(11,*)ftl
         write(*,997)ftl    

      
         i=1
 15      read(11,*,end=25)itemp,xx(i),yy(i),zmg(i),zz(i),dd,ee,ff
           if(zz(i).gt.tz+zmin)goto 25
	     if(i.gt.1) then
              if(zz(i).lt.zz(i-1))then
                  write(*,*)'Reverse data:', zz(i-1), zz(i)
                  stop
              endif
	     endif
            xx(i)=(xx(i))*cos((ymean)*3.1415926535d0/180d0)
            if(zmg(i).gt.xmg0-0.000001)i=i+1
            goto 15
 25      continue
         nn=i-1

         nnc=0

         call polyse(tx,ty, npoly, xx, yy, nn, ind)

         do 10 i=1,nn
            if(zz(i).gt.tz+zmin) ind(i)=0 
            if(zz(i).lt.   tstart) ind(i)=0

            zmg(i)=zmg(i)-xmg0
            zz(i)=zz(i)-zmin
            zprob(i)=1.0d0

           if(ind(i).eq.1)then
             nnc=nnc+1
c             print*, i,xx(i),yy(i),zz(i),zmg(i),ind(i),zprob(i)
           endif
   10   continue          

        tstart=tstart-zmin
       


      endif

    
c---------------------------------------------------------------
c
c     Broadcast all the data to other process from process 0
c
c---------------------------------------------------------------


c-------------------------------------------------------
c    end of sending data
c-------------------------------------------------------

C-------------------------------------------------------
c    Read and send model parameters
C------------------------------------------------------
       n=8
       if(myrank.eq.0)then 
            write(*,*)'Please input initial parameters:'
            write(*,*)'(mu, A, c, alfa, p, d, q, gamma)'  
             read(*,*) (b(i),i=1,n)
            write(6,1020) n
            write(6,1030)  (b(i),i=1,n)
            write(*,*)'Please input n_p and minimum bandwidth:'
            read(*,*)npp,bwm
       endif

C       write(*,*)'myrank',myrank,(b(i),i=1,n)


      do i=1,n
         x(i)=sqrt(b(i))
      enddo


 991  format(1x,i6,4f12.4,4f5.2)
 993  format(1x,'mx, my= ',i7,i7)
 995  format('tx,ty,tz,xmin,ymin,xmg0,zmin,tsta',/8f10.3,
     &/' nn =',i6,' nnc=',i6)
 997  format(1x, 'Data set  ',10a8)
 1020 format(1x,3x,'input data'/1h ,5x,'n=',i3,3x,'itr=',i5,3x,
     2          'ier=',i3,3x,'eps=',1pe16.7)
 1030 format(1x,'x=',5e14.5)
 1040 format(1x, /' mle = ',5e12.5/('       ',9e12.5))
 1110 format(1h , 8f20.9)
      end


c*********************************************
c   
       subroutine dav()

       implicit real * 8 (a-h,o-z)
c       include 'mpif.h'
       include 'common.inc'
       external func15
 
       do 70 i=1,n
   70      x(i)=sqrt(b(i))

         call davidn(x,n,0,func15)
     
       do 80 i=1,n
         b(i)=x(i)**2
 80    continue

      if(myrank.eq.0)then 
    
         write(6,1040) (b(i),i=1,n)
         open(22,file='para')
         write(22,1110)(b(i),i=1,n)
      endif
      
c      write(*,*)'endof dav()'

      return
 991  format(1x,f16.8)
 999  format(1x, 8f12.8)
 1020 format(1x,3x,'input data'/1h ,5x,'n=',i3,3x,'itr=',i5,3x,
     2          'ier=',i3,3x,'eps=',1pe16.7)
 1030 format(1x,'x=',5e14.5)
 1040 format(1x, /' mle = ',5e12.5/('       ',9e12.5))
 1110 format(1h , 8f20.9)
      end
c***********************************************************************
 

c***********************************************************************
      subroutine func15(n,b,f,h,ifg)
c-----------------------------------------------------------------------
c     likelihood function of modified oomori type
c     etas point process (with trend: not realized).
c     the space-time <<anisotropic>> version.
c \lambda(t,x,y)=\mu+\sum k/(t-t_j+c)^p
c *\exp{ ( (x-x_j)^2+(y-y_j)^2 )/2/\exp{2\alpham_j} }
c-----------------------------------------------------------------------
      implicit real * 8 (a-h,o-z)

c      include 'mpif.h'
      include 'common1.inc'

      dimension r(31,31)
      dimension b(50),h(50),ggt(50),gt(50),at(50), gtemp(50)
      dimension rnx(4),rny(4)
      real*8 gg(8),gg0(8)
 

C      if(b(2).ge.1)b(2)=1    

c------------------Parallel part-----------------------------------      

      ff=0d0
      df1=0d0
      df2=0d0
      df3=0d0
      df4=0d0
      df5=0d0
      df6=0d0
      df7=0d0
      df8=0d0
 
      ftemp=0d0
      df1temp=0d0
      df2temp=0d0
      df3temp=0d0
      df4temp=0d0
      df5temp=0d0
      df6temp=0d0
      df7temp=0d0
      df8temp=0d0

      do i=ista,iend
       if(ind(i).eq.1)then
        call xlamb(i,b, temp, gtemp)
        if(temp.gt.1d-25)then 
             ftemp=ftemp+log(temp)
         else 
             ftemp=ftemp-100
        endif
        df1temp=df1temp+gtemp(1)/temp 
        df2temp=df2temp+gtemp(2)/temp 
        df3temp=df3temp+gtemp(3)/temp 
        df4temp=df4temp+gtemp(4)/temp 
        df5temp=df5temp+gtemp(5)/temp 
        df6temp=df6temp+gtemp(6)/temp 
        df7temp=df7temp+gtemp(7)/temp 
        df8temp=df8temp+gtemp(8)/temp 
       endif
      enddo

c        write(*,*)'in task', myrank, ftemp
c        write(*,*)'in task', myrank, 'd1ftemp=',df1temp
c        write(*,*)'in task', myrank, 'd2ftemp=',df2temp
c        write(*,*)'in task', myrank, 'd3ftemp=',df3temp
c        write(*,*)'in task', myrank, 'd4ftemp=',df4temp
c        write(*,*)'in task', myrank, 'd5ftemp=',df5temp
c        write(*,*)'in task', myrank, 'd6ftemp=',df6temp
c        write(*,*)'in task', myrank, 'd7ftemp=',df7temp

      ff=ftemp
      df1=df1temp
      df2=df2temp
      df3=df3temp
      df4=df4temp
      df5=df5temp
      df6=df6temp
      df7=df7temp
      df8=df8temp

       
c       write(*,*)ff,df1,df2,df3,df4,df5,df6,df7     

       do i=1,n
          gg0(i)=0d0
          gg(i)=0d0
       enddo

       fv0temp=0       


       do i=ista2,iend2
       call xint(i,b,fvtemp,gg)

           do j=1,n
              gg0(j)=gg0(j)+gg(j)
           enddo
c           write(*,*)'in task', myrank, i,'gg3=',gg0(3)
           fv0temp=fv0temp+fvtemp 
C      write(*,*)'in task', myrank, 'fv0=',fvtemp, tz

       enddo

c     write(*,*)'in task', myrank, 'gg1=',gg0(1)
c     write(*,*)'in task', myrank, 'gg2=',gg0(2)
c     write(*,*)'in task', myrank, 'gg3=',gg0(3)
c     write(*,*)'in task', myrank, 'gg4=',gg0(4)
c     write(*,*)'in task', myrank, 'gg5=',gg0(5)
c     write(*,*)'in task', myrank, 'gg6=',gg0(6)
c     write(*,*)'in task', myrank, 'gg7=',gg0(7)
             

	fv0=fv0temp
	h1=gg0(1)
	h2=gg0(2)
	h3=gg0(3)
	h4=gg0(4)
	h5=gg0(5)
	h6=gg0(6)
	h7=gg0(7)
	h8=gg0(8)

      
c      if(myrank.eq.0) write(*,*)myrank, fv0,h1,h2,h3,h4,h5,h6,h7,h8

       if(myrank.eq.0)then 
            fv0=fv0+xint0*b(1)*b(1)
            h1=xint0*b(1)*2
       endif
c      write(*,*)'in task', myrank, 'ff1=',ff1

c       write(*,*)'after bcast in task', myrank, 'ff1=',ff1


        f=-ff+fv0
        h(1)=-df1+h1
        h(2)=-df2+h2
        h(3)=-df3+h3
        h(4)=-df4+h4
        h(5)=-df5+h5
        h(6)=-df6+h6
        h(7)=-df7+h7
        h(8)=-df8+h8
    
        if(myrank.eq.0)write(*,991)f,(h(i),i=1,8)
c        if(myrank.eq.0)write(*,993)ff,df1,df2,df3,df4,df5,df6,df7,df8
c        if(myrank.eq.0)write(*,993)fv0,h1,h2,h3,h4,h5,h6,h7,h8
        if(myrank.eq.0)write(*,992)(b(i),i=1,8)
c-------------------end of the parallel part ---------------------------

   
      return
 991  format(1x,'function value=', f12.4,/ ' Gradient= ',/8f12.4)
 992  format(1x,'at b=',/8f9.4/)
 993  format(1x,8f12.4)
 994  format(1x,i8,f19.7)
      end
 


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C
C   This subroutine calculates the variable bandwith for each event
C         given np and minimum bandwidth xlband
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
       subroutine bandwcalc(np, xlband)

      implicit real*8 (a-h, o-z)
c     include 'mpif.h'
      include 'common.inc'

      real*8 dbuffer(100)

      do i=ista2, iend2
        do kkk=1,np
           dbuffer(kkk)=10d39 
C           write(*,*)dbuffer(i)
        enddo
      
     
         do 5 j=1,nn
             if(i.ne.j)then
                temp=sqrt((xx(i)-xx(j))**2+(yy(i)-yy(j))**2)
c                write(*,*)i,j,temp
                 do 10 k=1,np
                   if(dbuffer(k).gt.temp)then
                       do kk=np,k+1,-1
                             dbuffer(kk)=dbuffer(kk-1)
                       enddo
c                         write(*,*)'1', k,temp,dbuffer(k)
                       dbuffer(k)=temp
c                         write(*,*)k,temp,dbuffer(k)
                       goto 15
                   endif
 
 10             continue
             endif
 15          continue
C          write(*,*)(dbuffer(ii),ii=1,6)
 5       enddo
       
c       write(*,*)'i',myrank, i,nn, dbuffer(np), ista2, iend2
       zbandw(i)=dbuffer(np)
C       write(*,*)zbandw(i),xlband
       if(zbandw(i).lt.xlband)zbandw(i)=xlband
      enddo             
      
      	write(*,*) 'finishing bandwidth calculation'
 
      return
      end

C--------------------------------------------------------------------------
C
C  This subroutine calculates the thinning probabilities
C     zprob(j)=the probability of j-th event being an independent event
C
C--------------------------------------------------------------------------
      subroutine probcalc()
      implicit real*8 (a-h, o-z)
c      include 'mpif.h'
      include 'common.inc'

      real gtemp(50)
      

	write(*,*)'starting prob calculation'
      do i=ista,iend
        bk=zbkgd(i)*x(1)*x(1)
        call xlamb(i,x, b2, gtemp)
        zprob(i)=bk/b2
      enddo
      do i=0, nprocs-1
        if(i.eq.0)is1=1
        if(i.ne.0)is1=sqrt(float(nn)*float(nn)*
     &              float(i)/float(nprocs))+1 
        if(i.ne.nprocs-1)ie1=sqrt(float(nn)*float(nn)*
     &                float(i+1)/float(nprocs))
        if(i.eq.nprocs-1)ie1=nn
c        call mpi_bcast(zprob(is1),ie1-is1+1,mpi_double_precision,i,
c     &      mpi_comm_world, ierr)
      enddo 
    

      if(myrank.eq.0)then
c           write(*,*)'Probabilities to be independent events:'
c           write(*,990)(i,zprob(i),i=1,nn)
      endif
	write(*,*)'finishing background calculation'
     
 990  format(8(1x,i5,f7.4))
      end

C--------------------------------------------------------------------------
C
C  This subroutine calculates the thinning probabilities
C     for the probability matrix
C  output as (i, j, \rho_ij)
C--------------------------------------------------------------------------
      subroutine outpmat(fp)
      implicit real*8 (a-h, o-z)
c      include 'mpif.h'
      include 'common.inc'

      character*8 fp


      real*8 x1(10), gtemp(50)


      if(myrank.eq.1) then
          open(39, file=fp)
      
          xmu=x(1)**2
          a2=x(2)**2
          c=x(3)**2
          alfa=x(4)**2
          p=x(5)**2
          d=x(6)**2
          q=x(7)**2
          gamma=x(8)**2
      
         do i=1,nn
            bk=zbkgd(i)*xmu
            call xlamb(i,x, b2, gtemp)
           
	      if(i.gt.1)then
            do j=1, i-1
               delt=zz(i)-zz(j)

               pr1=exp(alfa*zmg(j))
               ssig=d*exp(gamma*zmg(j))
               bbb=(q-1)/pi/ssig      
               dist2=(xx(i)-xx(j))**2+(yy(i)-yy(j))**2
               pr2=(p-1)/c*(1d0+delt/c)**(-p)
               pr3=bbb*(dist2/ssig+1d0)**(-q)         
  
               s=a2*pr1*pr2*pr3

               temprob=s/b2

               if(temprob.gt.1e-20) write(39,995)i,j,temprob

            enddo
            endif
            write(39,995)i,i,bk/b2

        enddo
        close (39)
        endif
 995    format(1x,i8,i8, f20.16)
      end







