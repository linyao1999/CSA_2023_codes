subroutine nudging()
	
use vars
use params
use microphysics, only: micro_field, index_water_vapor
implicit none

real(8) Tz(nzm), qz(nzm),buffer(nzm), factor_q(nzm), buffer2(nzm), q_clearsky(nzm)
real coef, coef1, factor
integer i,j,k,l

character *120 filename
character *6 filetype
character *5 sepchar
character *10 timechar



call t_startf ('nudging')

tnudge = 0.
qnudge = 0.
unudge = 0.
vnudge = 0.

coef = 1./tauls

if(donudging_uv) then
    do k=1,nzm
      if(z(k).ge.nudging_uv_z1.and.z(k).le.nudging_uv_z2) then
        unudge(k)=unudge(k) - (u0(k)-ug0(k))*coef
        vnudge(k)=vnudge(k) - (v0(k)-vg0(k))*coef
        do j=1,ny
          do i=1,nx
             dudt(i,j,k,na)=dudt(i,j,k,na)-(u0(k)-ug0(k))*coef
             dvdt(i,j,k,na)=dvdt(i,j,k,na)-(v0(k)-vg0(k))*coef
          end do
        end do
      end if
    end do
endif

coef = 1./tautqls

if(donudging_tq.or.donudging_t) then
    coef1 = dtn / tautqls
    do k=1,nzm
      if(z(k).ge.nudging_t_z1.and.z(k).le.nudging_t_z2) then
        tnudge(k)=tnudge(k) -(t0(k)-tg0(k)-gamaz(k))*coef
        do j=1,ny
          do i=1,nx
             t(i,j,k)=t(i,j,k)-(t0(k)-tg0(k)-gamaz(k))*coef1
          end do
        end do
      end if
    end do
endif

if(donudging_tq.or.donudging_q) then
    coef1 = dtn / tautqls
    do k=1,nzm
      if(z(k).ge.nudging_q_z1.and.z(k).le.nudging_q_z2) then
        qnudge(k)=qnudge(k) -(q0(k)-qg0(k))*coef
        do j=1,ny
          do i=1,nx
             micro_field(i,j,k,index_water_vapor)=micro_field(i,j,k,index_water_vapor)-(q0(k)-qg0(k))*coef1
          end do
        end do
      end if
    end do
endif

if(donudging_q_clearsky) then    !change to donudging_q_clearsky
   factor_q = 0.0 ! number to divide by (number of grid points that are unsaturated):
   do k=1,nzm
      if(z(k).ge.nudging_q_z1.and.z(k).le.nudging_q_z2) then
         q_clearsky(k) = 0.
         do j=1,ny
            do i=1,nx
               if ((qcl(i,j,k) + qci(i,j,k) + qpl(i,j,k) + qpi(i,j,k)).le.1e-5) then
                  factor_q(k) = factor_q(k) + 1
                  q_clearsky(k) = q_clearsky(k) + micro_field(i,j,k,index_water_vapor)
               end if
            end do
         end do
      end if
   end do
   
   
   if(dompi) call task_sum_real8(q_clearsky,buffer,nzm)
   if(dompi) call task_sum_real8(factor_q,buffer2,nzm)
   do k=1,nzm
      if(z(k).ge.nudging_q_z1.and.z(k).le.nudging_q_z2) then
         if (buffer2(k).ne.0.0) then
            q_clearsky(k)=buffer(k)/buffer2(k) !this has now the mean !check check tomorrow what if the condition is not met
         end if
!! Writing
!         if((mod(nstep,nsave2D).eq.0).and.(masterproc))  then  
!            sepchar=""
!            write(timechar,'(i10)') nstep
!            do l=1,11-lenstr(timechar)-1
!               timechar(l:l)='0'
!            end do
!            filetype=".txt"            
!            filename='./q_clearsky/'//trim(case)//'_'//trim(caseid)//'_'//timechar(1:10)//filetype//sepchar
!            open(46,file=filename,status='unknown',form='formatted',position='append') 
!            write(46,'(E16.5)') q_clearsky(k)*dtn/tauq_clearsky 
!            close(46)
!         end if
!Output         
         do j=1,ny
            do i=1,nx
               if ((qcl(i,j,k) + qci(i,j,k) + qpl(i,j,k) + qpi(i,j,k)).le.1e-5) then
                  micro_field(i,j,k,index_water_vapor) = micro_field(i,j,k,index_water_vapor) + (q_clearsky(k) - micro_field(i,j,k,index_water_vapor))*dtn/tauq_clearsky
               end if
            end do
         end do
      end if
   end do   

end if


if(do_T_homo) then
        factor = 1./dble(nx*ny)
        do k=1,nzm
           if(z(k).ge.homo_Tq_z1.and.z(k).le.homo_Tq_z2) then
              Tz(k) = 0.
              do j=1,ny
                 do i=1,nx
                    Tz(k) = Tz(k) + t(i,j,k)
                 end do
              end do
              Tz(k) = Tz(k) * factor
              buffer(k) = Tz(k)
           end if
        end do

        factor = 1./float(nsubdomains)
        if(dompi) call task_sum_real8(Tz,buffer,nzm)
        do k=1,nzm
           if(z(k).ge.homo_Tq_z1.and.z(k).le.homo_Tq_z2) then
           Tz(k)=buffer(k)*factor
           do j=1,ny
             do i=1,nx
               t(i,j,k) = Tz(k)
             end do
           end do
           end if
        end do
end if

if(do_q_homo) then
        factor = 1./dble(nx*ny)
        coef1 = dtn / tautqls
        do k=1,nzm
           if(k.ge.homo_Tq_z1.and.k.le.homo_Tq_z2) then
           qz(k) = 0.
           do j=1,ny
             do i=1,nx
                qz(k) = qz(k) + micro_field(i,j,k,index_water_vapor)
             end do
           end do
           qz(k) = qz(k) * factor
           buffer(k) = qz(k)
           end if
        end do

        factor = 1./float(nsubdomains)
        if(dompi) call task_sum_real8(qz,buffer,nzm)

        do k=1,nzm
           if(k.ge.homo_Tq_z1.and.k.le.homo_Tq_z2) then
           qz(k)=buffer(k)*factor
           do j=1,ny
             do i=1,nx
                micro_field(i,j,k,index_water_vapor) = micro_field(i,j,k,index_water_vapor) - (micro_field(i,j,k,index_water_vapor)-qz(k))*coef1
             end do
           end do
           end if
        end do
end if

call t_stopf('nudging')

end subroutine nudging
