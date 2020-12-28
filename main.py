from euler_1d_weno import *
import scipy.linalg as la 
import numpy as np
import sys,os
import pandas as pd
import math


#Select run mode and initial / boundary conditions
Run_Mode_Options = ['WENO-JS','WENO-Z']
runMode     = Run_Mode_Options[1]
Initial_Condition = ['SOD', 'LAX', '123', 'Shockdensity', 'blastwave']
init = Initial_Condition[1]
BC_Options = ['Non-Reflecting','Neumann','Wall','Force-Free']
left_bc  = BC_Options[0]; right_bc = BC_Options[0]
Adv_Options = ['WENO','LINEAR-FD']
Advection   = Adv_Options[0]

#useEndtime : True if you want to set end time of simulation. Nt will be set 100,000 automatically. If not(False), you must set Nt manually 
useEndtime = True


# Specify the number of points in the domain (includes ghost points)
Nx = 500

# Specifiy target CFL and total number of steps
CFL = 0.4 
Nt = 1000

#set the time to end program
#in the paper; SOD:2s / Lax:1.3s / 123:1.0s / shock-density:2s / blastwaves:0.038s
target_time=1.5
if (useEndtime):
    Nt = 100000

os.system('clear')

#make result path
result_path = 'result_csv'
result_file = result_path+'/'+"result_"+runMode+"_"+init+'_'+str(Nx)+".csv"
if not os.path.exists('result_csv'):
    os.makedirs('result_csv')    

# Specify the overall domain size

if init=="blastwave":
    X_min,X_max = 0.00,1.00
else : 
    X_min,X_max = -5.00,5.00

#initialize flux
q_init,X,dx,P_51 = init_cond(X_min,X_max,Nx,init)

#allocate arrays for computing and storing the solution
q = np.copy(q_init)
Q = np.zeros((q.shape[0]-6,q.shape[1],Nt+1))
state_variable_of_interest = np.zeros((q.shape[0]-6,Nt+1)) 
Q[:,:,0] = q_init[3:-3,:]
q1,q2 = np.zeros(q.shape),np.zeros(q.shape)

#perform the time integration
t_vec = np.zeros(Nt+1)

#show info on terminal
print('\n=====================================================')
print('   Selected advection scheme is: %s' % Advection)
print('   Selected initial condition is: %s' % init)
print('   Selected problem definition is: %s' % runMode)
print('   Left-end boundary condition is: %s' % left_bc)
print('   Right-end boundary condition is: %s' % right_bc)
print('=====================================================\n')
if (useEndtime):
    print('Performing time integration for Endtime t = %2.3f total steps...\n' % target_time)
else : 
    print('Performing time integration with Nt = %d total steps...\n' % Nt)

#make new empty list to save result as csv
result_q = []    
time_list=[]
i=1
time=0



## 'for' iteration method when Nt was fixed. 
## Instead of using this method,
## enumerate Nt was set and 'target_time' used for 'while' method


#iteration start
if (useEndtime):    
    while True :
        if time < target_time :

            ws_max = np.max(euler_1d_wavespeed(q))
            dt = CFL*dx/ws_max
            #update the time history
            t_vec[i] = t_vec[i-1] + dt
            #display to terminal
            print('%s : n = %d,  CFL = %1.2f,  dt = %1.2es,  t = %1.2es' % (runMode,i,CFL,dt,t_vec[i]))
            q = update_ghost_pts(q,left_bc,right_bc)
            L0 = spatial_rhs(q,dx,Advection,left_bc,right_bc,runMode) 
            q1[3:-3,:] = q[3:-3,:] + L0*dt
            q1 = update_ghost_pts(q1,left_bc,right_bc)
            L1 = spatial_rhs(q1,dx,Advection,left_bc,right_bc,runMode) 
            q2[3:-3,:] = (3/4)*q[3:-3,:] + (1/4)*q1[3:-3,:] + (1/4)*L1*dt
            q2 = update_ghost_pts(q2,left_bc,right_bc)
            L2 = spatial_rhs(q2,dx,Advection,left_bc,right_bc,runMode) 
            q[3:-3,:] = (1/3)*q[3:-3,:] + (2/3)*q2[3:-3,:]  + (2/3)*L2*dt    
            #update the stored history
            Q[:,:,i] = q[3:-3,:]

            time = t_vec[i]
            time_list.append(time)
            data = q[3:-3,0]
            data = data.tolist()
            result_q.append(data)
            i+=1
        else : 
            break
        
else : 
    for i in range(1,Nt+1):
        ws_max = np.max(euler_1d_wavespeed(q))
        dt = CFL*dx/ws_max
        #update the time history
        t_vec[i] = t_vec[i-1] + dt
        #display to terminal
        print('%s : n = %d,  CFL = %1.2f,  dt = %1.2es,  t = %1.2es' % (runMode,i,CFL,dt,t_vec[i]))
        q = update_ghost_pts(q,left_bc,right_bc)
        L0 = spatial_rhs(q,dx,Advection,left_bc,right_bc,runMode)  
        q1[3:-3,:] = q[3:-3,:] + L0*dt
        q1 = update_ghost_pts(q1,left_bc,right_bc)
        L1 = spatial_rhs(q1,dx,Advection,left_bc,right_bc,runMode) 
        q2[3:-3,:] = (3/4)*q[3:-3,:] + (1/4)*q1[3:-3,:] + (1/4)*L1*dt
        q2 = update_ghost_pts(q2,left_bc,right_bc)
        L2 = spatial_rhs(q2,dx,Advection,left_bc,right_bc,runMode) 
        q[3:-3,:] = (1/3)*q[3:-3,:] + (2/3)*q2[3:-3,:]  + (2/3)*L2*dt    
        #update the stored history
        Q[:,:,i] = q[3:-3,:]

        time = t_vec[i]
        time_list.append(time)
        data = q[3:-3,0]
        data = data.tolist()
        result_q.append(data)



print('Program complete.\n')

#save result as csv files
time_df = pd.DataFrame(time_list)
result_df = pd.DataFrame(result_q)
result_new = pd.concat([time_df,result_df],axis=1)
result_new.to_csv(result_file)
