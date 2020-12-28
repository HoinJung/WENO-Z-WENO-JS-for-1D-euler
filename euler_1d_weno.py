# N is all the points including ghosts

def init_cond(X_min,X_max,N,init):
#compute the initial condition on the grid for Reimann problem
#coordinate system centered at fluid interface (default is x=0, otherwise x=x_bnd)

    import numpy as np
    from math import sin
    
    #define constants
    gam = 1.4
    R = 286.9

    #allocate space for variables
    q_init = np.zeros((N,3))
    rho = np.zeros(N)
    u = np.zeros(N)
    P = np.zeros(N)
    T = np.zeros(N)

    #define the grid with N total points (ghost points outside domain)
    X = np.zeros(N)
    X[3:-3] = np.linspace(X_min,X_max,N-6)
    dx = X[4]-X[3]
    for i in range(3):
        X[i] = X_min+(i-3)*dx
        X[-(i+1)] = X_max+(3-i)*dx 
        
    #set the initial condition
    x0=0.0
    if init == "SOD":
        rho1 = 0.125
        U1 = 0
        P1 = 0.1
        rho4 = 1.0
        U4 = 0
        P4 = 1
    elif init == "LAX":
        rho1 = 0.445
        U1 = 0.698
        P1 = 3.528
#         P1 = 0.3528
        rho4 = 0.5
        U4 = 0
        P4 = 0.5710
    elif init == "123":
        rho1 = 1.0
        U1 = -2.0
        P1 = 0.4
        rho4 = 1.0
        U4 = 2.0
        P4 = 0.4
    elif init == "Shockdensity":
        rho1 = 3.857143
        U1 = 2.629369
        P1 = 31/3
        rho4 = 1
        U4 = 0
        P4 = 1.0
        x0 = -4.0
    elif init == "blastwave":
        rho1 = 1
        U1 = 0
        P1 = 1000
        rho4 = 1
        U4 = 0
        P4 = 0.01
        rho6 = 1
        U6 = 0
        P6 = 100
        
        x0 = 0.1
        x1 = 0.9
        T6= P6/(R*rho6)
        
    T1 = P1/(R*rho1)
    T4 = P4/(R*rho4)
    if init=="blastwave":
        for i in range(N):
            if(X[i]<=x0):
                P[i] = P1
                T[i] = T1
                u[i] = U1
            elif(X[i]>x0 and X[i]<=x1):
                P[i] = P4
                T[i] = T4
                u[i] = U4
            else:
                P[i] = P6
                T[i] = T6
                u[i] = U6
            rho[i] = P[i]/(R*T[i])
    elif init=="Shockdensity":
        for i in range(N):
            if(X[i]<=x0):
                P[i] = P1
                T[i] = T1
                u[i] = U1
                rho[i] = P[i]/(R*T[i])
            else:
                P[i] = P4
                T[i] = T4
                u[i] = U4
                rho[i] = 1+0.2*sin(5*X[i])
    #for Riemman initial condition       
    else : 
        for i in range(N):
            if(X[i]<=x0):
                P[i] = P1
                T[i] = T1
                u[i] = U1
            else:
                P[i] = P4
                T[i] = T4
                u[i] = U4
            rho[i] = P[i]/(R*T[i])
#         u[i] = 0.0

    #define the initial condition vector
    q_init[:,0] = rho
    q_init[:,1] = rho*u
    q_init[:,2] = P/(gam-1.0) + 0.5*rho*u**2
    
    
    if P4 > P1 : 
        P_51 = int(P4/P1)
    else :
        P_51 = int(P1/P4)

    print('Intiial condition generated successfully.')
    
    return q_init, X, dx, P_51


def phys_flux(q):
#q is an Nxnv matrix with N grid points and nv variables   

    import numpy as np

    #primitive variables
    gam = 1.4
    rho = q[:,0]
    u = q[:,1]/q[:,0]
    e = q[:,2]
    p = (gam-1.0)*(e-0.5*rho*u**2)

    #compute the physical flux
    f = np.zeros(q.shape)
    f[:,0] = rho*u
    f[:,1] = p+rho*u**2
    f[:,2] = u*(p+e)

    return f
	
def euler_1d_wavespeed(q):
#q is an Nxnv matrix with N grid points and nv variables   

    import scipy.linalg as la
    import numpy as np
    import sys,os
    
    #primitive variables
    gam = 1.4
    R = 286.9
    rho = q[:,0]
    u = q[:,1]/q[:,0]
    e = q[:,2]
    p = (gam-1.0)*(e-0.5*rho*u**2)
#     t = p/(R*rho)
    t = abs(p/(R*rho))

    if (t.min()<0):
        print('\n============================================================')
        print('     Warning: Negative temperature detected!!!')
        print('     Solution is numerically unstable. Now exiting....' % t.min())
        print('============================================================\n\n')
        sys.exit()
#     c = np.sqrt(gam*p/rho) 
    c = np.sqrt(abs(gam*p/rho)) 

    #define max wavespeed(s) on the grid for LF splitting
    ws = np.zeros(q.shape[1])
    for j in range(q.shape[1]):
        ws[j] = la.norm(u+(j-1)*c,np.inf) 

    return ws


def update_ghost_pts(q,left_bc,right_bc):

    #assign left-end ghost cell values
    if (left_bc == 'Wall'):
        for i in range(3):
            q[(2-i),0] =  q[(i+4),0]
            q[(2-i),1] = -q[(i+4),1]
            q[(2-i),2] =  q[(i+4),2]
    if (left_bc == 'Neumann'):
        for i in range(3):
            q[(2-i),:] =  q[3,:]
    else:
        for i in range(3):
            q[(2-i),:]  = (10*(i+1))**10

#     assign right-end ghost cell values
    if (right_bc == 'Wall'):
        for i in range(3):
            q[-(3-i),0] =  q[-(i+5),0]
            q[-(3-i),1] = -q[-(i+5),1]
            q[-(3-i),2] =  q[-(i+5),2]
    if (right_bc == 'Neumann'):
        for i in range(3):
            q[-(3-i),:] =  q[-4,:]
    else:
        for i in range(3):
            q[-(3-i),:] = (10*(i+1))**10

    return(q)

def char_numerical_flux(q,adv,runMode):

    import numpy as np
    
    # Compute the fluxes on the entire grid
    f = phys_flux(q)

    # Compute the state vector at the x_{1+1/2} points
    q_i_p_half = (q[2:q.shape[0]-3,:] + q[3:q.shape[0]-2,:])*0.5
    
    # -------------------------------------------------------------------------
    
    # Number of x_{i+1/2} points on the domain at which the flux is computed
    N_x_p_half = q_i_p_half.shape[0]
    
    # Number of state variables
    Nvar = q.shape[1]
    
    # WENO full stencil size 
    stencil_size = 5
    
    # Number of ghost points at a boundary
    Ng = 3
    
    # -------------------------------------------------------------------------

    # Compute the max wavespeeds on the entire grid
    ws = euler_1d_wavespeed(q[Ng:q.shape[0]-Ng,:])

    # Initialize the arrays
    f_char_p = np.zeros((Nvar, stencil_size))
    f_char_m = np.zeros((Nvar, stencil_size))
    f_char_i_p_half = np.zeros((N_x_p_half, Nvar))
    
    # Loop through each x_{i+1/2} point on the grid
    # Compute the f_char_p and f_char_m terms for phi_weno5
    # Compute the fifth order accurate weno flux-split terms
    # Add them together to obatin to find f_char_i_p_half
    
    for i in range(N_x_p_half):
        #ws = euler_1d_wavespeed(q[i:i+stencil_size+1,:])
        qi, fi = proj_to_char(q[i:i+stencil_size+1,:], f[i:i+stencil_size+1,:], q_i_p_half[i])
        
        for j in range(stencil_size):
            f_char_p[:,j] = (0.5*( (fi[j,:]).T + (np.diag(ws)).dot((qi[j,:]).T) )).T
            f_char_m[:,j] = (0.5*( (fi[j+1,:]).T - (np.diag(ws)).dot((qi[j+1,:]).T) )).T

        # Compute the i + 1/2 points flux
        if runMode == 'WENO-JS' :
            method = phi_weno5
        elif runMode == 'WENO-Z' :
            method = phi_wenoZ
        for k in range(0, Nvar):
            f_char_i_p_half[i,k] = method(f_char_p[k,:],adv) + method(f_char_m[k,::-1],adv)    
    
    return f_char_i_p_half


def phi_weno5(f_char_p_s,adv):
    '''
    Function which computes a 5th-order WENO reconstruction of the numerical
    flux at location x_{i+1/2}, works regardless of the sign of f'(u)
    '''

    import numpy as np

    #assign the fluxes at each point in the full stencil 
    f_i_m_2 = f_char_p_s[0]
    f_i_m_1 = f_char_p_s[1]
    f_i     = f_char_p_s[2]
    f_i_p_1 = f_char_p_s[3]
    f_i_p_2 = f_char_p_s[4]
    
    #estimate of f_{i+1/2} for each substencil
    f0 = (1/3)*f_i_m_2 - (7/6)*f_i_m_1 + (11/6)*f_i
    f1  = (-1/6)*f_i_m_1 + (5/6)*f_i + (1/3)*f_i_p_1
    f2  = (1/3)*f_i + (5/6)*f_i_p_1 - (1/6)*f_i_p_2
    
    #smoothness indicators for the solution on each substencil 
    beta_0 = (13/12)*(f_i_m_2 - 2*f_i_m_1 + f_i)**2 + (1/4)*(f_i_m_2 - 4*f_i_m_1 + 3*f_i)**2
    beta_1 = (13/12)*(f_i_m_1 - 2*f_i + f_i_p_1)**2 + (1/4)*(f_i_m_1 - f_i_p_1)**2
    beta_2 = (13/12)*(f_i - 2*f_i_p_1 + f_i_p_2)**2 + (1/4)*(3*f_i - 4*f_i_p_1 + f_i_p_2)**2

    #unscaled nonlinear weights 
    epsilon = 1e-6
    w0_tilde = 0.3/(epsilon + beta_0)**2
    w1_tilde = 0.6/(epsilon + beta_1)**2
    w2_tilde = 0.1/(epsilon + beta_2)**2
    
    #scaled nonlinear weights
    w0 = w0_tilde/(w0_tilde + w1_tilde + w2_tilde)
    w1 = w1_tilde/(w0_tilde + w1_tilde + w2_tilde)
    w2 = w2_tilde/(w0_tilde + w1_tilde + w2_tilde)
    
    #overwrite WENO nonlinear weights with optimal linear weights
    if (adv=='LINEAR-FD'): w0 = 0.1; w1 = 0.6; w2 = 0.3;
   
    #linear convex combination of (3) substencil reconstructions
    f_char_i_p_half_p_s = w0*f0 + w1*f1 + w2*f2

    return f_char_i_p_half_p_s

def phi_wenoZ(f_char_p_s,adv):
    '''
    Function which computes a 5th-order WENO reconstruction of the numerical
    flux at location x_{i+1/2}, works regardless of the sign of f'(u)
    '''

    import numpy as np

    #assign the fluxes at each point in the full stencil 
    f_i_m_2 = f_char_p_s[0]
    f_i_m_1 = f_char_p_s[1]
    f_i     = f_char_p_s[2]
    f_i_p_1 = f_char_p_s[3]
    f_i_p_2 = f_char_p_s[4]
    
    #estimate of f_{i+1/2} for each substencil
    f0 = (1/3)*f_i_m_2 - (7/6)*f_i_m_1 + (11/6)*f_i
    f1  = (-1/6)*f_i_m_1 + (5/6)*f_i + (1/3)*f_i_p_1
    f2  = (1/3)*f_i + (5/6)*f_i_p_1 - (1/6)*f_i_p_2
    
    #smoothness indicators for the solution on each substencil 
    beta_0 = (13/12)*(f_i_m_2 - 2*f_i_m_1 + f_i)**2 + (1/4)*(f_i_m_2 - 4*f_i_m_1 + 3*f_i)**2
    beta_1 = (13/12)*(f_i_m_1 - 2*f_i + f_i_p_1)**2 + (1/4)*(f_i_m_1 - f_i_p_1)**2
    beta_2 = (13/12)*(f_i - 2*f_i_p_1 + f_i_p_2)**2 + (1/4)*(3*f_i - 4*f_i_p_1 + f_i_p_2)**2

    #unscaled nonlinear weights 
    epsilon = 1e-40
    tau = abs(beta_0 - beta_2)
    q = 2
    w0_tilde = 0.3*(1+(tau/(beta_0+epsilon))**q)
    w1_tilde = 0.6*(1+(tau/(beta_1+epsilon))**q)
    w2_tilde = 0.1*(1+(tau/(beta_2+epsilon))**q)
    
    #scaled nonlinear weights
    w0 = w0_tilde/(w0_tilde + w1_tilde + w2_tilde)
    w1 = w1_tilde/(w0_tilde + w1_tilde + w2_tilde)
    w2 = w2_tilde/(w0_tilde + w1_tilde + w2_tilde)
    
    #overwrite WENO nonlinear weights with optimal linear weights
    if (adv=='LINEAR-FD'): w0 = 0.1; w1 = 0.6; w2 = 0.3;
   
    #linear convex combination of (3) substencil reconstructions
    f_char_i_p_half_p_s = w0*f0 + w1*f1 + w2*f2

    return f_char_i_p_half_p_s

def proj_to_char(q,f,q_st):
    '''
    q is a nsxnv matrix of conservative variables (ns = num pts in current stencil)  
    f is a nsxnv matrix of conservative fluxes (ns = num pts in current stencil)  
    q_st is a 1xnv vector with nv variables of average state 

    '''
    import numpy as np

    #primitive variables at x_{i+1/2}
    gam = 1.4
    rho = q_st[0]
    u = q_st[1]/q_st[0]
    e = q_st[2]
    p = (gam-1)*(e-0.5*rho*u**2)
    c = np.sqrt(gam*p/rho) 

    #matrix of left eigenvectors of A (eigenvalues in order u-c, u, and u+c)
    L = np.zeros((3,3))
    L[0,0] = 0.5*(0.5*(gam-1.0)*(u/c)**2+(u/c))
    L[1,0] = 1.0-0.5*(gam-1.0)*(u/c)**2
    L[2,0] = 0.5*(0.5*(gam-1.0)*(u/c)**2-(u/c))
    L[0,1] = -(0.5/c)*((gam-1.0)*(u/c)+1.0)
    L[1,1] = (gam-1.0)*u/c**2
    L[2,1] = -(0.5/c)*((gam-1.0)*(u/c)-1.0)
    L[0,2] = L[2,2] = (gam-1.0)/(2*c**2)
    L[1,2] = -(gam-1.0)/c**2
    
    #project solution/flux into characteristic space for each point in stencil
    q_char = L.dot(q.T).T
    f_char = L.dot(f.T).T

    return q_char,f_char

def spatial_rhs(q_cons,dx,adv,left_bc,right_bc,runMode):

    '''
    f_char is a Ni x nv matrix of the characteristic flux only at interior adjacent flux interfaces
    q_cons is a Np x nv matrix of the conservative variables full domain

    '''
    import numpy as np
    import sys,os

    #compute the flux at the x_{i+1/2} points (characteristic proj.)
    f_char = char_numerical_flux(q_cons,adv,runMode)
    
    # Compute the state vector at the x_{1+1/2} points
    q_i_p_half = (q_cons[2:q_cons.shape[0]-3,:] + q_cons[3:q_cons.shape[0]-2,:])*0.5

    # Initialize arrays
    N = f_char.shape[0]
    R = np.zeros((N,3,3))

    # Compute the R matrix at every half point flux location
    for i in range(N):

        #approximate state at x_{i+1/2}
        q_st = q_i_p_half[i,:]

        #primitive variables at x_{i+1/2}
        gam = 1.4
        rho = q_st[0]
        u = q_st[1]/q_st[0]
        e = q_st[2]
        p = (gam-1)*(e-0.5*rho*u**2)
        c = np.sqrt(gam*p/rho) 

        #matrix of right eigenvectors of A (eigenvalues in order u-c, u, and u+c)    
        R[i,0,:] = 1.0
        R[i,1,0] = u-c
        R[i,1,1] = u
        R[i,1,2] = u+c
        R[i,2,0] = c**2/(gam-1.0)+0.5*u**2-u*c
        R[i,2,1] = 0.5*u**2
        R[i,2,2] = c**2/(gam-1.0)+0.5*u**2+u*c

    # Initialize rhs
    rhs = np.zeros((N-1,f_char.shape[1]))

    # Compute qdot at left and right boundaries
    rhs[0,:] = left_b_qdot(np.array(q_cons[3:8,:]),dx,left_bc)
    rhs[-1,:] = right_b_qdot(np.array(q_cons[-8:-3,:]),dx,right_bc)

    #update the rhs values on the interior (and possibly the boundary)
    i_start = 1; i_end = rhs.shape[0]-1
    if (left_bc == 'Neumann'): i_start -= 1
    if (right_bc == 'Neumann'): i_end += 1
    for i in range(i_start,i_end):   
        
        # Local Right Eigenmatrices
        R_p_half = R[i+1,:,:]
        R_m_half = R[i,:,:]
 
        # The local qdot
        rhs[i,:] = (-1/dx)*(R_p_half.dot((f_char[i+1,:]))-R_m_half.dot((f_char[i,:])))
 
    return rhs
    
def left_b_qdot(q,h,bc_type):
    
    import numpy as np
    
    #primitive variables
    gam = 1.4
    rhoarr = q[:,0]
    uarr = q[:,1]/q[:,0]
    earr = q[:,2]
    parr = (gam-1.0)*(earr-0.5*rhoarr*uarr**2)
    carr = np.sqrt(gam*parr/rhoarr) 

    R = np.zeros((3,3))
    
    u = uarr[0]
    rho = rhoarr[0]
    c = carr[0] 
    
    #matrix of right eigenvectors of A (eigenvalues in order u-c, u, and u+c)
    R[0,:] = 1.0
    R[1,0] = u-c
    R[1,1] = u
    R[1,2] = u+c
    R[2,0] = c**2/(gam-1.0)+0.5*u**2-u*c
    R[2,1] = 0.5*u**2
    R[2,2] = c**2/(gam-1.0)+0.5*u**2+u*c    
    
    # Compute spatial gradients
    drhodx = (-25*rhoarr[0]+48*rhoarr[1]-36*rhoarr[2]+16*rhoarr[3]-3*rhoarr[4])/(12*h)
    dpdx =   (-25*parr[0]  +48*parr[1]  -36*parr[2]  +16*parr[3]  -3*parr[4]  )/(12*h)
    dudx =   (-25*uarr[0]  +48*uarr[1]  -36*uarr[2]  +16*uarr[3]  -3*uarr[4]  )/(12*h)
    
    # Apply the NRBC
    f = np.ones(3)
    for j in range(3): 
        if(R[1,j]>0): f[j]=0
    
    # Compute the wave amplitudes
    L1 =  f[0]*(u-c)*(dpdx-rho*c*dudx)/(2*c**2)
    L2 =  f[1]*u*(drhodx-dpdx/c**2)
    L3 =  f[2]*(u+c)*(dpdx+rho*c*dudx)/(2*c**2)
    
    # Apply the Wall BC
    if (bc_type == 'Wall'): 
        L3 = L1
    elif (bc_type == 'Force-Free'):
        L3 = L1+2*rho*c*(u*dudx)

    # Transform back to conservative form
    qdot = -R.dot(np.array([L1,L2,L3]))

    # Bypass characteristic BC if 'Neumann' is selected
    if (bc_type == 'Neumann'): qdot = np.zeros(qdot.shape) 

    return qdot
    
def right_b_qdot(q,h,bc_type):
    
    import scipy.interpolate as interp
    import numpy as np
    
    #primitive variables
    gam = 1.4
    rhoarr = q[:,0]
    uarr = q[:,1]/q[:,0]
    earr = q[:,2]
    parr = (gam-1.0)*(earr-0.5*rhoarr*uarr**2)
    carr = np.sqrt(gam*parr/rhoarr) 

    R = np.zeros((3,3))
    
    u = uarr[-1]
    rho = rhoarr[-1]
    c = carr[-1] 
    
    #matrix of right eigenvectors of A (eigenvalues in order u-c, u, and u+c)
    R[0,:] = 1.0
    R[1,0] = u-c
    R[1,1] = u
    R[1,2] = u+c
    R[2,0] = c**2/(gam-1.0)+0.5*u**2-u*c
    R[2,1] = 0.5*u**2
    R[2,2] = c**2/(gam-1.0)+0.5*u**2+u*c    
    
    # Compute spatial gradients
    drhodx = (3*rhoarr[-5]-16*rhoarr[-4]+36*rhoarr[-3]-48*rhoarr[-2]+25*rhoarr[-1])/(12*h)
    dpdx =   (3*parr[-5]  -16*parr[-4]  +36*parr[-3]  -48*parr[-2]  +25*parr[-1]  )/(12*h)
    dudx =   (3*uarr[-5]  -16*uarr[-4]  +36*uarr[-3]  -48*uarr[-2]  +25*uarr[-1]  )/(12*h)
    
    # Apply the NRBC
    f = np.ones(3)
    for j in range(3):
        if(R[1,j]<0): f[j]=0
    
    # Compute the wave amplitudes
    L1 =  f[0]*(u-c)*(dpdx-rho*c*dudx)/(2*c**2)
    L2 =  f[1]*u*(drhodx-dpdx/c**2)
    L3 =  f[2]*(u+c)*(dpdx+rho*c*dudx)/(2*c**2)

    # Apply the Wall BC
    if (bc_type == 'Wall'): 
        L1 = L3
    elif (bc_type == 'Force-Free'):
        L1 = L3-2*rho*c*(u*dudx)

    # Transform back to conservative form
    qdot = -R.dot(np.array([L1,L2,L3]))

    # Bypass characteristic BC if 'Neumann' is selected
    if (bc_type == 'Neumann'): qdot = np.zeros(qdot.shape)

    return qdot


