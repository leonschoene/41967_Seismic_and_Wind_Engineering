# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 19:25:31 2024

@author: Leon Schöne
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_mode_shape(mode, freq, omegan, U, X, IX, bound, side):
    # Vibration mode to plot

    freq_m = freq[mode-1]
    omegan_m = omegan[mode-1]

    # Mode shape vector normalized to unity
    vec = np.zeros(12,dtype=float)
    vec = U[:,mode-1]
                                 
    # Finding maximum absolut value and normalize to maximal original value                           
    MVec = max(vec)
    mVec = min(vec)

    if abs(MVec) > abs(mVec):
        mxVec = MVec
    else:
        mxVec = mVec
        
    Vplot = vec/mxVec
    
    # Arrange mode shape to take into account the boundary condition and horizontal dof for each node
    # 2 dofs per node

    Vplotall = np.array([[0],
                         [0],
                         [Vplot[1-1]],
                         [Vplot[3-1]],
                         [Vplot[4-1]],
                         [Vplot[6-1]],
                         [Vplot[7-1]],
                         [Vplot[9-1]],
                         [Vplot[10-1]],
                         [Vplot[12-1]],
                         [0],
                         [0],
                         [Vplot[1-1]],
                         [Vplot[2-1]],
                         [Vplot[4-1]],
                         [Vplot[5-1]],
                         [Vplot[7-1]],
                         [Vplot[8-1]],
                         [Vplot[10-1]],
                         [Vplot[11-1]]])

    # Plot mode shape
    E_total = np.size(IX,0) # total number of elements

    # Division points per element
    divp = 20

    A = np.zeros((3,3))
    beam_ex = np.zeros((3,divp))

    scale = 1.0


    # Loop through all element from 1 to 12
    # Calculate the deflection

    t = np.linspace(0, 3*2*np.pi/omegan_m, 100) # time [second]


    fig1 = plt.figure(figsize=(4, 6))
    # for time_step in t:
    #     # Clear the previous plot
    #     plt.clf()
    time_step = 3*2*np.pi/omegan_m
    for e in range(1, E_total+1):
         # Length of each element
         vector = np.array([[X[IX[e-1,1]-1, 0] - X[IX[e-1,0]-1, 0],  X[IX[e-1,1]-1, 1] - X[IX[e-1,0]-1, 1]]])
         l = np.sqrt((vector[:,0]**2) + (vector[:,1]**2))
         
         # Divison points
         x = np.linspace(0,l,divp)
         
         # Shape function
         N1e = -(1 - 3*((x/l)**2)+2*((x/l)**3))
         N2e = x*((1 - (x/l))**2)
         N3e = -(3*((x/l)**2) - 2*((x/l)**3))
         N4e = ((x/l)-1)*(x**2)/l
         
         # From local to global coordinate
         edof_1 = 2*IX[e-1,0]-1
         edof_2 = 2*IX[e-1,0]
         edof_3 = 2*IX[e-1,1]-1
         edof_4 = 2*IX[e-1,1]
         # Deflection
         u1 = Vplotall[edof_1-1]
         u2 = Vplotall[edof_2-1]
         u3 = Vplotall[edof_3-1]
         u4 = Vplotall[edof_4-1]
         
         if (e == 9) or (e==10) or (e==11) or (e==12):
             u1 = 0
             u3 = 0 # These members do not have dof1 and dof3, due to the columns whose axial elongation is ignored.
         
         # Total shape function
         Ne = u1*N1e + u2*N2e + u3*N3e + u4*N4e
         
         # Transformation matrix
         alpha = np.arctan(vector[:,1]/vector[:,0])*180/np.pi # degree
         
         A[0,0] = np.cos(alpha*np.pi/180)
         A[0,1] = -np.sin(alpha*np.pi/180)
         A[0,2] = vector[:,0] 
         A[1,0] = np.sin(alpha*np.pi/180)
         A[1,1] = np.cos(alpha*np.pi/180)
         A[1,2] = vector[:,1] 
         A[2,2] = 1
         
         beam_ex[0,:] = np.transpose(x)
         beam_ex[1,:] = np.transpose(Ne)
         
         # Max.Displacement in global coordinate
         beam_transf = np.matmul(A,beam_ex)
         
         if (e==1) or (e==2) or (e==3) or (e==4) or (e==5) or (e==6) or (e==7) or (e==8):
             # Apply displacement to the structure
             xx_deformed = (beam_transf[0,:]*np.cos(omegan_m*time_step)) + X[IX[e-1,0]-1, 0]
             yy_deformed = beam_transf[1,:] + X[IX[e-1,0]-1, 1]
         else:
             yy_deformed = (beam_transf[1,:]*np.cos(omegan_m*time_step)) + X[IX[e-1,0]-1, 1]
             xx_deformed = beam_transf[0,:] + X[IX[e-1,0]-1, 0]
         if e == 9:
             xx_deformed = xx_deformed + Vplot[0]*np.cos(omegan_m*time_step)
         if e==10:
             xx_deformed = xx_deformed + Vplot[3]*np.cos(omegan_m*time_step)
         if e==11:
             xx_deformed = xx_deformed + Vplot[6]*np.cos(omegan_m*time_step)
         if e==12:
             xx_deformed = xx_deformed + Vplot[9]*np.cos(omegan_m*time_step)
         
         # Undeformed configuration
         xx = X[IX[e-1,:]-1,0]
         yy = X[IX[e-1,:]-1,1]
         
         h1 = plt.plot(xx,yy, "b--")
         h2 = plt.plot(xx_deformed*scale, yy_deformed*scale, "red")  
             
            
    plt.xlim(-2, 12)
    plt.ylim(0, 16)
         
    # plt.title("Mode shape (mode " + str(mode) + ") , fn = " + str(round(freq_m,2)) + " Hz, time = " + str(round(time_step,2)) + " s")
    plt.xlabel("Building width [m]")
    plt.ylabel("Building height [m]")
    plt.legend(['Undeformed state', 'Deformed state'])
    plt.grid()
    # Pause for a short moment to create the animation effect
    plt.pause(0.2)
            
        
    plt.show()
    fig1.savefig(f"../plots/{side}_side_Mode_{mode}.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')
    
    
    
    
def plot_mode_shape_without_annotations(mode, freq, omegan, U, X, IX, bound, side):
    # Vibration mode to plot

    freq_m = freq[mode-1]
    omegan_m = omegan[mode-1]

    # Mode shape vector normalized to unity
    vec = np.zeros(12,dtype=float)
    vec = U[:,mode-1]
    #Vplot = Vplot/np.max(np.abs((Vplot[np.arange(0,np.size(Vplot),3)],
                                 #Vplot[np.arange(1,np.size(Vplot),3)])))
                                 
    # Finding maximum absolut value and normalize to maximal original value                           
    MVec = max(vec)
    mVec = min(vec)

    if abs(MVec) > abs(mVec):
        mxVec = MVec
    else:
        mxVec = mVec
        
    Vplot = vec/mxVec
    
    # Arrange mode shape to take into account the boundary condition and horizontal dof for each node
    # 2 dofs per node

    Vplotall = np.array([[0],
                         [0],
                         [Vplot[1-1]],
                         [Vplot[3-1]],
                         [Vplot[4-1]],
                         [Vplot[6-1]],
                         [Vplot[7-1]],
                         [Vplot[9-1]],
                         [Vplot[10-1]],
                         [Vplot[12-1]],
                         [0],
                         [0],
                         [Vplot[1-1]],
                         [Vplot[2-1]],
                         [Vplot[4-1]],
                         [Vplot[5-1]],
                         [Vplot[7-1]],
                         [Vplot[8-1]],
                         [Vplot[10-1]],
                         [Vplot[11-1]]])

    # Plot mode shape
    E_total = np.size(IX,0) # total number of elements

    # Division points per element
    divp = 20

    A = np.zeros((3,3))
    beam_ex = np.zeros((3,divp))

    scale = 1.0


    # Loop through all element from 1 to 12
    # Calculate the deflection

    t = np.linspace(0, 3*2*np.pi/omegan_m, 100) # time [second]


    fig1 = plt.figure(figsize=(4, 6))
    time_step = 3*2*np.pi/omegan_m
    for e in range(1, E_total+1):
         # Length of each element
         vector = np.array([[X[IX[e-1,1]-1, 0] - X[IX[e-1,0]-1, 0],  X[IX[e-1,1]-1, 1] - X[IX[e-1,0]-1, 1]]])
         l = np.sqrt((vector[:,0]**2) + (vector[:,1]**2))
         
         # Divison points
         x = np.linspace(0,l,divp)
         
         # Shape function
         N1e = -(1 - 3*((x/l)**2)+2*((x/l)**3))
         N2e = x*((1 - (x/l))**2)
         N3e = -(3*((x/l)**2) - 2*((x/l)**3))
         N4e = ((x/l)-1)*(x**2)/l
         
         # From local to global coordinate
         edof_1 = 2*IX[e-1,0]-1
         edof_2 = 2*IX[e-1,0]
         edof_3 = 2*IX[e-1,1]-1
         edof_4 = 2*IX[e-1,1]
         # Deflection
         u1 = Vplotall[edof_1-1]
         u2 = Vplotall[edof_2-1]
         u3 = Vplotall[edof_3-1]
         u4 = Vplotall[edof_4-1]
         
         if (e == 9) or (e==10) or (e==11) or (e==12):
             u1 = 0
             u3 = 0 # These members do not have dof1 and dof3, due to the columns whose axial elongation is ignored.
         
         # Total shape function
         Ne = u1*N1e + u2*N2e + u3*N3e + u4*N4e
         
         # Transformation matrix
         alpha = np.arctan(vector[:,1]/vector[:,0])*180/np.pi # degree
         
         A[0,0] = np.cos(alpha*np.pi/180)
         A[0,1] = -np.sin(alpha*np.pi/180)
         A[0,2] = vector[:,0] 
         A[1,0] = np.sin(alpha*np.pi/180)
         A[1,1] = np.cos(alpha*np.pi/180)
         A[1,2] = vector[:,1] 
         A[2,2] = 1
         
         beam_ex[0,:] = np.transpose(x)
         beam_ex[1,:] = np.transpose(Ne)
         
         # Max.Displacement in global coordinate
         beam_transf = np.matmul(A,beam_ex)
         
         if (e==1) or (e==2) or (e==3) or (e==4) or (e==5) or (e==6) or (e==7) or (e==8):
             # Apply displacement to the structure
             xx_deformed = (beam_transf[0,:]*np.cos(omegan_m*time_step)) + X[IX[e-1,0]-1, 0]
             yy_deformed = beam_transf[1,:] + X[IX[e-1,0]-1, 1]
         else:
             yy_deformed = (beam_transf[1,:]*np.cos(omegan_m*time_step)) + X[IX[e-1,0]-1, 1]
             xx_deformed = beam_transf[0,:] + X[IX[e-1,0]-1, 0]
         if e == 9:
             xx_deformed = xx_deformed + Vplot[0]*np.cos(omegan_m*time_step)
         if e==10:
             xx_deformed = xx_deformed + Vplot[3]*np.cos(omegan_m*time_step)
         if e==11:
             xx_deformed = xx_deformed + Vplot[6]*np.cos(omegan_m*time_step)
         if e==12:
             xx_deformed = xx_deformed + Vplot[9]*np.cos(omegan_m*time_step)
         
         # Undeformed configuration
         xx = X[IX[e-1,:]-1,0]
         yy = X[IX[e-1,:]-1,1]
         
         h1 = plt.plot(xx,yy, "b--")
         h2 = plt.plot(xx_deformed*scale, yy_deformed*scale, "red")  
             
    plt.xlim(-2, 12)
    plt.ylim(0, 16)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.xticks([])  # x-Achsen-Ticks entfernen
    plt.yticks([])  # y-Achsen-Ticks entfernen
    plt.pause(0.2)
            
        
    plt.show()
    fig1.savefig(f"../plots/{side}_side_Mode_{mode}_without_annotation.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')    