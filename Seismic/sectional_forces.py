# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:30:59 2024

@author: Leon Schöne
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_sectional_forces(E, Ib, Ic, Lb, Lc, U):
    
    # create local stiffness matrices for beam and column
    
    k_11b = 12*E*Ib/Lb**3
    k_12b = 6*E*Ib/Lb**2
    k_13b = -12*E*Ib/Lb**3
    k_14b = 6*E*Ib/Lb**2
    
    k_21b = 6*E*Ib/Lb**2
    k_22b = 4*E*Ib/Lb
    k_23b = -6*E*Ib/Lb**2
    k_24b = 2*E*Ib/Lb
    
    k_31b = -12*E*Ib/Lb**3
    k_32b = -6*E*Ib/Lb**2
    k_33b = 12*E*Ib/Lb**3
    k_34b = -6*E*Ib/Lb**2
    
    k_41b = 6*E*Ib/Lb**2
    k_42b = 2*E*Ib/Lb
    k_43b = -6*E*Ib/Lb**2
    k_44b = 4*E*Ib/Lb
    
    K_local_b = np.array([[k_11b, k_12b, k_13b, k_14b],
                          [k_21b, k_22b, k_23b, k_24b],
                          [k_31b, k_32b, k_33b, k_34b],
                          [k_41b, k_42b, k_43b, k_44b]])
    
    k_11c = 12*E*Ic/Lc**3
    k_12c = 6*E*Ic/Lc**2
    k_13c = -12*E*Ic/Lc**3
    k_14c = 6*E*Ic/Lc**2
   
    k_21c = 6*E*Ic/Lc**2
    k_22c = 4*E*Ic/Lc
    k_23c = -6*E*Ic/Lc**2
    k_24c = 2*E*Ic/Lc
   
    k_31c = -12*E*Ic/Lc**3
    k_32c = -6*E*Ic/Lc**2
    k_33c = 12*E*Ic/Lc**3
    k_34c = -6*E*Ic/Lc**2
   
    k_41c = 6*E*Ic/Lc**2
    k_42c = 2*E*Ic/Lc
    k_43c = -6*E*Ic/Lc**2
    k_44c = 4*E*Ic/Lc
   
    K_local_c = np.array([[k_11c, k_12c, k_13c, k_14c],
                          [k_21c, k_22c, k_23c, k_24c],
                          [k_31c, k_32c, k_33c, k_34c],
                          [k_41c, k_42c, k_43c, k_44c]])
    # print(K_local_b)
    # calculate forces in each element
    '''
    Each variable has four entries. First and third are the shear force, 
    the second and fourth entries are the moments.
    
    Calculation is done for each mode.
    '''
    sectional_forces_total = pd.DataFrame()
    
    for i in range(12):
        F1 = K_local_c @ np.array([[U.iloc[i,0]], [U.iloc[i,2]], [0], [0]])
        F2 = K_local_c @ np.array([[U.iloc[i,3]], [U.iloc[i,5]], [U.iloc[i,0]], [U.iloc[i,2]]])
        F3 = K_local_c @ np.array([[U.iloc[i,6]], [U.iloc[i,8]], [U.iloc[i,3]], [U.iloc[i,5]]])
        F4 = K_local_c @ np.array([[U.iloc[i,9]], [U.iloc[i,11]], [U.iloc[i,6]], [U.iloc[i,8]]])
        
        F5 = K_local_c @ np.array([[U.iloc[i,0]], [U.iloc[i,1]], [0], [0]])
        F6 = K_local_c @ np.array([[U.iloc[i,3]], [U.iloc[i,4]], [U.iloc[i,0]], [U.iloc[i,1]]])
        F7 = K_local_c @ np.array([[U.iloc[i,6]], [U.iloc[i,7]], [U.iloc[i,3]], [U.iloc[i,4]]])
        F8 = K_local_c @ np.array([[U.iloc[i,9]], [U.iloc[i,10]], [U.iloc[i,6]], [U.iloc[i,7]]])
        
        F9 = K_local_b @ np.array([[0], [U.iloc[i,2]], [0], [U.iloc[i,1]]])
        F10 = K_local_b @ np.array([[0], [U.iloc[i,5]], [0], [U.iloc[i,4]]])
        F11 = K_local_b @ np.array([[0], [U.iloc[i,8]], [0], [U.iloc[i,7]]])
        F12 = K_local_b @ np.array([[0], [U.iloc[i,11]], [0], [U.iloc[i,10]]])
    
        sectional_forces = pd.DataFrame({
            'Element 1': F1.flatten(),
            'Element 2': F2.flatten(),
            'Element 3': F3.flatten(),
            'Element 4': F4.flatten(),
            'Element 5': F5.flatten(),
            'Element 6': F6.flatten(),
            'Element 7': F7.flatten(),
            'Element 8': F8.flatten(),
            'Element 9': F9.flatten(),
            'Element 10': F10.flatten(),
            'Element 11': F11.flatten(),
            'Element 12': F12.flatten()
            })
        sectional_forces_total = pd.concat([sectional_forces_total, sectional_forces], ignore_index=True)
    
    return sectional_forces_total/1000 # in kN and kNm

def get_sectional_forces_2(E, Ib, Ic, Lb, Lc, U):
    
    # create local stiffness matrices for beam and column
    
    k_11b = 12*E*Ib/Lb**3
    k_12b = 6*E*Ib/Lb**2
    k_13b = -12*E*Ib/Lb**3
    k_14b = 6*E*Ib/Lb**2
    
    k_21b = 6*E*Ib/Lb**2
    k_22b = 4*E*Ib/Lb
    k_23b = -6*E*Ib/Lb**2
    k_24b = 2*E*Ib/Lb
    
    k_31b = -12*E*Ib/Lb**3
    k_32b = -6*E*Ib/Lb**2
    k_33b = 12*E*Ib/Lb**3
    k_34b = -6*E*Ib/Lb**2
    
    k_41b = 6*E*Ib/Lb**2
    k_42b = 2*E*Ib/Lb
    k_43b = -6*E*Ib/Lb**2
    k_44b = 4*E*Ib/Lb
    
    K_local_b = np.array([[k_11b, k_12b, k_13b, k_14b],
                          [k_21b, k_22b, k_23b, k_24b],
                          [k_31b, k_32b, k_33b, k_34b],
                          [k_41b, k_42b, k_43b, k_44b]])
    
    k_11c = 12*E*Ic/Lc**3
    k_12c = 6*E*Ic/Lc**2
    k_13c = -12*E*Ic/Lc**3
    k_14c = 6*E*Ic/Lc**2
   
    k_21c = 6*E*Ic/Lc**2
    k_22c = 4*E*Ic/Lc
    k_23c = -6*E*Ic/Lc**2
    k_24c = 2*E*Ic/Lc
   
    k_31c = -12*E*Ic/Lc**3
    k_32c = -6*E*Ic/Lc**2
    k_33c = 12*E*Ic/Lc**3
    k_34c = -6*E*Ic/Lc**2
   
    k_41c = 6*E*Ic/Lc**2
    k_42c = 2*E*Ic/Lc
    k_43c = -6*E*Ic/Lc**2
    k_44c = 4*E*Ic/Lc
   
    K_local_c = np.array([[k_11c, k_12c, k_13c, k_14c],
                          [k_21c, k_22c, k_23c, k_24c],
                          [k_31c, k_32c, k_33c, k_34c],
                          [k_41c, k_42c, k_43c, k_44c]])
    # print(K_local_b)
    # calculate forces in each element
    '''
    Each variable has four entries. First and third are the shear force, 
    the second and fourth entries are the moments.
    
    Calculation is done for each mode.
    '''
    sectional_forces_total = pd.DataFrame()
    
    for i in range(12):
        F1 = K_local_c @ np.array([[U[0]], [U[2]], [0], [0]])
        F2 = K_local_c @ np.array([[U[3]], [U[5]], [U[0]], [U[2]]])
        F3 = K_local_c @ np.array([[U[6]], [U[8]], [U[3]], [U[5]]])
        F4 = K_local_c @ np.array([[U[9]], [U[11]], [U[6]], [U[8]]])
        
        F5 = K_local_c @ np.array([[U[0]], [U[1]], [0], [0]])
        F6 = K_local_c @ np.array([[U[3]], [U[4]], [U[0]], [U[1]]])
        F7 = K_local_c @ np.array([[U[6]], [U[7]], [U[3]], [U[4]]])
        F8 = K_local_c @ np.array([[U[9]], [U[10]], [U[6]], [U[7]]])
        
        F9 = K_local_b @ np.array([[0], [U[2]], [0], [U[1]]])
        F10 = K_local_b @ np.array([[0], [U[5]], [0], [U[4]]])
        F11 = K_local_b @ np.array([[0], [U[8]], [0], [U[7]]])
        F12 = K_local_b @ np.array([[0], [U[11]], [0], [U[10]]])
    
        sectional_forces = pd.DataFrame({
            'Element 1': F1.flatten(),
            'Element 2': F2.flatten(),
            'Element 3': F3.flatten(),
            'Element 4': F4.flatten(),
            'Element 5': F5.flatten(),
            'Element 6': F6.flatten(),
            'Element 7': F7.flatten(),
            'Element 8': F8.flatten(),
            'Element 9': F9.flatten(),
            'Element 10': F10.flatten(),
            'Element 11': F11.flatten(),
            'Element 12': F12.flatten()
            })
        sectional_forces_total = pd.concat([sectional_forces_total, sectional_forces], ignore_index=True)
    
    return sectional_forces_total/1000 # in kN and kNm


def get_sectional_forces_3(E, Ib, Ic, Lb, Lc, U):
    
    # create local stiffness matrices for beam and column
    
    k_11b = 12*E*Ib/Lb**3
    k_12b = 6*E*Ib/Lb**2
    k_13b = -12*E*Ib/Lb**3
    k_14b = 6*E*Ib/Lb**2
    
    k_21b = 6*E*Ib/Lb**2
    k_22b = 4*E*Ib/Lb
    k_23b = -6*E*Ib/Lb**2
    k_24b = 2*E*Ib/Lb
    
    k_31b = -12*E*Ib/Lb**3
    k_32b = -6*E*Ib/Lb**2
    k_33b = 12*E*Ib/Lb**3
    k_34b = -6*E*Ib/Lb**2
    
    k_41b = 6*E*Ib/Lb**2
    k_42b = 2*E*Ib/Lb
    k_43b = -6*E*Ib/Lb**2
    k_44b = 4*E*Ib/Lb
    
    K_local_b = np.array([[k_11b, k_12b, k_13b, k_14b],
                          [k_21b, k_22b, k_23b, k_24b],
                          [k_31b, k_32b, k_33b, k_34b],
                          [k_41b, k_42b, k_43b, k_44b]])
    
    k_11c = 12*E*Ic/Lc**3
    k_12c = 6*E*Ic/Lc**2
    k_13c = -12*E*Ic/Lc**3
    k_14c = 6*E*Ic/Lc**2
   
    k_21c = 6*E*Ic/Lc**2
    k_22c = 4*E*Ic/Lc
    k_23c = -6*E*Ic/Lc**2
    k_24c = 2*E*Ic/Lc
   
    k_31c = -12*E*Ic/Lc**3
    k_32c = -6*E*Ic/Lc**2
    k_33c = 12*E*Ic/Lc**3
    k_34c = -6*E*Ic/Lc**2
   
    k_41c = 6*E*Ic/Lc**2
    k_42c = 2*E*Ic/Lc
    k_43c = -6*E*Ic/Lc**2
    k_44c = 4*E*Ic/Lc
   
    K_local_c = np.array([[k_11c, k_12c, k_13c, k_14c],
                          [k_21c, k_22c, k_23c, k_24c],
                          [k_31c, k_32c, k_33c, k_34c],
                          [k_41c, k_42c, k_43c, k_44c]])
    # print(K_local_b)
    # calculate forces in each element
    '''
    Each variable has four entries. First and third are the shear force, 
    the second and fourth entries are the moments.
    
    Calculation is done for each mode.
    '''
    sectional_forces_total = pd.DataFrame()
    
    for i in range(12):
        F1 = K_local_c @ np.array([[U.iloc[0,i]], [U.iloc[2,i]], [0], [0]])
        F2 = K_local_c @ np.array([[U.iloc[3,i]], [U.iloc[5,i]], [U.iloc[0,i]], [U.iloc[2,i]]])
        F3 = K_local_c @ np.array([[U.iloc[6,i]], [U.iloc[8,i]], [U.iloc[3,i]], [U.iloc[5,i]]])
        F4 = K_local_c @ np.array([[U.iloc[9,i]], [U.iloc[11,i]], [U.iloc[6,i]], [U.iloc[8,i]]])
        
        F5 = K_local_c @ np.array([[U.iloc[0,i]], [U.iloc[1,i]], [0], [0]])
        F6 = K_local_c @ np.array([[U.iloc[3,i]], [U.iloc[4,i]], [U.iloc[0,i]], [U.iloc[1,i]]])
        F7 = K_local_c @ np.array([[U.iloc[6,i]], [U.iloc[7,i]], [U.iloc[3,i]], [U.iloc[4,i]]])
        F8 = K_local_c @ np.array([[U.iloc[9,i]], [U.iloc[10,i]], [U.iloc[6,i]], [U.iloc[7,i]]])
        
        F9 = K_local_b @ np.array([[0], [U.iloc[2,i]], [0], [U.iloc[1,i]]])
        F10 = K_local_b @ np.array([[0], [U.iloc[5,i]], [0], [U.iloc[4,i]]])
        F11 = K_local_b @ np.array([[0], [U.iloc[8,i]], [0], [U.iloc[7,i]]])
        F12 = K_local_b @ np.array([[0], [U.iloc[11,i]], [0], [U.iloc[10,i]]])
    
        sectional_forces = pd.DataFrame({
            'Element 1': F1.flatten(),
            'Element 2': F2.flatten(),
            'Element 3': F3.flatten(),
            'Element 4': F4.flatten(),
            'Element 5': F5.flatten(),
            'Element 6': F6.flatten(),
            'Element 7': F7.flatten(),
            'Element 8': F8.flatten(),
            'Element 9': F9.flatten(),
            'Element 10': F10.flatten(),
            'Element 11': F11.flatten(),
            'Element 12': F12.flatten()
            })
        sectional_forces_total = pd.concat([sectional_forces_total, sectional_forces], ignore_index=True)
    
    return sectional_forces_total/1000 # in kN and kNm


def get_SF_SRSS(df, elements):
    results = {
        "F1_SRSS": [],
        "F2_SRSS": [],
        "F3_SRSS": [],
        "F4_SRSS": []
    }

    for i in range(elements):
        F1 = 0
        F2 = 0
        F3 = 0
        F4 = 0
        for j in range(0, 4 * elements, 4):
            F1 += df.iloc[j, i] ** 2
            F2 += df.iloc[j + 1, i] ** 2
            F3 += df.iloc[j + 2, i] ** 2
            F4 += df.iloc[j + 3, i] ** 2
        # Calculate SRSS for each F value and save in dictionary
        results["F1_SRSS"].append(np.sqrt(F1))
        results["F2_SRSS"].append(np.sqrt(F2))
        results["F3_SRSS"].append(np.sqrt(F3))
        results["F4_SRSS"].append(np.sqrt(F4))

    # Create new DataFrame from dictionaries
    srss_df = pd.DataFrame(results)
    
    # Transpose DataFrame to get elements as columns and Forces as rows 
    srss_df = srss_df.T
    srss_df.columns = [f"Element_{i+1}" for i in range(elements)]  # rename column
    
    return srss_df
    
def get_SF_CQC(df, damp_ratio, omega, elements):
    results = {
        "F1_CQC": [],
        "F2_CQC": [],
        "F3_CQC": [],
        "F4_CQC": []
    }
    for j in range(0,elements):
        F1 = 0
        F2 = 0
        F3 = 0
        F4 = 0
        for i in range(0, 4*elements, 4):
            inner_sum1 = 0
            inner_sum2 = 0
            inner_sum3 = 0
            inner_sum4 = 0
            for n in range(0,4*elements, 4):
                beta_in = omega[int(i/4)] / omega[int(n/4)]
                
                if beta_in == 1:
                    rho_in = 1
                else:
                    rho_in = (8*damp_ratio**2*(1+beta_in)*beta_in**(3/2)) / ((1-beta_in**2)**2+4*damp_ratio**2*beta_in*(1+beta_in)**2)
                    
                inner_sum1 = inner_sum1 + rho_in * df.iloc[i, j] * df.iloc[n, j]
                inner_sum2 = inner_sum2 + rho_in * df.iloc[i+1, j] * df.iloc[n+1, j]
                inner_sum3 = inner_sum3 + rho_in * df.iloc[i+2, j] * df.iloc[n+2, j]
                inner_sum4 = inner_sum4 + rho_in * df.iloc[i+3, j] * df.iloc[n+3, j]
            F1 = F1 + inner_sum1
            F2 = F2 + inner_sum2
            F3 = F3 + inner_sum3
            F4 = F4 + inner_sum4
        results["F1_CQC"].append(np.sqrt(F1))
        results["F2_CQC"].append(np.sqrt(F2))
        results["F3_CQC"].append(np.sqrt(F3))
        results["F4_CQC"].append(np.sqrt(F4))
            
    cqc_df = pd.DataFrame(results)
    
    # Transpose DataFrame to get elements as columns and Forces as rows 
    cqc_df = cqc_df.T
    cqc_df.columns = [f"Element_{i+1}" for i in range(elements)]  # rename column
    
    return cqc_df


def plotdiagram(X,IX,ne,Mz,scale,title):
    # This function plots the deformed and undeformed structure
    # Element type
    nnodes = np.size(IX,1)-1

    if nnodes == 2:
        order = np.array([0,1],dtype=int)
    i = 0
    plt.figure(figsize=(8, 8))
    for e in range(0,ne):
        
        # Initial geometry
        plt.plot(X[IX[e,order]-1,0],X[IX[e,order]-1,1],"b--") 
        
        # Deformed geometry
        X1 = X[IX[e,0]-1,0:2]
        a0 = X[IX[e,1]-1,0:2]-X[IX[e,0]-1,0:2]
        alpha = np.arctan((X[IX[e,1]-1,1] - X[IX[e,0]-1,1])/(X[IX[e,1]-1,0] - X[IX[e,0]-1,0]))
        mzx1 = (Mz[2*i]*scale)*np.sin(alpha) + X[IX[e,0]-1,0]
        mzx2 = (Mz[2*i + 1]*scale)*np.sin(alpha) + X[IX[e,1]-1,0]
        mzy1 = -(Mz[2*i]*scale)*np.cos(alpha) + X[IX[e,0]-1,1]
        mzy2 = -(Mz[2*i+1]*scale)*np.cos(alpha) + X[IX[e,1]-1,1]
        
        mzx = np.array([X[IX[e,0]-1,0], mzx1, mzx2, X[IX[e,1]-1,0]])
        mzy = np.array([X[IX[e,0]-1,1], mzy1, mzy2, X[IX[e,1]-1,1]])
        plt.plot(mzx,mzy,"red")
               
        i = i + 1
        
    
    plt.text(X[IX[1,1]-1,0], X[IX[1,1]-1,1], str(round(Mz[3],1)))
    plt.text(X[IX[1,0]-1,0], X[IX[1,0]-1,1], str(round(Mz[2],1)))
    plt.text(X[IX[0,0]-1,0], X[IX[0,0]-1,1], str(round(Mz[0],1)))
    plt.text(X[IX[2,1]-1,0], X[IX[2,1]-1,1], str(round(Mz[5],1)))
    
    plt.xlim([min(X[:,0])- 3, max(X[:,0])+3])
    plt.ylim([min(X[:,1])- 3, max(X[:,1])+3])
    plt.title(title)
    plt.xlabel("X-axis [m]")
    plt.ylabel("Y-axis [m]")
    plt.legend(["Structure", title])
    plt.grid()  


























