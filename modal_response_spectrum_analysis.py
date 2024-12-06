# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:17:30 2024

@author: Leon Schöne
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

g = 9.81
q = 4
a_gR = 0.35 * g
gamma_I = 1.4
a_g = gamma_I * a_gR
damp_ratio = 3 # given in %
damp_corr_fac = np.max([0.55, np.sqrt(10/(5+damp_ratio))])



def plot_horizontal_elastic_response_spectrum():
    T_B = 0.15
    T_C = 0.5
    T_D = 2
    S = 1.4

    S_e_B = S*damp_corr_fac*2.5
    S_e_D = S*damp_corr_fac*2.5*(T_C/T_D)

    S_e = np.array([])
    for time_step in np.arange(0, 4.01, 0.01):
        if time_step <= T_B:
            S_e_i = a_g*S*(1+(time_step/T_B)*(damp_corr_fac*2.5-1))
            S_e = np.append(S_e, S_e_i)
        elif time_step <= T_C:
            S_e_i = a_g*S*damp_corr_fac*2.5
            S_e = np.append(S_e, S_e_i)
        elif time_step <= T_D:
            S_e_i = a_g*S*damp_corr_fac*2.5*(T_C/time_step)
            S_e = np.append(S_e, S_e_i)
        else:
            S_e_i = a_g*S*damp_corr_fac*2.5*((T_C*T_D)/time_step**2)
            S_e = np.append(S_e, S_e_i)
            
    T = np.arange(0, 4.01, 0.01)
    fig1 = plt.figure(figsize=(5,3))
    plt.plot(T, S_e/a_g)
    # plt.plot(T, S_e)
    plt.vlines(T_B, 0, S_e_B, colors='black', linestyles=':')
    plt.vlines(T_C, 0, S_e_B, colors='black', linestyles=':')
    plt.vlines(T_D, 0, S_e_D, colors='black', linestyles=':')
    plt.xlim(0,)
    plt.ylim(0,)
    plt.xlabel('Period [s]')
    plt.ylabel('$S_e$/$a_g$')
    plt.show()
    fig1.savefig(f"../plots/elastic_response_spectrum.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')
    
    return np.array([T, S_e]).T  # S_e
    
    
def plot_vertical_elastic_response_spectrum():
    T_B = 0.05
    T_C = 0.15
    T_D = 1
    a_vg = 0.9 * a_g

    S_ve_B = a_vg*damp_corr_fac*3
    S_ve_D = a_vg*damp_corr_fac*3*(T_C/T_D)

    S_ve = np.array([])
    for time_step in np.arange(0, 4.01, 0.01):
        if time_step <= T_B:
            S_ve_i = a_vg*(1+(time_step/T_B)*(damp_corr_fac*3-1))
            S_ve = np.append(S_ve, S_ve_i)
        elif time_step <= T_C:
            S_ve_i = a_vg*damp_corr_fac*3
            S_ve = np.append(S_ve, S_ve_i)
        elif time_step <= T_D:
            S_ve_i = a_vg*damp_corr_fac*3*(T_C/time_step)
            S_ve = np.append(S_ve, S_ve_i)
        else:
            S_ve_i = a_vg*damp_corr_fac*3*((T_C*T_D)/time_step**2)
            S_ve = np.append(S_ve, S_ve_i)
            
    T = np.arange(0, 4.01, 0.01)
    fig2 = plt.figure(figsize=(5,5))
    plt.plot(T, S_ve/a_vg)
    plt.vlines(T_B, 0, S_ve_B/a_vg, colors='black', linestyles=':')
    plt.vlines(T_C, 0, S_ve_B/a_vg, colors='black', linestyles=':')
    plt.vlines(T_D, 0, S_ve_D/a_vg, colors='black', linestyles=':')
    plt.xlim(0,)
    plt.ylim(0,)
    plt.xlabel('Period [s]')
    plt.ylabel('$S_e$/$a_g$')
    plt.show()
    

def plot_design_spectrum_elastic_analysis():
    T_B = 0.15
    T_C = 0.5
    T_D = 2
    S = 1.4
    beta = 0.2

    S_d_B = S*2.5/q
    S_d_D = max(S*2.5/q*(T_C/T_D), beta)

    S_d = np.array([])
    for time_step in np.arange(0, 4.01, 0.01):
        if time_step <= T_B:
            S_d_i = a_g*S*(2/3 + (time_step/T_B)*(2.5/q - 2/3))
            S_d = np.append(S_d, S_d_i)
        elif time_step <= T_C:
            S_d_i = a_g*S*2.5/q
            S_d = np.append(S_d, S_d_i)
        elif time_step <= T_D:
            S_d_i = max(a_g*S*2.5/q*(T_C/time_step), beta*a_g)
            S_d = np.append(S_d, S_d_i)
        else:
            S_d_i = max(a_g*S*2.5/q*((T_C*T_D)/time_step**2), beta*a_g)
            S_d = np.append(S_d, S_d_i)
            
    T = np.arange(0, 4.01, 0.01)
    fig3 = plt.figure(figsize=(5,3))
    plt.plot(T, S_d/a_g)
    # plt.plot(T, S_d)
    plt.vlines(T_B, 0, S_d_B, colors='black', linestyles=':')
    plt.vlines(T_C, 0, S_d_B, colors='black', linestyles=':')
    plt.vlines(T_D, 0, S_d_D, colors='black', linestyles=':')
    plt.xlim(0,)
    plt.ylim(0,)
    plt.xlabel('Period [s]')
    plt.ylabel('$S_d$/$a_g$')
    plt.show()
    fig3.savefig(f"../plots/design_response_spectrum.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')
    
    return np.array([T, S_d]).T   # S_d



def get_spectral_displacement(S_xn, period, dofs):
    '''
    Parameters
    ----------
    S_xn    : Spectral accelerations as an array with two columns period and acceleration
    period  : Periods of the structure for each mode
    dofs    : Number of DOFs

    Returns
    -------
    S_Dx    : Spectral displacement for each mode
    '''
    S_Dx = np.array([])
    for i in range(dofs):
        target_period = period[i]
        index = np.abs(S_xn[:, 0] - target_period).argmin()
        S_x_i = S_xn[index, 1]*(period[i]/(2*np.pi))**2
        S_Dx = np.append(S_Dx, S_x_i)
        
    return S_Dx


def get_SRSS(df, dofs, modes):
    SRSS = []
    for i in range(0,dofs):
        r_squared = 0 
        for j in range(0,modes):
            r_squared = r_squared + df.iloc[i, j]**2
        SRSS.append(np.sqrt(r_squared))
    
    return np.array(SRSS)


def get_CQC(df, damp_ratio, omega, dofs, modes):
    CQC = []
    for j in range(0,dofs):
        r_squared = 0
        r_first_part = 0
        for i in range(0, modes):
            r_second_part = 0
            r_first_part = r_first_part + df.iloc[j, i]**2
            for n in range(0,modes):
                beta_in = omega[i] / omega[n]
                if beta_in == 1:
                    rho_in = 1
                else:
                    rho_in = (8*damp_ratio**2*(1+beta_in)*beta_in**(3/2)) / ((1-beta_in**2)**2+4*damp_ratio**2*beta_in*(1+beta_in)**2)
                r_second_part = r_second_part + rho_in * df.iloc[j, i] * df.iloc[j, n]
        r_squared = r_first_part + r_second_part
        # r_squared = r_second_part
        CQC.append(np.sqrt(r_squared))
            
            
    return np.array(CQC)



def get_CQC3(df, damp_ratio, omega, dofs, modes):
    CQC = []
    for j in range(0,dofs):
        r_squared = 0
        for i in range(0, modes):
            inner_sum = 0 
            for n in range(0,modes):
                beta_in = omega[i] / omega[n]
                
                if beta_in == 1:
                    rho_in = 1
                else:
                    rho_in = (8*damp_ratio**2*(1+beta_in)*beta_in**(3/2)) / ((1-beta_in**2)**2+4*damp_ratio**2*beta_in*(1+beta_in)**2)
                    
                inner_sum = inner_sum + rho_in * df.iloc[j, i] * df.iloc[j, n]
            r_squared = r_squared + inner_sum
        CQC.append(np.sqrt(r_squared))
            
            
    return np.array(CQC)

    
# plot_horizontal_elastic_response_spectrum()

# plot_vertical_elastic_response_spectrum()  
 
# plot_design_spectrum_elastic_analysis() 
    