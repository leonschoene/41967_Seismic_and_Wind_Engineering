# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 22:19:04 2024

@author: lwdsc
"""

import numpy as np


def DuhamelInt(t, dt, Acc, omega, m, zeta, MPF, V):
    Nred = len(t[::2])
    ndof, nmodes = V.shape
    
    x_modal = np.zeros((ndof, Nred, nmodes))
    x_total = np.zeros((ndof, Nred))

    for mode in range(nmodes):
        omega_mode = omega[mode]
        omegad_mode = omega_mode * np.sqrt(1 - zeta**2)
        
        m_mode = m[mode]
        
        M1 = 4 * np.exp(-zeta * omega_mode * dt)
        M2 = np.exp(-2 * zeta * omega_mode * dt)
        F = dt / (3 * m_mode * omegad_mode)
        
        fN = m_mode * Acc
        yN = fN * np.cos(omegad_mode * t)
        zN = fN * np.sin(omegad_mode * t)
        
        AN = np.zeros_like(fN)
        BN = np.zeros_like(fN)
        XN = np.zeros_like(fN)
        
        for i in range(2, len(fN), 2):  
            AN[i] = AN[i - 2] * M2 + F * (yN[i - 2] * M2 + yN[i - 1] * M2 + yN[i])
            BN[i] = BN[i - 2] * M2 + F * (zN[i - 2] * M2 + zN[i - 1] * M1 + zN[i])
            XN[i] = AN[i] * np.sin(omegad_mode * t[i]) - BN[i] * np.cos(omegad_mode * t[i])
        
        XN_red = XN[::2]
        
        phi_mode = V[:, mode]
        
        MPF_mode = MPF[mode]

        x_modal[:, :, mode] = MPF_mode * np.outer(phi_mode, XN_red)
    
    x_total[:, :] = np.sum(x_modal[:, :, :], axis=2)
    
    t_total = t[1::2]
    dt_total = np.mean(np.diff(t_total))
    
    return x_total, x_modal, t_total, dt_total

