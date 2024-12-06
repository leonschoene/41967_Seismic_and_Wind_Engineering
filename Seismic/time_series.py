# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 22:19:04 2024

@author: lwdsc
"""

import numpy as np

# timestep = 0.0014929679096641068
# dt = timestep
# zeta = 0.03

# tN = np.loadtxt('../input/Acc.txt', usecols = 0) # Natural periods
# Acc = np.loadtxt('../input/Acc.txt', usecols = 1)

# omega = np.array([12.6517, 39.7113, 69.2859, 95.0124, 191.958, 231.142, 274.224, 309.008, 570.763, 650.982, 774.938, 888.923])

# modal_mass = np.array([74306.2, 84130.2, 81206.7, 89850.1, 20688.7, 33085, 31775.3, 31639.2, 9621.91, 7224.1, 6834.67, 6141.12])

# MPF = np.array([1.27617, -0.41031, 0.246802, -0.129381, -8.76952e-17, 6.61134e-17, 1.09892e-16, -9.33985e-17, 0.0272981, 0.0520618, 0.0419728, 0.0214452])

def DuhamelInt(t, dt, Acc, omega, m, zeta, MPF, V):
    Nred = len(t[::2])
    ndof, nmodes = V.shape
    # _, ndir = omega.shape
    
    x_modal = np.zeros((ndof, Nred, nmodes))
    x_total = np.zeros((ndof, Nred))

    # for direc in range(ndir):
    for mode in range(nmodes):
        # Berechnung der Frequenzen
        omega_mode = omega[mode]
        omegad_mode = omega_mode * np.sqrt(1 - zeta**2)
        
        # Konstanten
        # if m.shape[1] < ndir:
        #     m_mode = m[mode]
        # else:
        m_mode = m[mode]
        
        M1 = 4 * np.exp(-zeta * omega_mode * dt)
        M2 = np.exp(-2 * zeta * omega_mode * dt)
        F = dt / (3 * m_mode * omegad_mode)
        
        # Lastdefinition
        fN = m_mode * Acc
        yN = fN * np.cos(omegad_mode * t)
        zN = fN * np.sin(omegad_mode * t)
        
        # Initialisierung von AN, BN, XN
        AN = np.zeros_like(fN)
        BN = np.zeros_like(fN)
        XN = np.zeros_like(fN)
        
        # Berechnung von AN, BN und XN
        for i in range(2, len(fN), 2):  # Start bei i=2, dann 4, 6, ...
            AN[i] = AN[i - 2] * M2 + F * (yN[i - 2] * M2 + yN[i - 1] * M2 + yN[i])
            BN[i] = BN[i - 2] * M2 + F * (zN[i - 2] * M2 + zN[i - 1] * M1 + zN[i])
            XN[i] = AN[i] * np.sin(omegad_mode * t[i]) - BN[i] * np.cos(omegad_mode * t[i])
        
        # Reduzieren der Schritte auf jeden zweiten Wert
        XN_red = XN[::2]
        
        # Modalvorbereitung
        # if V.shape[2] < ndir:
        #     phi_mode = V[:, mode]
        # else:
        phi_mode = V[:, mode]
        
        # if MPF.shape[1] < ndir:
        #     MPF_mode = MPF[mode]
        # else:
        MPF_mode = MPF[mode]
        
        # Berechnung der modalen Antwort
        x_modal[:, :, mode] = MPF_mode * np.outer(phi_mode, XN_red)
    
    # Summieren der modalen Antworten
    x_total[:, :] = np.sum(x_modal[:, :, :], axis=2)
    
    # Zeitinformationen
    t_total = t[1::2]
    dt_total = np.mean(np.diff(t_total))
    
    return x_total, x_modal, t_total, dt_total

