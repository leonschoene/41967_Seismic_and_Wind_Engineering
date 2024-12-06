# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 18:51:04 2024

@author: Leon Schöne
"""

import numpy as np

def modal_analysis(U, M, K, l): 
    """
    

    Parameters
    ----------
    U : U matrix from eigenvalue problem.
    M : Mass matrix.
    K : Stiffness matrix.
    l : length vector.

    Returns
    -------
    m_j : modal mass.
    k_j : modal stiffness.
    L_n : .
    GAMMA: .
    M_eff: effective modal mass.

    """
    m_j = np.array([])
    k_j = np.array([])
    L_n = np.array([])
    for i in range(1,13,1):
        m_j_i = U[:,i-1].T @ M @ U[:,i-1]
        k_j_i = U[:,i-1].T @ K @ U[:,i-1]
        L_n_i = U[:,i-1].T @ M @ l
        m_j = np.append(m_j, m_j_i)
        k_j = np.append(k_j, k_j_i)
        L_n = np.append(L_n, L_n_i)
        
    GAMMA = L_n/m_j
    M_eff = GAMMA*L_n
    
    return m_j, k_j, L_n, GAMMA, M_eff
