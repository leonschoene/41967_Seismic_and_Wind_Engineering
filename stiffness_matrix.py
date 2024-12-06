# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:27:25 2024

@author: Leon Schöne
"""

import numpy as np

def stiffness(E_c, I_beam , I_col, L_beam, L_col):
    """
    

    Parameters
    ----------
    E_c     : in [N/m^2];   Modulus of Elasticity of Concrete.
    I_beam  : in [m^4];     Second moment of inertia for the beam.
    I_col   : in [m^4];     Second moment of inertia for the column.
    L_beam  : in [m];       Length of the beam.
    L_col   : in [m];       Length of the column.

    Returns
    -------
    K : Stiffness Matrix (n by n).

    """
    ### Stiffness Coefficients
    
    # Stiffness Coefficients for beams
    k_11b = 12*E_c*I_beam/L_beam**3
    k_21b = 6*E_c*I_beam/L_beam**2
    k_31b = -12*E_c*I_beam/L_beam**3
    k_41b = 6*E_c*I_beam/L_beam**2
    
    k_12b = 6*E_c*I_beam/L_beam**2
    k_22b = 4*E_c*I_beam/L_beam
    k_32b = -6*E_c*I_beam/L_beam**2
    k_42b = 2*E_c*I_beam/L_beam
    
    k_13b = -12*E_c*I_beam/L_beam**3
    k_23b = -6*E_c*I_beam/L_beam**2
    k_33b = 12*E_c*I_beam/L_beam**3
    k_43b = -6*E_c*I_beam/L_beam**2
    
    k_14b = 6*E_c*I_beam/L_beam**2
    k_24b = 2*E_c*I_beam/L_beam
    k_34b = -6*E_c*I_beam/L_beam**2
    k_44b = 4*E_c*I_beam/L_beam
    
    # Stiffness Coefficients for column

    k_11c = 12*E_c*I_col/L_col**3
    k_21c = 6*E_c*I_col/L_col**2
    k_31c = -12*E_c*I_col/L_col**3
    k_41c = 6*E_c*I_col/L_col**2

    k_12c = 6*E_c*I_col/L_col**2
    k_22c = 4*E_c*I_col/L_col
    k_32c = -6*E_c*I_col/L_col**2
    k_42c = 2*E_c*I_col/L_col

    k_13c = -12*E_c*I_col/L_col**3
    k_23c = -6*E_c*I_col/L_col**2
    k_33c = 12*E_c*I_col/L_col**3
    k_43c = -6*E_c*I_col/L_col**2

    k_14c = 6*E_c*I_col/L_col**2
    k_24c = 2*E_c*I_col/L_col
    k_34c = -6*E_c*I_col/L_col**2
    k_44c = 4*E_c*I_col/L_col
    
    # Stiffness Matrix for beam
    '''
    Getting global stiffness by combining local stiffness coefficients 
    to a global stiffness matrix
    '''

    k_11 = 2 * (k_11c + k_33c)
    k_21 = k_31 = k_71 = k_81 = k_91 = k_101 = k_111 = k_121 = 0 
    k_41 = 2 * k_13c
    k_51 = k_23c
    k_61 = k_23c

    k_12 = k_62 = k_72 = k_82 = k_92 = k_102 = k_112 = k_122 = 0 
    k_22 = k_22c + k_44c + k_44b
    k_32 = k_24b
    k_42 = k_14c
    k_52 = k_24c

    k_13 = k_53 = k_73 = k_83 = k_93 = k_103 = k_113 = k_123 = 0
    k_23 = k_42b
    k_33 = k_22c + k_44c + k_22b
    k_43 = k_14c
    k_63 = k_24c

    k_14 = 2 * k_31c
    k_24 = k_41c
    k_34 = k_41c
    k_44 = 2 * (k_11c + k_33c)
    k_54 = k_64 = k_104 = k_114 = k_124 = 0
    k_74 = 2 * k_13c
    k_84 = k_23c
    k_94 = k_23c

    k_15 = k_32c
    k_25 = k_42c
    k_35 = k_95 = k_105 = k_115 = k_125 = 0
    k_45 = k_12c + k_34c
    k_55 = k_22c + k_44c + k_44b
    k_65 = k_24b
    k_75 = k_14c
    k_85 = k_24c

    k_16 = k_32c
    k_26 = k_46 = k_86 = k_106 = k_116 = k_126 = 0
    k_36 = k_42c
    k_56 = k_42b
    k_66 = k_22c + k_44c + k_22b
    k_76 = k_14c
    k_96 = k_24c

    k_17 = k_27 = k_37 = k_87 = k_97 = 0
    k_47 = 2 * k_31c
    k_57 = k_41c
    k_67 = k_41c
    k_77 = 2 * (k_11c + k_33c)
    k_107 = 2 * k_13c
    k_117 = k_23c
    k_127 = k_23c

    k_18 = k_28 = k_38 = k_68 = k_128 = 0
    k_48 = k_32c
    k_58 = k_42c
    k_78 = k_12c + k_34c
    k_88 = k_22c + k_44c + k_44b 
    k_98 = k_24b
    k_108 = k_14c
    k_118 = k_24c

    k_19 = k_29 = k_39 = k_59 = k_79 = k_119 = 0
    k_49 = k_32c
    k_69 = k_42c
    k_89 = k_42b
    k_99 = k_22c + k_44c + k_22b
    k_109 = k_14c
    k_129 = k_24c

    k_110 = k_210 = k_310 = k_410 = k_510 = k_610 = 0
    k_710 = 2 * k_31c
    k_810 = k_41c
    k_910 = k_41c
    k_1010 = 2 * k_11c
    k_1110 = k_21c
    k_1210 = k_21c

    k_111 = k_211 = k_311 = k_411 = k_511 = k_611 = k_911 = 0
    k_711 = k_32c
    k_811 = k_42c
    k_1011 = k_12c
    k_1111 = k_22c + k_44b
    k_1211 = k_24b

    k_112 = k_212 = k_312 = k_412 = k_512 = k_612 = k_812 = 0
    k_712 = k_32c
    k_912 = k_42c
    k_1012 = k_12c
    k_1112 = k_42b
    k_1212 = k_22c + k_22b

    K = np.array([[k_11, k_12, k_13, k_14, k_15, k_16, k_17, k_18, k_19, k_110, k_111, k_112],
                  [k_21, k_22, k_23, k_24, k_25, k_26, k_27, k_28, k_29, k_210, k_211, k_212],
                  [k_31, k_32, k_33, k_34, k_35, k_36, k_37, k_38, k_39, k_310, k_311, k_312],
                  [k_41, k_42, k_43, k_44, k_45, k_46, k_47, k_48, k_49, k_410, k_411, k_412],
                  [k_51, k_52, k_53, k_54, k_55, k_56, k_57, k_58, k_59, k_510, k_511, k_512],
                  [k_61, k_62, k_63, k_64, k_65, k_66, k_67, k_68, k_69, k_610, k_611, k_612],
                  [k_71, k_72, k_73, k_74, k_75, k_76, k_77, k_78, k_79, k_710, k_711, k_712],
                  [k_81, k_82, k_83, k_84, k_85, k_86, k_87, k_88, k_89, k_810, k_811, k_812],
                  [k_91, k_92, k_93, k_94, k_95, k_96, k_97, k_98, k_99, k_910, k_911, k_912],
                  [k_101, k_102, k_103, k_104, k_105, k_106, k_107, k_108, k_109, k_1010, k_1011, k_1012],
                  [k_111, k_112, k_113, k_114, k_115, k_116, k_117, k_118, k_119, k_1110, k_1111, k_1112],
                  [k_121, k_122, k_123, k_124, k_125, k_126, k_127, k_128, k_129, k_1210, k_1211, k_1212]])
    
    return K
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    