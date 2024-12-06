# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:45:11 2024

@author: Leon Schöne
"""

import numpy as np

def mass(rho_c, A_b, A_c, L_beam, L_col, m0_, m1_, m2_):
    """
    

    Parameters
    ----------
    rho_c   : in [kg/m^3];  Density of concrete.
    A_b     : in [m^2];     Cross-section area of beam.
    A_c     : in [m^2];     Cross-section area of column.
    L_beam  : in [m];       Length of the beam.
    L_col   : in [m];       Length of the column.
    m0_     : in [kg];      Lumped mass of first story.
    m1_     : in [kg];      Lumped mass of second and third story.
    m2_     : in [kg];      Lumped mass of roof.

    Returns
    -------
    M : Mass matrix (n by n)

    """
    
    ### Mass Coefficients

    # Mass Coefficients for long beam

    m_11b = 156*rho_c*A_b*L_beam/420
    m_21b = 22*L_beam*rho_c*A_b*L_beam/420
    m_31b = 54*rho_c*A_b*L_beam/420
    m_41b = -13*L_beam*rho_c*A_b*L_beam/420

    m_12b = 22*L_beam*rho_c*A_b*L_beam/420
    m_22b = 4*L_beam**2*rho_c*A_b*L_beam/420
    m_32b = 13*L_beam*rho_c*A_b*L_beam/420
    m_42b = -3*L_beam**2*rho_c*A_b*L_beam/420

    m_13b = 54*rho_c*A_b*L_beam/420
    m_23b = 13*L_beam*rho_c*A_b*L_beam/420
    m_33b = 156*rho_c*A_b*L_beam/420
    m_43b = -22*L_beam*rho_c*A_b*L_beam/420

    m_14b = -13*L_beam*rho_c*A_b*L_beam/420
    m_24b = -3*L_beam**2*rho_c*A_b*L_beam/420
    m_34b = -22*L_beam*rho_c*A_b*L_beam/420
    m_44b = 4*L_beam**2*rho_c*A_b*L_beam/420

    # Mass Coefficients for column

    m_11c = 156*(rho_c*A_c*L_col)/420
    m_21c = 22*L_col*(rho_c*A_c*L_col)/420
    m_31c = 54*(rho_c*A_c*L_col)/420
    m_41c = -13*L_col*(rho_c*A_c*L_col)/420

    m_12c = 22*L_col*(rho_c*A_c*L_col)/420
    m_22c = 4*L_col**2*(rho_c*A_c*L_col)/420
    m_32c = 13*L_col*(rho_c*A_c*L_col)/420
    m_42c = -3*L_col**2*(rho_c*A_c*L_col)/420

    m_13c = 54*(rho_c*A_c*L_col)/420
    m_23c = 13*L_col*(rho_c*A_c*L_col)/420
    m_33c = 156*(rho_c*A_c*L_col)/420
    m_43c = -22*L_col*(rho_c*A_c*L_col)/420

    m_14c = -13*L_col*(rho_c*A_c*L_col)/420
    m_24c = -3*L_col**2*(rho_c*A_c*L_col)/420
    m_34c = -22*L_col*(rho_c*A_c*L_col)/420
    m_44c = 4*L_col**2*(rho_c*A_c*L_col)/420

    ### Mass Matrix

    # Mass Matrix for long beam
    '''
    Getting global stiffness by combining local stiffness coefficients 
    to a global stiffness matrix
    '''

    m_11 = 2 * (m_11c + m_33c) + m0_
    m_21 = m_31 = m_71 = m_81 = m_91 = m_101 = m_111 = m_121 = 0 
    m_41 = 2 * m_13c
    m_51 = m_23c
    m_61 = m_23c

    m_12 = m_62 = m_72 = m_82 = m_92 = m_102 = m_112 = m_122 = 0 
    m_22 = m_22c + m_44c + m_44b
    m_32 = m_24b
    m_42 = m_14c
    m_52 = m_24c

    m_13 = m_53 = m_73 = m_83 = m_93 = m_103 = m_113 = m_123 = 0
    m_23 = m_42b
    m_33 = m_22c + m_44c + m_22b
    m_43 = m_14c
    m_63 = m_24c

    m_14 = 2 * m_31c
    m_24 = m_41c
    m_34 = m_41c
    m_44 = 2 * (m_11c + m_33c) + m1_
    m_54 = m_64 = m_104 = m_114 = m_124 = 0
    m_74 = 2 * m_13c
    m_84 = m_23c
    m_94 = m_23c

    m_15 = m_32c
    m_25 = m_42c
    m_35 = m_95 = m_105 = m_115 = m_125 = 0
    m_45 = m_12c + m_34c
    m_55 = m_22c + m_44c + m_44b
    m_65 = m_24b
    m_75 = m_14c
    m_85 = m_24c

    m_16 = m_32c
    m_26 = m_46 = m_86 = m_106 = m_116 = m_126 = 0
    m_36 = m_42c
    m_56 = m_42b
    m_66 = m_22c + m_44c + m_22b
    m_76 = m_14c
    m_96 = m_24c

    m_17 = m_27 = m_37 = m_87 = m_97 = 0
    m_47 = 2 * m_31c
    m_57 = m_41c
    m_67 = m_41c
    m_77 = 2 * (m_11c + m_33c) + m1_
    m_107 = 2 * m_13c
    m_117 = m_23c
    m_127 = m_23c

    m_18 = m_28 = m_38 = m_68 = m_128 = 0
    m_48 = m_32c
    m_58 = m_42c
    m_78 = m_12c + m_34c
    m_88 = m_22c + m_44c + m_44b
    m_98 = m_24b
    m_108 = m_14c
    m_118 = m_24c

    m_19 = m_29 = m_39 = m_59 = m_79 = m_119 = 0
    m_49 = m_32c
    m_69 = m_42c
    m_89 = m_42b
    m_99 = m_22c + m_44c + m_22b
    m_109 = m_14c
    m_129 = m_24c

    m_110 = m_210 = m_310 = m_410 = m_510 = m_610 = 0
    m_710 = 2 * m_31c
    m_810 = m_41c
    m_910 = m_41c
    m_1010 = 2 * m_11c + m2_
    m_1110 = m_21c
    m_1210 = m_21c

    m_111 = m_211 = m_311 = m_411 = m_511 = m_611 = m_911 = 0
    m_711 = m_32c
    m_811 = m_42c
    m_1011 = m_12c
    m_1111 = m_22c + m_44b
    m_1211 = m_24b

    m_112 = m_212 = m_312 = m_412 = m_512 = m_612 = m_812 = 0
    m_712 = m_32c
    m_912 = m_42c
    m_1012 = m_12c
    m_1112 = m_42b
    m_1212 = m_22c + m_22b

    M = np.array([[m_11, m_12, m_13, m_14, m_15, m_16, m_17, m_18, m_19, m_110, m_111, m_112],
                  [m_21, m_22, m_23, m_24, m_25, m_26, m_27, m_28, m_29, m_210, m_211, m_212],
                  [m_31, m_32, m_33, m_34, m_35, m_36, m_37, m_38, m_39, m_310, m_311, m_312],
                  [m_41, m_42, m_43, m_44, m_45, m_46, m_47, m_48, m_49, m_410, m_411, m_412],
                  [m_51, m_52, m_53, m_54, m_55, m_56, m_57, m_58, m_59, m_510, m_511, m_512],
                  [m_61, m_62, m_63, m_64, m_65, m_66, m_67, m_68, m_69, m_610, m_611, m_612],
                  [m_71, m_72, m_73, m_74, m_75, m_76, m_77, m_78, m_79, m_710, m_711, m_712],
                  [m_81, m_82, m_83, m_84, m_85, m_86, m_87, m_88, m_89, m_810, m_811, m_812],
                  [m_91, m_92, m_93, m_94, m_95, m_96, m_97, m_98, m_99, m_910, m_911, m_912],
                  [m_101, m_102, m_103, m_104, m_105, m_106, m_107, m_108, m_109, m_1010, m_1011, m_1012],
                  [m_111, m_112, m_113, m_114, m_115, m_116, m_117, m_118, m_119, m_1110, m_1111, m_1112],
                  [m_121, m_122, m_123, m_124, m_125, m_126, m_127, m_128, m_129, m_1210, m_1211, m_1212]])
    
    return M
