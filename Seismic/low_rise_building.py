# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:30:50 2024

@author: Leon Schöne
"""

import numpy as np
import scipy as sp
import sys as sys
import matplotlib.pyplot as plt
from scipy import linalg as linalg
import pandas as pd

from stiffness_matrix import stiffness
from mass_matrix import mass
from modal_analysis import modal_analysis
from mode_shape_plot import plot_mode_shape
from mode_shape_plot import plot_mode_shape_without_annotations
from modal_response_spectrum_analysis import plot_horizontal_elastic_response_spectrum
from modal_response_spectrum_analysis import plot_design_spectrum_elastic_analysis
from modal_response_spectrum_analysis import get_spectral_displacement
from modal_response_spectrum_analysis import get_SRSS
from modal_response_spectrum_analysis import get_CQC3
from sectional_forces import get_sectional_forces
from sectional_forces import get_sectional_forces_3
from sectional_forces import get_SF_SRSS
from sectional_forces import get_SF_CQC
from generate_latex_table import GenerateLatexTable
from generate_latex_table import GenerateLatexTable2
from sectional_forces import plot_sectionforces
from sectional_forces import plotdiagram
from time_series import DuhamelInt


##### INPUT DATA #####

# Building Dimensions

W_building = 9              # m; building width
D_building = 7.5            # m; building depth
H_story = 3.5               # m; story height
w1 = w2 = w3 = w4 = 0.5     # m; width of column section at each story
d1 = d2 = d3 = d4 = 0.5     # m; depth of column section at each story
h1 = h2 = h3 = h4 = 0.8     # m; height of beam sections at each storey level
t1 = t2 = t3 = t4 = 0.35    # m; width of beam sections at each storey level
slab = 0.18                 # m; slab thickness
t_f = 0.02                  # m; thickness of glass facade

### Geometry Properties

I_beam = t1*h1**3 / 12      # m^4; second moment of inertia of the beam
I_col = w1*d1**3 / 12       # m^4; second moment of inertia of the column

L_beam_l = W_building - (2*w1/2)   # m; lenght of long beam
L_beam_s = D_building - (2*d1/2)   # m; length of short beam
L_col = H_story                    # m; lenght of column

A_c = w1*d1         # m^2; cross-section area of column
A_b = h1*t1         # m^2; cross-section area of beam

# Material Properties

E_c = 3.4e10      # N/m^2; Young’s Modulus of Elasticity for Concrete Material
rho_c = 2.4e3     # kg/m^3; Mass Density of Concrete Material --- grade C25/30
f_yk = 5e8        # N/m^2; Steel grade C
rho_gl = 2.6e3    # kg/m^3; Mass Density of Glass Material

# Vertical Loads

DL1 = 4e3     # N/m^2; Dead load at first three story levels
DL2 = 2.5e3   # N/m^2; Dead load at the roof level
LL = 2e3      # N/m^2; Live load at all storey levels

# Other Parameter

damp_ratio = 0.03   # damping ratio

# Response Spectra

g = 9.81
a_gR = 0.35 * g
gamma_I = 1.4
a_g = gamma_I * a_gR
S = 1.4
q = 4
 


##### MASS AND LOAD CALCULATION #####

ms = rho_c*slab*(W_building-w1)*(D_building-d1)/2   # kg; mass of half slab

# Lumped loads per story 
story1 = (DL1 + 0.2*LL)*(W_building-w1)*(D_building-d1)/2 * 1/9.81   # kg; half load of story transferred from N to kg
story2 = (DL2 + 0.2*LL)*(W_building-w1)*(D_building-d1)/2 * 1/9.81   # kg; half load of story transferred from N to kg

# Facade
A_f = 4*H_story*2*((W_building-w1)+(D_building-d1))     # m^2; Area of total facade
V_f = A_f*t_f                                           # m^3; Volume of glass facade
mf = V_f*rho_gl/2/4                                     # kg; half mass of glass facade per story

A_ff = 2*(W_building+D_building)/2*t_f

# Additional masses that are not included in mass per length (Matrix below)
mbs = rho_c*A_b*(D_building-w1)  # kg; mass of short beams
mbl = rho_c*A_b*(W_building-w1)  # kg; mass of long beams

m0_s = story1 + ms + mbl + mf/2
m1_s = story1 + ms + mbl + mf
m2_s = story2 + ms + mbl + mf/2

m0_l = story1 + ms + mbs + mf/2
m1_l = story1 + ms + mbs + mf
m2_l = story2 + ms + mbs + mf/2


# Total mass
tot_l_beam = 2*(L_beam_l+L_beam_s)*4
tot_l_col = 4*4*H_story

m_tot_structure = rho_c * (A_c*tot_l_col + A_b*tot_l_beam)
m_tot_others = 4*ms + 4*mf + 3*story1 + story2 + 8*(mbs+mbl)

M_total = (m_tot_structure + m_tot_others)/2



##### CALACULATING STIFFNESS AND MASS MATRIX #####

# Stiffness matrix for long (l) and short (s) building side
K_l = stiffness(E_c, I_beam , I_col, L_beam_l, L_col)
K_s = stiffness(E_c, I_beam , I_col, L_beam_s, L_col)

# Mass matrix for long (l) and short (s) building side
M_l = mass(rho_c, A_b, A_c, L_beam_l, L_col, m0_l, m1_l, m2_l)
M_s = mass(rho_c, A_b, A_c, L_beam_s, L_col, m0_s, m1_s, m2_s)



##### VIBRATION ANALYSIS #####

# Generalized eigenvalue problem
(D_l,U_l) = linalg.eig(K_l,M_l)
(D_s,U_s) = linalg.eig(K_s,M_s)

# Natural frequencies from eigenvalues
omegan_l = np.sqrt(D_l).real
omegan_s = np.sqrt(D_s).real

# Sort frequencies and mode shapes
iw_l = np.argsort(omegan_l)
omegan_l = omegan_l[iw_l]
freq_l  = omegan_l/(2*np.pi)
U_l = U_l[:, iw_l]
period_l = 1/freq_l

iw_s = np.argsort(omegan_s)
omegan_s = omegan_s[iw_s]
freq_s  = omegan_s/(2*np.pi)
U_s = U_s[:,iw_s]
period_s = 1/freq_s

# Normalization of Mode Shapes
def get_normalized_mode_shapes(U):
    # Finding maximum absolut value and normalize to maximal original value 
    U_norm = np.empty((12, 0))
    for mode in range(12):                          
        MVec = max(U[:,mode])
        mVec = min(U[:,mode])
        
        if abs(MVec) > abs(mVec):
            mxVec = MVec
        else:
            mxVec = mVec
            
        U_norm_mode = (U[:,mode]/mxVec).reshape(-1, 1)
        U_norm = np.hstack((U_norm, U_norm_mode))
    
    return U_norm

U_l_norm = get_normalized_mode_shapes(U_l)
U_s_norm = get_normalized_mode_shapes(U_s)

# Calculating modal mass and stiffness
l = np.array([1,0,0,1,0,0,1,0,0,1,0,0])

m_j_l, k_j_l, L_n_l, GAMMA_l, M_eff_l = modal_analysis(U_l_norm, M_l, K_l, l)
m_j_s, k_j_s, L_n_s, GAMMA_s, M_eff_s = modal_analysis(U_s_norm, M_s, K_s, l)



##### POST PROCESSING - PLOTTING #####

# Geometry

# Coordinate matrix [x, y, z]
X_l = np.array([[0,          0,         0],
                [0,          H_story,   0],
                [0,          2*H_story, 0],
                [0,          3*H_story, 0],
                [0,          4*H_story, 0],
                [W_building, 0,         0],
                [W_building, H_story,   0],
                [W_building, 2*H_story, 0],
                [W_building, 3*H_story, 0],
                [W_building, 4*H_story, 0]])

X_s = np.array([[0,          0,         0],
                [0,          H_story,   0],
                [0,          2*H_story, 0],
                [0,          3*H_story, 0],
                [0,          4*H_story, 0],
                [D_building, 0,         0],
                [D_building, H_story,   0],
                [D_building, 2*H_story, 0],
                [D_building, 3*H_story, 0],
                [D_building, 4*H_story, 0]])

# Topology matrix [node 1, node 2]
IX = np.array([[1, 2],
               [2, 3],
               [3, 4],
               [4, 5],
               [6, 7],
               [7, 8],
               [8, 9],
               [9, 10],
               [2, 7],
               [3, 8],
               [4, 9],
               [5, 10]])
# Boundary condition [node, dof]
bound = np.array([[1, 1],
                  [1, 2],
                  [6, 1],
                  [6, 2]])
    
mode = 2
dofs = 12
modes = dofs

# plot_mode_shape(mode, freq_l, omegan_l, U_l, X_l, IX, bound, 'long')
# plot_mode_shape(mode, freq_s, omegan_s, U_s, X_s, IX, bound, 'short')

# Plotting for combined figure, therefore without annotations
# for mode in range(1, modes+1):
#     plot_mode_shape_without_annotations(mode, freq_l, omegan_l, U_l, X_l, IX, bound, 'long')
#     plot_mode_shape_without_annotations(mode, freq_s, omegan_s, U_s, X_s, IX, bound, 'short')



##### MODAL RESPONSE SPECTRUM ANALYSIS #####

# Get an array with period and acceleration from Eurocode
S_e = plot_horizontal_elastic_response_spectrum()
S_d = plot_design_spectrum_elastic_analysis()

# Get spectral displacement for Eurocode response    
S_De_l = get_spectral_displacement(S_e, period_l, dofs)
S_De_s = get_spectral_displacement(S_e, period_s, dofs)
S_Dd_l = get_spectral_displacement(S_d, period_l, dofs)
S_Dd_s = get_spectral_displacement(S_d, period_s, dofs)

# Calculate modal displacement for elastic and design spectrum
u_e_l = pd.DataFrame().rename_axis("DOFs")
u_d_l = pd.DataFrame().rename_axis("DOFs")
u_e_s = pd.DataFrame().rename_axis("DOFs")
u_d_s = pd.DataFrame().rename_axis("DOFs")

for mode_shape in range(1, dofs+1):
    u_e_l_i = U_l_norm[:,mode_shape-1]*GAMMA_l[mode_shape-1]*S_De_l[mode_shape-1]
    u_d_l_i = U_l_norm[:,mode_shape-1]*GAMMA_l[mode_shape-1]*S_Dd_l[mode_shape-1]
    u_e_l.insert(mode_shape-1, f"Mode {mode_shape}", u_e_l_i, False)
    u_d_l.insert(mode_shape-1, f"Mode {mode_shape}", u_d_l_i, False)    
    
    u_e_s_i = U_s_norm[:,mode_shape-1]*GAMMA_s[mode_shape-1]*S_De_s[mode_shape-1]
    u_d_s_i = U_s_norm[:,mode_shape-1]*GAMMA_s[mode_shape-1]*S_Dd_s[mode_shape-1]
    u_e_s.insert(mode_shape-1, f"Mode {mode_shape}", u_e_s_i, False)
    u_d_s.insert(mode_shape-1, f"Mode {mode_shape}", u_d_s_i, False)
    
    
u_e_l.index = u_e_l.index + 1
u_d_l.index = u_d_l.index + 1
u_e_s.index = u_e_s.index + 1
u_d_s.index = u_d_s.index + 1

# Calculate total deformations with SRSS and CQC method
  
SRSS_el = get_SRSS(u_e_l, dofs, modes)*q
SRSS_dl = get_SRSS(u_d_l, dofs, modes)*q
SRSS_es = get_SRSS(u_e_s, dofs, modes)*q 
SRSS_ds = get_SRSS(u_d_s, dofs, modes)*q  

CQC_el = get_CQC3(u_e_l, damp_ratio, omegan_l, dofs, modes)*q
CQC_dl = get_CQC3(u_d_l, damp_ratio, omegan_l, dofs, modes)*q
CQC_es = get_CQC3(u_e_s, damp_ratio, omegan_s, dofs, modes)*q
CQC_ds = get_CQC3(u_d_s, damp_ratio, omegan_s, dofs, modes)*q

# Calculate sectional forces 

# Modal sectional forces. Calculated for each mode/element
sectional_forces_l = get_sectional_forces_3(E_c, I_beam, I_col, L_beam_l, L_col, u_d_l)
sectional_forces_s = get_sectional_forces_3(E_c, I_beam, I_col, L_beam_s, L_col, u_d_s)

# Total sectional forces by using combination methods SRSS and CQC
elements = 12

SF_SRSS_l = get_SF_SRSS(sectional_forces_l, elements)
SF_SRSS_s = get_SF_SRSS(sectional_forces_s, elements)

SF_CQC_l = get_SF_CQC(sectional_forces_l, damp_ratio, omegan_l, elements)
SF_CQC_s = get_SF_CQC(sectional_forces_s, damp_ratio, omegan_s, elements)

# Combination of long side and short side
# Since CQC is larger, just CQC is used for the equivalent static forces

SF_comb_l = SF_CQC_l + 0.3*SF_CQC_s
SF_comb_s = SF_CQC_s + 0.3*SF_CQC_l



##### TIME HISTORY ANALYSIS #####

timestep = 0.005    # in seconds


# Importing raw acceleration data and fit it in one column
raw_data = pd.read_csv('../PALMSPR_MVH045.TXT', sep='\s+', header=None)                 # new version for whitespace as seperator
time_history = pd.DataFrame(raw_data.values.flatten(), columns=['PGA']).dropna()

# Scale raw data 
scale = a_g * S
time_history['PGA_scaled'] = time_history['PGA'] * scale
time_history['PGA_normalized'] = time_history['PGA_scaled'] / max(time_history['PGA'])

t = np.arange(0, len(time_history)*timestep, timestep)

time_history.insert(0, 'Time', t)

x = time_history['Time']
y = time_history['PGA_normalized']

idx_max_PGA = abs(time_history['PGA_normalized']).idxmax().item()
max_PGA = time_history['PGA_normalized'].loc[idx_max_PGA].item()
max_PGA_time = time_history['Time'].loc[idx_max_PGA].item()

# Plot PGA time history
fig1 = plt.figure(figsize=(5,3))
plt.plot(x,y, linewidth = 0.5)
plt.plot(max_PGA_time, max_PGA, color = 'r', marker = 'o', fillstyle = 'none', markeredgewidth=0.5)
plt.xlim(0,)
plt.xlabel('Time [s]')
plt.ylabel('Ground acceleration [m/s²]')
plt.text(max_PGA_time+0.5, max_PGA, f'{max_PGA:.4f}', ha='left', va='center', color='black', fontsize=10)
plt.show()
fig1.savefig(f"../plots/time_history.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')



# Load period and spectral acceleration in array for plotting
# Text file is generated in elastic_response_spectrum_time_history.py
Tn_plot = np.loadtxt('../output/aN_spectra.txt', usecols = 0)
xN_plot = np.loadtxt('../output/aN_spectra.txt', usecols = 1)

fig2 = plt.figure(figsize=(5,3))
plt.plot(Tn_plot, xN_plot/a_g, linewidth = 0.5)
plt.xlim(0,)
plt.ylim(0,)
plt.xlabel('Period [s]')
plt.ylabel('$S_e$/$a_g$')
fig2.savefig(f"../plots/elastic_response_spectrum_time_history.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')
plt.show()

T = np.arange(0, 4.01, 0.01)
fig3 = plt.figure(figsize=(5,3))
plt.plot(Tn_plot, xN_plot/a_g, label = 'Spec. Acc. THA', linewidth = 0.5)
plt.plot(T, S_e[:,1]/a_g, label = 'Spec. Acc. EC 8', linewidth = 0.5)
plt.xlim(0,)
plt.ylim(0,)
plt.xlabel('Period [s]')
plt.ylabel('$S_e$/$a_g$')
plt.legend()
fig3.savefig(f"../plots/elastic_response_spectrum_comparison.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')
plt.show()



# Load period and spectral acceleration from time history in one array 
S_th = np.loadtxt('../output/aN_spectra.txt')
x_N = np.loadtxt('../output/xN_spectra.txt')
th_response = np.column_stack((S_th[:,0], x_N))

timestep = 0.0014929679096641068
dt = timestep
zeta = 0.03

tN = np.loadtxt('../input/Acc.txt', usecols = 0) # Natural periods
Acc = np.loadtxt('../input/Acc.txt', usecols = 1)

x_total_l, x_modal_l, t_total_l, dt_total_l = DuhamelInt(tN, dt, Acc, omegan_l, m_j_l, zeta, GAMMA_l, U_l_norm)
x_total_s, x_modal_s, t_total_s, dt_total_s = DuhamelInt(tN, dt, Acc, omegan_s, m_j_s, zeta, GAMMA_s, U_s_norm)

total_disp_thl = np.zeros(12)
total_disp_ths = np.zeros(12)
for i in range(modes):
    total_disp_thl[i] = max(abs(x_total_l[i,:]))*q
    total_disp_ths[i] = max(abs(x_total_s[i,:]))*q

fig4 = plt.figure(figsize=(8,3))
for i in [0,3,6,9]:
    plt.plot(t_total_l, q*x_total_l[i,:], linewidth=0.2, label=f"DOF {i+1}")
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m]')
plt.xlim(0,)
plt.legend()
plt.show()
fig4.savefig(f"../plots/time_series_horizontal_DOFs_long.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')

fig5 = plt.figure(figsize=(8,3))
for i in [0,3,6,9]:
    plt.plot(t_total_s, q*x_total_s[i,:], linewidth=0.2, label=f"DOF {i+1}")
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m]')
plt.xlim(0,)
plt.legend()
plt.show()
fig5.savefig(f"../plots/time_series_horizontal_DOFs_short.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')


# Comparison of MRSA and THA
acc_th_l = []
acc_th_s = []
mrsa = []
number_periods = 4
for i in range(0, number_periods):
    idx_th = np.abs(Tn_plot - period_l[i]).argmin()
    acc_th_i = xN_plot[idx_th]/a_g
    acc_th_l.append(acc_th_i)
    idx_th = np.abs(Tn_plot - period_s[i]).argmin()
    acc_th_i = xN_plot[idx_th]/a_g
    acc_th_s.append(acc_th_i)
    idx_mrsa = np.abs(S_e[:,0] - period_l[i]).argmin()
    mrsa_i = float(S_e[idx_mrsa,1]/a_g)
    mrsa.append(mrsa_i)
    


T = np.arange(0, 4.01, 0.01)
fig3 = plt.figure(figsize=(5,3))
plt.plot(Tn_plot, xN_plot/a_g, label = 'Spec. Acc. THA', linewidth = 0.5)
plt.plot(T, S_e[:,1]/a_g, label = 'Spec. Acc. EC 8', linewidth = 0.5)
plt.scatter(period_l[0:number_periods], acc_th_l, color = 'r', marker = 'o', linewidth=0.5, facecolors='none', s=40, label='THA spec. acc. long')
plt.scatter(period_l[0:number_periods], acc_th_s, color = 'g', marker = 'o', linewidth=0.5, facecolors='none', s=40, label='THA spec. acc. short')
plt.scatter(period_l[0:number_periods], mrsa, color = 'b', marker = 'o', linewidth=0.5, facecolors='none', s=40, label='MRSA spec. acc.')
plt.xlim(0,)
plt.ylim(0,)
plt.xlabel('Period [s]')
plt.ylabel('$S_e$/$a_g$')
plt.legend()
fig3.savefig(f"../plots/response_spectrum_comparison_periods.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')
plt.show()



##### Table Printing #####

GenerateLatexTable(S_th, ["Period", "Spectral Acceleration"], ["2.4f", "2.4f"], "Spectral_acceleration_th" )

Headings = ["Element 1", "Element 2", "Element 3", "Element 4", "Element 5", "Element 6", "Element 7", "Element 8", "Element 9", "Element 10", "Element 11", "Element 12"]
format_str = [".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f"]

GenerateLatexTable(SF_CQC_l.to_numpy(), Headings, format_str, "Section_forces_CQC_l" )
GenerateLatexTable(SF_CQC_s.to_numpy(), Headings, format_str, "Section_forces_CQC_s" )
GenerateLatexTable(SF_SRSS_l.to_numpy(), Headings, format_str, "Section_forces_SRSS_l" )
GenerateLatexTable(SF_SRSS_s.to_numpy(), Headings, format_str, "Section_forces_SRSS_s" )


combined_long = np.column_stack((np.arange(1, 13), SRSS_el, CQC_el, SRSS_dl, CQC_dl ))
combined_short = np.column_stack((np.arange(1, 13), SRSS_es, CQC_es, SRSS_ds, CQC_ds))
Headings = ["DOF", "SRSS", "CQC", "SRSS", "CQC"]
format_str = [".0f", ".4f", ".4f", ".4f", ".4f"]
GenerateLatexTable(combined_long, Headings, format_str, "Deformation_long")
GenerateLatexTable(combined_short, Headings, format_str, "Deformation_short")


frequencies_periods = np.column_stack((omegan_l, omegan_s, freq_l, freq_s, period_l, period_s))
Headings = ["long side", "short side", "long side", "short side", "long side", "short side"]
format_str = [".2f", ".2f", ".2f", ".2f", ".2f", ".2f"]
GenerateLatexTable(frequencies_periods, Headings, format_str, "frequencies_periods")


Headings = ["DOF", "Mode 1", "Mode 2", "Mode 3", "Mode 4", "Mode 5", "Mode 6", "Mode 7", "Mode 8", "Mode 9", "Mode 10", "Mode 11", "Mode 12"]
format_str = [".0f", ".4f", "0.4f", ".4f", ".4f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f"]
GenerateLatexTable(u_d_l.reset_index().to_numpy(), Headings, format_str, "modal_displacements_design_l" )
GenerateLatexTable(u_e_l.reset_index().to_numpy(), Headings, format_str, "modal_displacements_elastic_l" )
GenerateLatexTable(u_d_s.reset_index().to_numpy(), Headings, format_str, "modal_displacements_design_s" )
GenerateLatexTable(u_e_s.reset_index().to_numpy(), Headings, format_str, "modal_displacements_elastic_s" )


Headings = ["Element 1", "Element 2", "Element 3", "Element 4", "Element 5", "Element 6", "Element 7", "Element 8", "Element 9", "Element 10", "Element 11", "Element 12"]
format_str = [".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f"]
GenerateLatexTable(sectional_forces_l.to_numpy(), Headings, format_str, "Modal_Section_forces__l" )
GenerateLatexTable(sectional_forces_s.to_numpy(), Headings, format_str, "Modal_Section_forces__s" )


combined_th = np.column_stack((np.arange(1, 13), total_disp_thl, total_disp_ths))
Headings = ["DOF", "long side", "short side"]
format_str = [".0f", ".4f", ".4f"]
GenerateLatexTable(combined_th, Headings, format_str, "total_displacements_time_history")


combined_displacements = np.column_stack((np.arange(1, 13), SRSS_el, CQC_el, total_disp_thl, SRSS_es, CQC_es, total_disp_ths))
Headings = ["DOF", "SRSS", "CQC", "THA", "SRSS", "CQC", "THA"]
format_str = [".0f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f"]
GenerateLatexTable(combined_displacements, Headings, format_str, "Comparison_displacements")


combined_eff_M_Gamma = np.column_stack((np.arange(1,13), GAMMA_l, M_eff_l, GAMMA_s, M_eff_s))
Headings = ["Mode", "Gamma", "M_eff", "Gamma", "M_eff"]
format_str = [".0f", ".4f", ".4f", ".4f", ".4f"]
GenerateLatexTable(combined_eff_M_Gamma, Headings, format_str, "Effective_masses_Gamma")


Headings = ["Force", "Element 1", "Element 2", "Element 3", "Element 4", "Element 5", "Element 6", "Element 7", "Element 8", "Element 9", "Element 10", "Element 11", "Element 12"]
format_str = [".0f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f"]
combined_SF_loads_l = np.column_stack((np.arange(1,5), SF_comb_l))
combined_SF_loads_s = np.column_stack((np.arange(1,5), SF_comb_s))
GenerateLatexTable(combined_SF_loads_l, Headings, format_str, "SF_load_l")
GenerateLatexTable(combined_SF_loads_s, Headings, format_str, "SF_load_s")


























