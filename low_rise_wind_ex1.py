# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:32:45 2024

@author: Leon Schöne
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sympy import symbols, Eq, solve
import math
from generate_latex_table import GenerateLatexTable
from generate_latex_table import GenerateLatexTable3

##### FUNCTIONS ###############################################################

def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier

def plot_reduced_variate(v, extreme, storm, filename, R=1, unit='$\mathrm{kN/m^2}$'):
    N = len(v)
    v.sort()
    
    v_mean = np.mean(v)
    v_std = np.std(v)
    
    f_rel = np.arange(1, N+1)/(N+1)
    f_red = -np.log((-np.log(f_rel**R)))
    
    slope, intercept, r_value, p_value, std_err = linregress(v, f_red)
    
    # Values for exceedence probability for a 50 year event
    if extreme == 'minimum':
        y_value = float(-np.log((-np.log((1-0.78)**R))))
        # x_trend = np.linspace(min(v)+min(v)/5, max(v)+max(v)/5, 20)
        x_trend = np.linspace(min(v), max(v), 20)
    elif extreme == 'maximum':
        y_value = float(-np.log((-np.log((0.78)**R))))
        # x_trend = np.linspace(min(v)-min(v)/10, max(v)+max(v)/10, 20)
        x_trend = np.linspace(min(v), max(v), 20)
    else: 
        print("ERROR: No extreme given!")
        
    trendline = slope * x_trend + intercept
    
    # Variablen definieren
    x, y = symbols('x y')
    
    # Geradengleichungen
    eq1 = Eq(y, y_value)  # horizontal line at 0.98
    eq2 = Eq(y, slope * x + intercept)  # trendline
        
    # Schnittpunkt berechnen
    solution = solve((eq1, eq2), (x, y))
    # print(f"Intersection point: x = {solution[x]}, y = {solution[y]}")
    
    fig1 = plt.figure(figsize=(5,5.5))
    plt.scatter(v, f_red, label=f"$v_{{{storm},{extreme}}}$")
    plt.plot(x_trend, trendline, label="Trendlinie", color="red")
    plt.hlines(y_value, xmin=min(v)-abs(min(v)*0.05), xmax=solution[x], color='black', linestyles='--')
    plt.vlines(solution[x], ymin=trendline[0]-abs(trendline[0]*0.1) , ymax=y_value, color='black', linestyles='--')
    
    # plt.text(solution[x]+0.05, (trendline[0]), f" $X_{{78\\%}}$ = {solution[x]:.2f}", fontsize=10, color="black", rotation=90)
    # plt.text((solution[x]-x_trend[0])/2+x_trend[0],y_value+0.05, f"0.78 = {y_value:.4f}", fontsize=10, color="black", horizontalalignment='center')
    
    # plt.grid(which = 'both')
    plt.xlim(min(v)-abs(min(v)*0.05), ) #x_trend[0], round_up(x_trend[-1], 2)
    plt.ylim(trendline[0]-abs(trendline[0]*0.1), math.ceil(f_red[-1]))
    
    plt.xlabel(f'Wind Pressure Coefficient')
    plt.ylabel('Reduced Variate = -ln(-ln($f_{rel}$))')
    plt.legend(loc='upper left')
    plt.annotate(f"$\mu$ = {v_mean:.2f}\n$\sigma$ = {v_std:.2f}\nR = {r_value:.2f}\n\n0.78 = {y_value:.4f}\n$X_{{78\\%}}$ = {solution[x]:.2f}", xy=(0.025, 0.625), xycoords='axes fraction', color='black', bbox=dict(boxstyle="round,pad=0.2", edgecolor='lightgray', facecolor='white'))
    
    plt.show()
    fig1.savefig(f"../plots_wind/Order_Statistic_{filename}_{storm}_{extreme}.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')
    
##### PROGRAM #################################################################
  
# Parameter definition
FileName = 'cpcent.00'  # Name of input file
Nstorm = 12             # Number of sub-series
Ntap = 18               # Number of taps
length = 4096           # Number of time steps per storm

Series = np.zeros((Nstorm * length, Ntap))  # Pre-allocate space for fast data handling
qhmwk = np.zeros(Nstorm)                    # Pre-allocate for velocity pressure

# Open file for reading
with open('../input/'+FileName, 'r') as fid:
    # Loop over all 12 data blocks
    for istorm in range(1, Nstorm + 1):
        f1 = (istorm - 1) * length
        f2 = istorm * length
        qhmwk[istorm - 1] = float(fid.readline().strip())                   # Read velocity pressure [kN/m^2]
        cp = np.loadtxt([fid.readline().strip() for _ in range(length)])    # Read data block with 18 columns (= data time series)
        Series[f1:f2, :] = cp                                               # Save the data block in the Series array                   
        # print(f"Storm Number considered: {istorm} {qhmwk[istorm - 1]}")     # Print information about the storm being processed

# Check dimensions of the resulting Series array
m1, n1 = Series.shape

mean_values = np.zeros((Nstorm, Ntap))
std_values = np.zeros((Nstorm, Ntap))
min_values = np.zeros((Nstorm, Ntap))
max_values = np.zeros((Nstorm, Ntap))
# Calculate mean values and standard deviation
for i in range(0, Ntap):
    for j in range(0, Nstorm):
        mean = np.mean(Series[j*length: (j+1)*length, i])
        std = np.std(Series[j*length: (j+1)*length, i])
        min_val = min(Series[j*length: (j+1)*length, i])
        max_val = max(Series[j*length: (j+1)*length, i])
        mean_values[j,i] = mean
        std_values[j,i] = std
        min_values[j,i] = min_val
        max_values[j,i] = max_val

main_mean = np.zeros((Ntap, 2))     # mean of mean values of each storm for each signal, mean of standard deviation of each storm for each signal
for i in range(Ntap):
    mean_mean = np.mean(mean_values[:,i])
    std_mean = np.mean(std_values[:,i])
    main_mean[i,0] = mean_mean
    main_mean[i,1] = std_mean
    

fig1 = plt.figure(figsize=(10,4))
plt.plot(np.arange(1,19,1), main_mean[:,0].T, color='black')
plt.errorbar(np.arange(1,19,1), main_mean[:,0].T, yerr=main_mean[:,1].T, fmt='o', capsize=3, color='black')

plt.xlabel('Signal')
plt.ylabel('Pressure coefficient')
plt.xticks(np.arange(1,19,1), np.arange(1,19,1))
plt.show()
fig1.savefig(f"../plots_wind/Mean_values_{FileName}.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')

# Extreme value analysis "order statistic" (A.5)
interested_signals = [3, 5, 12, 16]

# for i in interested_signals:
#     v_min = np.array(min_values[:,i-1])
#     v_max = np.array(max_values[:,i-1])
#     plot_reduced_variate(v_min, "minimum", i, FileName)
#     plot_reduced_variate(v_max, "maximum", i, FileName)
    
# Fractile-Factor method (A.6)
gamma = 0.577216
p = 0.78

c_frac = np.sqrt(6)/np.pi *(-gamma-np.log(-np.log(p)))

extremes_mean = np.zeros((Ntap,4))
for i in range(Ntap):
    extremes_mean[i,0] = np.mean(min_values[:,i])
    extremes_mean[i,1] = np.mean(max_values[:,i])
    extremes_mean[i,2] = np.std(min_values[:,i])
    extremes_mean[i,3] = np.std(max_values[:,i])


X_78min = extremes_mean[:,0] - c_frac*extremes_mean[:,2]
X_78max = extremes_mean[:,1] + c_frac*extremes_mean[:,3]


# Exercise B.1
EC_pressure = np.array([1,1,1,1,0,0,0,0,0,0.2,0,0,0,0,0,0,0,0])
EC_sucction = np.array([0,0,0,0,-2,-1.2,-1.2,-1.2,-1.2,-0.6,-0.6,-0.6,-0.6,-0.6,-0.485,-0.485,-0.485,-0.485])


dev_pressure = abs((main_mean[:,0] - EC_pressure)/EC_pressure)*100  # deviation from EC
dev_sucction = abs((main_mean[:,0] - EC_sucction)/EC_sucction)*100  # deviation from EC

# Exercise B.2
EC_cpe = np.array([1,1,1,1,-2,-1.2,-1.2,-1.2,-1.2,-0.6,-0.6,-0.6,-0.6,-0.6,-0.485,-0.485,-0.485,-0.485])

q_b = 0.405
q_p = 0.982

W_EC = EC_cpe * q_p
X_78 = np.concatenate([X_78max[0:4], X_78min[4::]])
W_WT = X_78 * q_b

dev_W = abs(abs((W_WT - W_EC))/ W_EC) * 100 # deviation from EC values

##### GENERATE TABLES #########################################################

# Exercise A.2
Ex2 = np.zeros((int(4*Ntap/2), 8))
for i in range(0, int(Ntap/2)):
    Ex2[4*i,0] = np.mean(mean_values[:,i])
    Ex2[4*i+1, 0] = np.std(mean_values[:,i])
    Ex2[4*i+2, 0] = max(mean_values[:,i])
    Ex2[4*i+3, 0] = min(mean_values[:,i])
    Ex2[4*i, 1] = np.mean(std_values[:,i])
    Ex2[4*i+1, 1] = np.std(std_values[:,i])
    Ex2[4*i+2, 1] = max(std_values[:,i])
    Ex2[4*i+3, 1] = min(std_values[:,i])
    Ex2[4*i, 2] = np.mean(max_values[:,i])
    Ex2[4*i+1, 2] = np.std(max_values[:,i])
    Ex2[4*i+2, 2] = max(max_values[:,i])
    Ex2[4*i+3, 2] = min(max_values[:,i])
    Ex2[4*i, 3] = np.mean(min_values[:,i])
    Ex2[4*i+1, 3] = np.std(min_values[:,i])
    Ex2[4*i+2, 3] = max(min_values[:,i])
    Ex2[4*i+3, 3] = min(min_values[:,i])
    
    Ex2[4*i,4] = np.mean(mean_values[:,i+9])
    Ex2[4*i+1, 4] = np.std(mean_values[:,i+9])
    Ex2[4*i+2, 4] = max(mean_values[:,i+9])
    Ex2[4*i+3, 4] = min(mean_values[:,i+9])
    Ex2[4*i, 5] = np.mean(std_values[:,i+9])
    Ex2[4*i+1, 5] = np.std(std_values[:,i+9])
    Ex2[4*i+2, 5] = max(std_values[:,i+9])
    Ex2[4*i+3, 5] = min(std_values[:,i+9])
    Ex2[4*i, 6] = np.mean(max_values[:,i+9])
    Ex2[4*i+1, 6] = np.std(max_values[:,i+9])
    Ex2[4*i+2, 6] = max(max_values[:,i+9])
    Ex2[4*i+3, 6] = min(max_values[:,i+9])
    Ex2[4*i, 7] = np.mean(min_values[:,i+9])
    Ex2[4*i+1, 7] = np.std(min_values[:,i+9])
    Ex2[4*i+2, 7] = max(min_values[:,i+9])
    Ex2[4*i+3, 7] = min(min_values[:,i+9])
    
Headings = ["Smean", "Srms", "Smax", "Smin", "Smean", "Srms", "Smax", "Smin"]
format_str = [".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f"]
GenerateLatexTable(Ex2, Headings, format_str, f"Exercise_A_2_{FileName}")


# Exercise A.3
main_mean_combined = np.column_stack((np.arange(1,19), main_mean))
Headings = ["Signal", "Mean Mean", "Mean Std"]
format_str = [".0f", ".3f", ".3f"]
GenerateLatexTable(main_mean_combined, Headings, format_str, f"Exercise_A_3_{FileName}")


# Exercise A.6
fractile_combined = np.column_stack((np.arange(1,19), extremes_mean[:,1], extremes_mean[:,3], X_78max, extremes_mean[:,0], extremes_mean[:,2], X_78min))
Headings = ["Signal", "Mean max", "Std max", "X_78max", "Mean min", "Std min", "X_78min"]
format_str = [".0f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f"]
GenerateLatexTable(fractile_combined, Headings, format_str, f"Exercise_A_6_{FileName}")


# Exercise B.1
zone = np.array(['D', 'D', 'D', 'D', 'G', 'H', 'H', 'H', 'H', 'J', 'I', 'I', 'I', 'I', 'E', 'E', 'E', 'E'])
C_pe1 = np.array(['1.0', '1.0', '1.0', '1.0', '-2.0/+0.0', '-1.2/+0.0', '-1.2/+0.0', '-1.2/+0.0', '-1.2/+0.0', '+0.2/-0.6', '-0.6', '-0.6', '-0.6', '-0.6', '-0.485', '-0.485', '-0.485', '-0.485'])
pc_combined = np.column_stack((np.arange(1,19), zone, [f"{value:.3f}" for value in main_mean[:, 0]], C_pe1, [f"{value:.3f}" for value in dev_sucction], [f"{value:.3f}" for value in dev_pressure]))
Headings = ["Signal", "Zone", "Mean WT", "EC", "Deviation suction", "Deviation pressure"]
GenerateLatexTable3(pc_combined, Headings, f"Exercise_B_1_{FileName}")


# Exercise B.2
B2_combined = np.column_stack((np.arange(1,19), zone, [f"{value:.3f}" for value in W_WT], [f"{value:.3f}" for value in W_EC], [f"{value:.3f}" for value in dev_W]))
Headings = ["Signal", "Zone", "Wind Tunnel", "EC", "Deviation pressure"]
GenerateLatexTable3(B2_combined, Headings, f"Exercise_B_2_{FileName}")


##### OTHERS ##################################################################

# for i in range(1, Ntap+1):
#     # Signal selection
#     SignalNo = i                    # Number of signal to be analysed (1-18)
#     X0 = Series[:, SignalNo - 1]    # Saving selected data to input vector (adjust for zero-based indexing)
       
#     # Sampling parameters
#     Fsamp = 1600                # Sampling frequency
#     DT = 1 / Fsamp              # Time step calculation
    
#     # Generate time axis
#     TAx = np.arange(0, m1) * DT
    
    
#     # # Optional: Ausgabe prüfen
#     # print(f"Series shape: {Series.shape}")
#     # print(f"Time axis length: {len(TAx)}")
    
#     fig1 = plt.figure(figsize=(15,5))
#     plt.plot(TAx, X0, linewidth=0.35)
#     plt.show()