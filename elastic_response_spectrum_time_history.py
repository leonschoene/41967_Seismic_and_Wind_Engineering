# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:15:26 2024

@author: Leon Schöne
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

timestep = 0.005    # in seconds

g = 9.81
a_gR = 0.35 * g
gamma_I = 1.4
a_g = gamma_I * a_gR
S = 1.4

raw_data = pd.read_csv('../PALMSPR_MVH045.TXT', delim_whitespace=True, header=None)
df = pd.DataFrame(raw_data.values.flatten(), columns=['PGA']).dropna()

scale = a_g * S
df['PGA_scaled'] = df['PGA'] * scale
df['PGA_normalized'] = df['PGA_scaled'] / max(df['PGA'])

t = np.arange(0, len(df)*timestep, timestep)

df.insert(0, 'Time', t)

x = df['Time']
y = df['PGA_normalized']

plt.plot(x,y)
plt.show()





timestep = 0.0014929679096641068
dt = timestep
zeta = 0.03

tN = np.loadtxt('../input/Acc.txt', usecols = 0) # Natural periods
Acc = np.loadtxt('../input/Acc.txt', usecols = 1)


Tn = np.linspace(0.01, 4, 500)

omega = 2*np.pi/Tn 
omegad = omega*np.sqrt(1-zeta**2)


M1 = 4 * np.exp(-zeta * omega * dt)
M2 = np.exp(-2 * zeta * omega * dt)
F = dt / (3 * omegad)

fN = Acc
yN = np.zeros((np.size(fN)))
zN = np.zeros((np.size(fN)))
xN_spectra = np.zeros((np.size(Tn)))
for T in range(0, np.size(Tn)):
    for i in range(0,np.size(fN)):
        yN[i] = fN[i]*np.cos(omegad[T]*tN[i])
        zN[i] = fN[i]*np.sin(omegad[T]*tN[i])
    
    AN = np.zeros((np.size(fN)))
    BN = np.zeros((np.size(fN)))
    xN = np.zeros((np.size(fN)))
    
    for i in range(3, np.size(fN) + 1, 2):
        AN[i-1] = AN[i-3]*M2[T] + F[T]*(yN[i-3]*M2[T] + yN[i-2]*M1[T] + yN[i-1])
        BN[i-1] = BN[i-3]*M2[T] + F[T]*(zN[i-3]*M2[T] + zN[i-2]*M1[T] + zN[i-1])
        xN[i-1] = AN[i-1]*np.sin(omegad[T]*tN[i-1]) - BN[i-1]*np.cos(omegad[T]*tN[i-1])
        
    # Extract non-zero xN
    xN_red = xN[::2]
    tN_red = tN[::2]
    
    xN_spectra[T] = max(abs(xN_red))
    
np.savetxt("../output/xN_spectra.txt", xN_spectra)



# Spectral acceleration
aN_spectra = np.zeros((np.size(Tn)))
for i in range(0, np.size(Tn)):
    aN_spectra[i] = xN_spectra[i]*((2*np.pi/Tn[i])**2)
fig2 = plt.figure(figsize=(10,6))
plt.plot(Tn, aN_spectra)

array = np.zeros((np.size(aN_spectra), 2))
array[:,0] = Tn
array[:,1] = aN_spectra

np.savetxt("../output/aN_spectra.txt", array)





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    