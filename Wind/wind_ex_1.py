# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:47:27 2024

@author: Leon Schöne
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
from matplotlib.ticker import LogFormatter
from scipy.stats import linregress
from sympy import symbols, Eq, solve
from generate_latex_table import GenerateLatexTable3

### Given Winddata

u_12 = np.array([19.97, 20.17, 19.41, 19.94, 20.63, 21.97, 19.92, 22.99, 23.60, 19.90, 20.04, 22.39, 20.01, 23.40, 20.75, 23.27, 21.78, 21.01, 21.3, 21.05])
u_12_sort = sorted(u_12)
N = len(u_12)
R = 1

u_12_mean = np.mean(u_12_sort)
u_12_std = np.std(u_12_sort)

f_rel = np.arange(1, N+1)/(N+1)
f_red = -np.log((-np.log(f_rel**R)))

combined = np.column_stack((f_rel*1000, u_12_sort))

slope, intercept, r_value, p_value, std_err = linregress(u_12_sort, f_red)

x_trend = np.arange(18.5, 27, 0.5)
trendline = slope * x_trend + intercept

# Values for exceedence probability for a 50 year event
y_value = float(-np.log((-np.log(0.98**R))))

# Define variables
x, y = symbols('x y')

# Line equation
eq1 = Eq(y, y_value)  # horizontal line at 0.98
eq2 = Eq(y, slope * x + intercept)  # trendline

# Calculate intersection point
solution = solve((eq1, eq2), (x, y))
print(f"Intersection point: x = {solution[x]}, y = {solution[y]}")


fig1 = plt.figure(figsize=(5,8))
plt.scatter(u_12_sort, f_red)
plt.plot(x_trend, trendline, label="Trendlinie", color="red")
plt.hlines(y_value, xmin=18.5, xmax=solution[x]+0.5, color='black', linestyles='--')
plt.vlines(solution[x], ymin=trendline[0] , ymax=y_value+0.3, color='black', linestyles='--')

plt.text(solution[x]+0.1, 1, f"{solution[x]:.2f}", fontsize=10, color="black")
plt.text(20,y_value+0.1, f"0.98 = {y_value:.4f}", fontsize=10, color="black")

plt.text(22,-1, f"$\mu$ = {u_12_mean:.2f} m/s\n$\sigma$ = {u_12_std:.2f} m/s", color='black')

# plt.grid(which = 'both')
plt.xlim(18.5,27)
plt.ylim(trendline[0],5)

plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Reduced Variate = -ln(-ln($f_{rel}$))')
plt.show()
fig1.savefig(f"../plots_wind/char_wind_speed_vertical.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')


fig2 = plt.figure(figsize=(8, 5))

# Plot points and trendline
plt.scatter(f_red, u_12_sort)
plt.plot(trendline, x_trend, label="Trendlinie", color="red")

# Adjust horizontal and vertical lines
plt.vlines(y_value, ymin=18.5, ymax=solution[x]+0.5, color='black', linestyles='--')
plt.hlines(solution[x], xmin=trendline[0], xmax=y_value+0.3, color='black', linestyles='--')

# Adjust text
plt.text(1, solution[x] + 0.1, f"{solution[x]:.2f}", fontsize=10, color="black")
plt.text(y_value + 0.1, 20, f"0.98 = {y_value:.4f}", fontsize=10, color="black", rotation=90)

plt.text(-1, 22, f"$\mu$ = {u_12_mean:.2f} m/s\n$\sigma$ = {u_12_std:.2f} m/s", color='black')

# Set axis limit
plt.xlim(trendline[0], 5)
plt.ylim(18.5, 27)

# Change axis label
plt.xlabel('Reduced Variate = -ln(-ln($f_{rel}$))')
plt.ylabel('Wind Speed [m/s]')

plt.show()
fig2.savefig(f"../plots_wind/char_wind_speed_horizontal.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')


########## GUMBEL DISTRIBUTION MAX ##########
xx = np.linspace(18, 28, 500)

gamma = 0.577216
alpha = np.pi/(u_12_std*np.sqrt(6))
u= u_12_mean-gamma/alpha

pdf = gumbel_r.pdf(xx, loc=u, scale=alpha)  # PDF
cdf = gumbel_r.cdf(xx, loc=u, scale=alpha)  # CDF

f_gumbelr = alpha * np.exp(-alpha*(xx-u)-np.exp(-alpha*(xx-u)))
F_gumbelr = np.exp(-np.exp(-alpha*(xx-u)))

# Plotten
fig3 = plt.figure(figsize=(5, 3))

# PDF
plt.plot(xx, f_gumbelr, label="PDF", color="blue")
plt.vlines(u_12_mean, 0, max(f_gumbelr), color='red', label='Mean')
plt.xlabel("Wind Speed [m/s]")
plt.ylabel("Probability")
plt.legend()
plt.xlim(xx[0],)
plt.ylim(0,)
plt.show()
fig3.savefig(f"../plots_wind/Ex1_PDF.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')

# CDF
fig4 = plt.figure(figsize=(5, 3))
plt.plot(xx, F_gumbelr, label="CDF", color="blue")
plt.vlines(u_12_mean, 0, max(F_gumbelr), color='red', label='Mean')
plt.xlabel("Wind Speed [m/s]")
plt.ylabel("Probability")
plt.legend()
plt.xlim(xx[0],)
plt.ylim(0,)
plt.show()
fig4.savefig(f"../plots_wind/Ex1_CDF.pdf", format = 'pdf', dpi = 1200, bbox_inches = 'tight')



########## GENERATING TABLES ###########

combined = np.column_stack(([f"{value:.0f}" for value in np.arange(1,21,1)], u_12))
Headings = ["Year", "Wind Speed [m/s]"]
GenerateLatexTable3(combined, Headings, "Chracteristic_Wind_Speed_Data")


combined = np.column_stack((u_12_sort, [f"{value:.0f}" for value in np.arange(1,21,1)], [f"{value:.3f}" for value in f_rel], [f"{value:.3f}" for value in f_red]))
Headings = ["Sorted ascending [m/s]", "Rank", "$f_{rel}$", "$f_{red}$"]
GenerateLatexTable3(combined, Headings, "Chracteristic_Wind_Speed_order_statisitic_data")

