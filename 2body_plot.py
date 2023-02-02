import os
import numpy as np
import time
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

# read data
n_trials = 50
n_data = 134
nv = 4
t_ind = 0
omega_ind = 1
theta_ind = 2
phi_ind = 3

ts = np.zeros((n_trials, n_data))
omegas = np.zeros((n_trials, n_data))
thetas = np.zeros((n_trials, n_data))
phis = np.zeros((n_trials, n_data))
fps = ["./2body_equi_data/trial_"+str(i)+".txt" for i in range(n_trials)]

for i in range(n_trials):
    f = open(fps[i], 'r')
    allLines = f.readlines()
    for j in range(n_data):
        datum = allLines[j].split()
        ts[i,j] = datum[t_ind]
        omegas[i,j] = datum[omega_ind]
        thetas[i,j] = datum[theta_ind]
        phis[i,j] = datum[phi_ind]

# plot
fig, (ax1,ax2) = plt.subplots(2, 1,figsize=(5, 5), sharex=True)
plt.subplots_adjust(left=0.10, bottom=0.08, right=.98, top=0.98, wspace=0.05, hspace=0.02)
ax1.set_ylabel(r"$\omega/n$")
ax2.set_ylabel(r"$\theta$ ($^{\circ}$)")
ax2.set_xlabel("Time (P)")

n_errs = 0
for i in range(n_trials):
    if np.any(omegas[i] > 2.5):
        n_errs += 1
        continue

    ax1.plot(ts[i],omegas[i], lw=1., color='black', alpha=0.2)
    ax2.plot(ts[i],thetas[i], lw=1., color='black', alpha=0.2)

if n_errs > 0:
    print(f"Omitting {n_errs} trials with spin rates > 2.5 n")

plt.savefig('2body_trials.png', dpi=300)
plt.clf()