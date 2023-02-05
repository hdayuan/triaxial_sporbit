import os
import numpy as np
import time
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=11)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

# read data
n_trials = 30
n_data = 200
nv = 4
t_ind = 0
omega_ind = 1
theta_ind = 2
phi_ind = 3

ts = np.zeros((n_trials, n_data))
omegas = np.zeros((n_trials, n_data))
thetas = np.zeros((n_trials, n_data))
phis = np.zeros((n_trials, n_data))
fps = ["./3bd_20i_1e-3j2_100Q_0.025dt/trial_"+str(i)+".txt" for i in 
range(n_trials)]

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
fig, (ax1,ax2,ax3) = plt.subplots(3, 1,figsize=(5, 8), sharex=True)
plt.subplots_adjust(left=0.15, bottom=0.1, right=.98, top=0.98, wspace=0.05, hspace=0.04)
ax1.set_ylabel(r"$\omega/n$")
ax2.set_ylabel(r"$\theta$ ($^{\circ}$)")
ax3.set_ylabel(r"$\phi$ ($^{\circ}$)")
ax3.set_xlabel("Time (P)")

n_errs = 0
for i in range(n_trials):
    if np.any(omegas[i] > 2.5):
        n_errs += 1
        continue

    ax1.plot(ts[i],omegas[i], lw=1., color='black', alpha=0.2)
    ax2.plot(ts[i],thetas[i], lw=1., color='black', alpha=0.2)
    if i == 0:
        ax3.plot(ts[i],phis[i], lw=1., color='black', alpha=0.2)

if n_errs > 0:
    print(f"Omitting {n_errs} trials with spin rates > 2.5 n")

plt.savefig('3body_trials.png', dpi=300)
plt.clf()

# plot trajectories (theta vs omega)
fig, (ax) = plt.subplots(1, 1,figsize=(5, 5))
plt.subplots_adjust(left=0.10, bottom=0.10, right=.98, top=0.98, wspace=0.02, hspace=0.02)
ax.set_ylabel(r"$\theta$ ($^{\circ}$)")
ax.set_xlabel(r"$\omega/n$")
for i in range(n_trials):
    ax.plot(omegas[i],thetas[i], lw=1., color='black', alpha=0.2)
plt.savefig('3body_trajs.png', dpi=300)
plt.clf()
