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

def get_fig_axs(nv):
    fig, axs = plt.subplots(6, 2,figsize=(5, 8), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=.98, top=0.92, wspace=0.04, hspace=0.04)
    ylabels = [r"$\omega/n$",r"$\theta$ ($^{\circ}$)",r"$\phi$ ($^{\circ}$)",r"$\psi$ ($^{\circ}$)",r"Eccentricity",r"Inclination"]
    for i in range(nv-1):
        axs[i,0].set_ylabel(ylabels[i])
    axs[nv-2,0].set_xlabel("Time (P)")
    axs[nv-2,1].set_xlabel("Time (P)")
    axs[0,0].set_title("Triaxial")
    axs[0,1].set_title("Oblate")

    return fig,axs

# ds is ds
def plot_trial(fig,axs,triax_out_data,j2_out_data,inds,ds):
    omega_ind,theta_ind,phi_ind,psi_ind,e_ind,inc_ind,t_ind = inds
    triax_ts = triax_out_data[t_ind,::ds]
    j2_ts = j2_out_data[t_ind,::ds]

    axs[omega_ind,0].plot(triax_ts[::ds],triax_out_data[omega_ind,::ds], lw=1., color='black', alpha=0.2)
    axs[theta_ind,0].plot(triax_ts[::ds],triax_out_data[theta_ind,::ds], lw=1., color='black', alpha=0.2)
    axs[phi_ind,0].plot(triax_ts[::ds],triax_out_data[phi_ind,::ds], lw=1., color='black', alpha=0.2)
    axs[psi_ind,0].plot(triax_ts[::ds],triax_out_data[psi_ind,::ds], lw=1., color='black', alpha=0.2)
    axs[e_ind,0].plot(triax_ts[::ds],triax_out_data[e_ind,::ds], lw=1., color='black', alpha=0.2)
    axs[inc_ind,0].plot(triax_ts[::ds],triax_out_data[inc_ind,::ds], lw=1., color='black', alpha=0.2)

    axs[omega_ind,1].plot(j2_ts[::ds],j2_out_data[omega_ind,::ds], lw=1., color='black', alpha=0.2)
    axs[theta_ind,1].plot(j2_ts[::ds],j2_out_data[theta_ind,::ds], lw=1., color='black', alpha=0.2)
    axs[phi_ind,1].plot(j2_ts[::ds],j2_out_data[phi_ind,::ds], lw=1., color='black', alpha=0.2)
    axs[psi_ind,1].plot(j2_ts[::ds],j2_out_data[psi_ind,::ds], lw=1., color='black', alpha=0.2)
    axs[e_ind,1].plot(j2_ts[::ds],j2_out_data[e_ind,::ds], lw=1., color='black', alpha=0.2)
    axs[inc_ind,1].plot(j2_ts[::ds],j2_out_data[inc_ind,::ds], lw=1., color='black', alpha=0.2)

if __name__=="__main__":

    together = False

    # read data
    n_trials = 20
    nv = 7
    omega_ind = 0
    theta_ind = 1
    phi_ind = 2
    psi_ind = 3
    e_ind = 4
    inc_ind = 5
    t_ind = 6
    inds = omega_ind,theta_ind,phi_ind,psi_ind,e_ind,inc_ind,t_ind

    ds = 1.e3

    dir_path = "./v2_3bd_20i_3j2_5tri_300Q_0.025dt"
    triax_fs = [dir_path+"/trial_"+str(i)+".txt" for i in range(n_trials)]
    j2_fs = [dir_path+"/trial_"+str(i)+".1.txt" for i in range(n_trials)]

    if together:
        fig_axs = get_fig_axs(nv)

    # n_errs = 0
    for i in range(n_trials):
        if not together:
            fig, axs = get_fig_axs(nv)

        f_triax = open(triax_fs[i], 'rb')
        triax_out_data = np.load(f_triax)

        f_j2 = open(j2_fs[i], 'rb')
        j2_out_data = np.load(f_j2)
        
        plot_trial(fig,axs,triax_out_data,j2_out_data,ds)

        if not together:
            plt.savefig('3body_trial_'+str(i)+'.png', dpi=300)
            plt.clf()

    # if n_errs > 0:
    #     print(f"Omitting {n_errs} trials with spin rates > 8n")

    if together:
        plt.savefig('3body_trials.png', dpi=300)
        plt.clf()

    # plot trajectories (theta vs omega)
    fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10, 5), sharey=True)
    plt.subplots_adjust(left=0.10, bottom=0.10, right=.98, top=0.98, wspace=0.02, hspace=0.02)
    ax1.set_ylabel(r"$\theta$ ($^{\circ}$)")
    ax1.set_xlabel(r"$\omega/n$")
    ax2.set_xlabel(r"$\omega/n$")
    ax1.set_title("Triaxial")
    ax2.set_title("Oblate")
    for i in range(n_trials):
        f_triax = open(triax_fs[i], 'rb')
        triax_out_data = np.load(f_triax)

        f_j2 = open(j2_fs[i], 'rb')
        j2_out_data = np.load(f_j2)

        ax1.plot(triax_out_data[omega_ind,::ds],triax_out_data[theta_ind,::ds], lw=1., color='black', alpha=0.2)
        ax2.plot(j2_out_data[omega_ind,::ds],triax_out_data[theta_ind,::ds], lw=1., color='black', alpha=0.2)

    plt.savefig('3body_trajs.png', dpi=300)
    plt.clf()
