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
    fig, axs = plt.subplots(nv-1, 2,figsize=(10, 16), sharex=True)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=.98, top=0.92, wspace=0.1, hspace=0.1)
    ylabels = [r"$\omega/n$",r"$\theta$ ($^{\circ}$)",r"$\phi$ ($^{\circ}$)",r"$\psi$ ($^{\circ}$)",r"$\phi$ ($^{\circ}$)",r"Inclination"]
    for i in range(nv-1):
        axs[i,0].set_ylabel(ylabels[i])
    axs[nv-2,0].set_xlabel("Time (P)")
    axs[nv-2,1].set_xlabel("Time (P)")
    axs[0,0].set_title("Triaxial")
    axs[0,1].set_title("Oblate")

    return fig,axs

# ds is ds
def plot_trial(fig,axs,triax_out_data,j2_out_data,inds,ds,alpha):
    omega_ind,theta_ind,phi_ind,psi_ind,e_ind,t_ind = inds
    triax_ts = triax_out_data[t_ind,::ds]
    j2_ts = j2_out_data[t_ind,::ds]

    # triax_ts = np.arange(0,3.e7 + 1,10000)
    # j2_ts = np.arange(0,3.e7 + 1,10000)
    # print(np.shape(triax_out_data))

    axs[omega_ind,0].plot(triax_ts,triax_out_data[omega_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[theta_ind,0].plot(triax_ts,triax_out_data[theta_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[phi_ind,0].plot(triax_ts,triax_out_data[phi_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[psi_ind,0].plot(triax_ts,triax_out_data[psi_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[e_ind,0].plot(triax_ts,triax_out_data[e_ind,::ds], lw=.5, color='black', alpha=alpha)
    #axs[inc_ind,0].plot(triax_ts,triax_out_data[inc_ind,::ds], lw=.5, color='black', alpha=alpha)

    axs[omega_ind,1].plot(j2_ts,j2_out_data[omega_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[theta_ind,1].plot(j2_ts,j2_out_data[theta_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[phi_ind,1].plot(j2_ts,j2_out_data[phi_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[psi_ind,1].plot(j2_ts,j2_out_data[psi_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[e_ind,1].plot(j2_ts,j2_out_data[e_ind,::ds], lw=.5, color='black', alpha=alpha)
    #axs[inc_ind,1].plot(j2_ts,j2_out_data[inc_ind,::ds], lw=.5, color='black', alpha=alpha)

if __name__=="__main__":

    together = True

    skip_trials = [26,32] # trials that didn't complete, skip them

    # read data
    n_trials = 20
    nv = 6
    omega_ind = 0
    theta_ind = 1
    phi_ind = 2
    psi_ind = 3
    e_ind = 4
    # inc_ind = 5
    t_ind = 5
    inds = omega_ind,theta_ind,phi_ind,psi_ind,e_ind,t_ind

    ds = int(1.e3)

    dir_path = "./v2_3bd_4sp_20i_3j2_5tri_300Q_0.025dt"
    plots_dir = os.path.join(dir_path,"plots")
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    triax_fs = [dir_path+"/trial_"+str(i)+".npy" for i in range(n_trials)]
    j2_fs = [dir_path+"/trial_"+str(i)+".1.npy" for i in range(n_trials)]

    if together:
        fig, axs = get_fig_axs(nv)
        alpha = 0.2
    else:
        alpha = 1.

    # n_errs = 0
    for i in range(n_trials):
        if i in skip_trials:
            continue
        if not together:
            fig, axs = get_fig_axs(nv)

        f_triax = open(triax_fs[i], 'rb')
        triax_out_data = np.load(f_triax)

        f_j2 = open(j2_fs[i], 'rb')
        j2_out_data = np.load(f_j2)
        # j2_out_data = np.zeros_like(triax_out_data)
        
        plot_trial(fig,axs,triax_out_data,j2_out_data,inds,ds,alpha)

        if not together:
            plt.savefig(os.path.join(plots_dir,'3body_trial_'+str(i)+'.png'), dpi=300)
            plt.clf()

    # if n_errs > 0:
    #     print(f"Omitting {n_errs} trials with spin rates > 8n")

    if together:
        plt.savefig(os.path.join(plots_dir,'3body_trials.png'), dpi=300)
        plt.clf()

    if together:
        # plot trajectories (theta vs omega)
        fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10, 5), sharey=True)
        plt.subplots_adjust(left=0.10, bottom=0.10, right=.98, top=0.90, wspace=0.02, hspace=0.02)
        ax1.set_ylabel(r"$\theta$ ($^{\circ}$)")
        ax1.set_xlabel(r"$\omega/n$")
        ax2.set_xlabel(r"$\omega/n$")
        ax1.set_title("Triaxial")
        ax2.set_title("Oblate")
        for i in range(n_trials):
            if i in skip_trials:
                continue
            
            f_triax = open(triax_fs[i], 'rb')
            triax_out_data = np.load(f_triax)

            f_j2 = open(j2_fs[i], 'rb')
            j2_out_data = np.load(f_j2)
            # j2_out_data = np.zeros_like(triax_out_data)

            ax1.plot(triax_out_data[omega_ind,::ds],triax_out_data[theta_ind,::ds], lw=1., color='black', alpha=0.2)
            ax2.plot(j2_out_data[omega_ind,::ds],j2_out_data[theta_ind,::ds], lw=1., color='black', alpha=0.2)

        plt.savefig(os.path.join(plots_dir,'3body_trajs.png'), dpi=300)
        plt.clf()
