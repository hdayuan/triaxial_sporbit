import os
import numpy as np
import time
import sys
import so_params as sops

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=11)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

def get_fig_axs():
    a = 2
    b = 4
    fig, axs = plt.subplots(a, b,figsize=(16, 10), sharex=False)
    plt.subplots_adjust(left=0.05, bottom=0.1, right=.98, top=0.92, wspace=0.2, hspace=0.05)
    ylabels = np.array([[r"$\Omega/n$",r"$\theta$ ($^{\circ}$)",r"$\phi$ ($^{\circ}$)",r"$\psi$ ($^{\circ}$)"],[r"$\beta$ ($^{\circ}$)",r"$\theta_{kl}$ ($^{\circ}$)",r"$tan^{-1}(s_y/s_x)$ ($^{\circ}$)",r"$\theta '$ ($^{\circ}$)"]]) # ,r"Inclination"]
    for i in range(b):
        axs[1,i].set_xlabel("Time (P)")
        for j in range(a):
            axs[j,i].set_ylabel(ylabels[j,i])

    # fig.suptitle('Trial i = triaxial, Trial i+0.1 = oblate')

    # fig, axs = plt.subplots(nv-1, 2,figsize=(10, 16), sharex=True)
    # plt.subplots_adjust(left=0.15, bottom=0.1, right=.98, top=0.92, wspace=0.1, hspace=0.1)
    # ylabels = [r"$\Omega/n$",r"$\theta$ ($^{\circ}$)",r"$\phi$ ($^{\circ}$)",r"$\psi$ ($^{\circ}$)",r"$\beta$ ($^{\circ}$)"] # ,r"Inclination"]
    # for i in range(nv-1):
    #     axs[i,0].set_ylabel(ylabels[i])
    # axs[nv-2,0].set_xlabel("Time (P)")
    # axs[nv-2,1].set_xlabel("Time (P)")
    # axs[0,0].set_title("Triaxial")
    # axs[0,1].set_title("Oblate")

    return fig,axs

# ds is ds
def plot_trial(triaxial_bool,fig,axs,data,ds,alpha,inds):
    
    # r is vector from planet to star !

    # first dimension of every stacked array is the component of the vector
    # second dimension is the time index
    rs = np.stack((data[inds['rx'],::ds],data[inds['ry'],::ds],data[inds['rz'],::ds]), axis=0)
    rs /= sops.many_mags(rs)
    vs = np.stack((data[inds['vx'],::ds],data[inds['vy'],::ds],data[inds['vz'],::ds]), axis=0)
    ss = np.stack((data[inds['si'],::ds],data[inds['sj'],::ds],data[inds['sk'],::ds]), axis=0)
    iss = np.stack((data[inds['ix'],::ds],data[inds['iy'],::ds],data[inds['iz'],::ds]), axis=0)
    js = np.stack((data[inds['jx'],::ds],data[inds['jy'],::ds],data[inds['jz'],::ds]), axis=0)
    ks = np.stack((data[inds['kx'],::ds],data[inds['ky'],::ds],data[inds['kz'],::ds]), axis=0)
    ts = data[inds['t'],::ds]

    n = np.sqrt(np.dot(vs[:,0],vs[:,0])) / np.sqrt(np.dot(rs[:,0],rs[:,0])) # mean-motion

    omega_to_ns = data[inds['omega'],::ds]
    theta_rad, phi_rad = sops.get_theta_phi(ss,iss,js,ks,rs,vs)
    thetas = np.degrees(theta_rad)
    phis = np.degrees(phi_rad)
    psis = np.degrees(sops.get_psi_v2(rs,iss,js,ks,ss))

    betas = np.degrees(sops.get_beta(ss))
    theta_kls = np.degrees(sops.get_theta_kl(ks,rs,vs))
    s_xyz = sops.many_ijk_to_xyz(ss,iss,js,ks)
    psi_primes = (psis - np.degrees(n*ts/2)) % 360 # np.degrees(np.arctan2(s_xyz[1], s_xyz[0]))
    theta_primes = sops.many_dots(rs,ks) # np.degrees(sops.get_theta_primes(triaxial_bool,ss,rs,vs,iss,js,ks))

    w_lps = 3.*3.003e-6*(.4/5)**3*np.ones_like(ts)
    w_chandler = 1e-3*ss[2]*omega_to_ns

    # omega/n, theta, phi, psi
    # beta, theta_{kl}, psi', theta'
    axs[0,0].scatter(ts,omega_to_ns,color='black',s=0.5,alpha=alpha)
    axs[0,1].scatter(ts,thetas,color='black',s=0.5,alpha=alpha)
    axs[0,2].scatter(ts,phis,color='black',s=0.5,alpha=alpha)
    axs[0,3].scatter(ts,psi_primes,color='black',s=0.5,alpha=alpha)

    axs[1,0].scatter(ts,s_xyz[0],color='black',s=0.5,alpha=alpha)
    axs[1,1].scatter(ts,s_xyz[1],color='black',s=0.5,alpha=alpha)
    axs[1,2].scatter(ts,s_xyz[2],color='black',s=0.5,alpha=alpha)
    axs[1,3].scatter(ts,w_chandler,color='black',s=0.5,alpha=alpha)

    # triax_ts = triax_out_data[inds['t'],::ds]
    # j2_ts = j2_out_data[inds['t'],::ds]

    # # triax_ts = np.arange(0,3.e7 + 1,10000)
    # # j2_ts = np.arange(0,3.e7 + 1,10000)
    # # print(np.shape(triax_out_data))

    # axs[inds['omega'],0].plot(triax_ts,triax_out_data[inds['omega'],::ds], lw=.5, color='black', alpha=alpha)
    # axs[theta_ind,0].plot(triax_ts,triax_out_data[theta_ind,::ds], lw=.5, color='black', alpha=alpha)
    # axs[phi_ind,0].plot(triax_ts,triax_out_data[phi_ind,::ds], lw=.5, color='black', alpha=alpha)
    # axs[pinds['si'],0].plot(triax_ts,triax_out_data[pinds['si'],::ds], lw=.5, color='black', alpha=alpha)
    # axs[inds['sk'],0].plot(triax_ts,triax_out_data[inds['sk'],::ds], lw=.5, color='black', alpha=alpha)
    # #axs[inc_ind,0].plot(triax_ts,triax_out_data[inc_ind,::ds], lw=.5, color='black', alpha=alpha)

    # axs[inds['omega'],1].plot(j2_ts,j2_out_data[inds['omega'],::ds], lw=.5, color='black', alpha=alpha)
    # axs[theta_ind,1].plot(j2_ts,j2_out_data[theta_ind,::ds], lw=.5, color='black', alpha=alpha)
    # axs[phi_ind,1].plot(j2_ts,j2_out_data[phi_ind,::ds], lw=.5, color='black', alpha=alpha)
    # axs[pinds['si'],1].plot(j2_ts,j2_out_data[pinds['si'],::ds], lw=.5, color='black', alpha=alpha)
    # axs[inds['sk'],1].plot(j2_ts,j2_out_data[inds['sk'],::ds], lw=.5, color='black', alpha=alpha)
    # #axs[inc_ind,1].plot(j2_ts,j2_out_data[inc_ind,::ds], lw=.5, color='black', alpha=alpha)

if __name__=="__main__":

    # Params to change each time
    version = 2.4
    perturber = True
    together = True
    if perturber:
        dir_name = "v"+str(version)+"_3body_data"
    else:
        dir_name = "v"+str(version)+"_2body_data"
    n_trials = 50
    skip_trials = [] # trials that didn't complete, skip them
    ds = int(1.e1)
    # End of params block

    if version == 2.3:
        val_names = ["ix","iy","iz","jx","jy","jz","kx","ky","kz","si","sj","sk","omega","rx","ry","rz","vx","vy","vz","t"] # r is vector from planet to star !
    if version == 2.4:
        if perturber:
            val_names = ["ix","iy","iz","jx","jy","jz","kx","ky","kz","si","sj","sk","omega","rx","ry","rz","vx","vy","vz","rpx","rpy","rpz","t"] # r and rp are vectors from planet 1 and 2 to star !
        else:
            val_names = ["ix","iy","iz","jx","jy","jz","kx","ky","kz","si","sj","sk","omega","rx","ry","rz","vx","vy","vz","t"] # r is vector from planet to star !
    inds = {val_names[i]:i for i in range(len(val_names))}

    plots_dir = os.path.join("plots",dir_name)
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    triax_fs = ["data/"+dir_name+"/trial_"+str(i)+".npy" for i in range(n_trials)]
    j2_fs = ["data/"+dir_name+"/trial_"+str(i)+".1.npy" for i in range(n_trials)]

    if together:
        fig, axs = get_fig_axs()
        alpha = 0.2
    else:
        alpha = 1.

    # n_errs = 0
    for i in range(n_trials):
        if i in skip_trials:
            continue

        elif i != 15:
            continue

        for j in range(2):

            if not together:
                fig, axs = get_fig_axs()

            if j == 0: # triax
                f = open(triax_fs[i], 'rb')
            else: # oblate
                f = open(j2_fs[i], 'rb')

            data = np.load(f)

            # f_j2 = open(j2_fs[i], 'rb')
            # j2_out_data = np.load(f_j2)
            # # j2_out_data = np.zeros_like(triax_out_data)
            
            plot_trial(j==0,fig,axs,data,ds,alpha,inds)

            if j == 0:
                save_name = '3body_trial_'+str(i)+'_tr_psis_v2.png'
            else:
                save_name = '3body_trial_'+str(i)+'_ob_psis_v2.png'

            if not together:
                plt.savefig(os.path.join(plots_dir,save_name), dpi=300)
                plt.clf()
                plt.close(fig)

    # if n_errs > 0:
    #     print(f"Omitting {n_errs} trials with spin rates > 8n")

    if together:
        plt.savefig(os.path.join(plots_dir,'3body_trials.png'), dpi=300)
        plt.clf()
        plt.close(fig)

    if together:
        # plot trajectories (theta vs omega)
        fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10, 5), sharey=True)
        plt.subplots_adjust(left=0.10, bottom=0.10, right=.98, top=0.90, wspace=0.02, hspace=0.02)
        ax1.set_ylabel(r"$\theta$ ($^{\circ}$)")
        ax1.set_xlabel(r"$\Omega/n$")
        ax2.set_xlabel(r"$\Omega/n$")
        ax1.set_title("Triaxial")
        ax2.set_title("Oblate")
        for i in range(n_trials):
            if i in skip_trials:
                continue
            
            f_triax = open(triax_fs[i], 'rb')
            data = np.load(f_triax)

            rs = np.stack((data[inds['rx'],::ds],data[inds['ry'],::ds],data[inds['rz'],::ds]), axis=0)
            vs = np.stack((data[inds['vx'],::ds],data[inds['vy'],::ds],data[inds['vz'],::ds]), axis=0)
            ss = np.stack((data[inds['si'],::ds],data[inds['sj'],::ds],data[inds['sk'],::ds]), axis=0)
            iss = np.stack((data[inds['ix'],::ds],data[inds['iy'],::ds],data[inds['iz'],::ds]), axis=0)
            js = np.stack((data[inds['jx'],::ds],data[inds['jy'],::ds],data[inds['jz'],::ds]), axis=0)
            ks = np.stack((data[inds['kx'],::ds],data[inds['ky'],::ds],data[inds['kz'],::ds]), axis=0)

            omega_to_ns = data[inds['omega'],::ds]
            theta_rad, phi_rad = sops.get_theta_phi(ss,iss,js,ks,rs,vs)
            thetas = np.degrees(theta_rad)

            ax1.plot(omega_to_ns,thetas, lw=1., color='black', alpha=0.2)

            f_j2 = open(j2_fs[i], 'rb')
            data = np.load(f_j2)

            rs = np.stack((data[inds['rx'],::ds],data[inds['ry'],::ds],data[inds['rz'],::ds]), axis=0)
            vs = np.stack((data[inds['vx'],::ds],data[inds['vy'],::ds],data[inds['vz'],::ds]), axis=0)
            ss = np.stack((data[inds['si'],::ds],data[inds['sj'],::ds],data[inds['sk'],::ds]), axis=0)
            iss = np.stack((data[inds['ix'],::ds],data[inds['iy'],::ds],data[inds['iz'],::ds]), axis=0)
            js = np.stack((data[inds['jx'],::ds],data[inds['jy'],::ds],data[inds['jz'],::ds]), axis=0)
            ks = np.stack((data[inds['kx'],::ds],data[inds['ky'],::ds],data[inds['kz'],::ds]), axis=0)

            omega_to_ns = data[inds['omega'],::ds]
            theta_rad, phi_rad = sops.get_theta_phi(ss,iss,js,ks,rs,vs)
            thetas = np.degrees(theta_rad)

            ax2.plot(omega_to_ns,thetas, lw=1., color='black', alpha=0.2)

        plt.savefig(os.path.join(plots_dir,'3body_trajs.png'), dpi=300)
        plt.clf()
        plt.close(fig)
