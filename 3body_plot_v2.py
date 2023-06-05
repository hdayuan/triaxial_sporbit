import os
import numpy as np
import time
import sys
import so_params as sops

import matplotlib
matplotlib.use('Agg')
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=11)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

def get_fig_axs():
    a = 3
    b = 2
    fig, axs = plt.subplots(a, b,figsize=(8, 6), sharex=True)
    plt.subplots_adjust(left=0.08, bottom=0.1, right=.98, top=0.92, wspace=0.1, hspace=0.08)
    ylabels = [r"$\Omega/n$",r"$\theta$ ($^{\circ}$)",r"$\beta$ ($^{\circ}$)"]
    # ylabels = np.array([[r"$\Omega/n$",r"$\theta$ ($^{\circ}$)",r"$\phi$ ($^{\circ}$)",r"$\psi$ ($^{\circ}$)"],[r"$\beta$ ($^{\circ}$)",r"$\theta_{kl}$ ($^{\circ}$)",r"$tan^{-1}(s_y/s_x)$ ($^{\circ}$)",r"$\theta '$ ($^{\circ}$)"]]) # ,r"Inclination"]
    for i in range(b):
        axs[a-1,i].set_xlabel(r"Time ($10^6P$)")
    for j in range(a):
        axs[j,0].set_ylabel(ylabels[j])

    axs[0,0].set_title("Triaxial")
    axs[0,1].set_title("Oblate")

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
def plot_trial(triaxial_bool,fig,axs,data,ds,alpha,inds,clr='black',lw=0.75):
    
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
    ts = data[inds['t'],::ds] / 1.e6

    n = np.sqrt(np.dot(vs[:,0],vs[:,0])) / np.sqrt(np.dot(rs[:,0],rs[:,0])) # mean-motion

    omega_to_ns = data[inds['omega'],::ds]
    theta_rad, phi_rad = sops.get_theta_phi(ss,iss,js,ks,rs,vs)
    thetas = np.degrees(theta_rad)
    # phis = np.degrees(phi_rad)
    # psis = np.degrees(sops.get_psi_v2(rs,iss,js,ks,ss))

    betas = np.degrees(sops.get_beta(ss))
    # theta_kls = np.degrees(sops.get_theta_kl(ks,rs,vs))
    # s_xyz = sops.many_ijk_to_xyz(ss,iss,js,ks)
    # psi_primes = (psis - np.degrees(n*ts/2)) % 360 # np.degrees(np.arctan2(s_xyz[1], s_xyz[0]))
    # theta_primes = sops.many_dots(rs,ks) # np.degrees(sops.get_theta_primes(triaxial_bool,ss,rs,vs,iss,js,ks))

    # w_lps = 3.*3.003e-6*(.4/5)**3*np.ones_like(ts)
    # w_chandler = 1e-3*ss[2]*omega_to_ns

    # omega/n, theta, phi, psi
    # beta, theta_{kl}, psi', theta'
    if triaxial_bool:
        col = 0
    else:
        col = 1
    axs[0,col].plot(ts,omega_to_ns,color=clr,lw=lw,alpha=alpha)
    axs[1,col].plot(ts,thetas,color=clr,lw=lw,alpha=alpha)
    axs[2,col].plot(ts,betas,color=clr,lw=lw,alpha=alpha)
    # axs[0,2].scatter(ts,phis,color='black',s=0.5,alpha=alpha)
    # axs[0,3].scatter(ts,psi_primes,color='black',s=0.5,alpha=alpha)

    # axs[1,0].scatter(ts,s_xyz[0],color='black',s=0.5,alpha=alpha)
    # axs[1,1].scatter(ts,s_xyz[1],color='black',s=0.5,alpha=alpha)
    # axs[1,2].scatter(ts,s_xyz[2],color='black',s=0.5,alpha=alpha)
    # axs[1,3].scatter(ts,w_chandler,color='black',s=0.5,alpha=alpha)

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

def plot_varying_color(data, ax, cbr=False):

    rs = np.stack((data[inds['rx'],::ds],data[inds['ry'],::ds],data[inds['rz'],::ds]), axis=0)
    vs = np.stack((data[inds['vx'],::ds],data[inds['vy'],::ds],data[inds['vz'],::ds]), axis=0)
    ss = np.stack((data[inds['si'],::ds],data[inds['sj'],::ds],data[inds['sk'],::ds]), axis=0)
    iss = np.stack((data[inds['ix'],::ds],data[inds['iy'],::ds],data[inds['iz'],::ds]), axis=0)
    js = np.stack((data[inds['jx'],::ds],data[inds['jy'],::ds],data[inds['jz'],::ds]), axis=0)
    ks = np.stack((data[inds['kx'],::ds],data[inds['ky'],::ds],data[inds['kz'],::ds]), axis=0)

    # varying color
    ts = data[inds["t"],::ds] / 1.e6

    omega_to_ns = data[inds['omega'],::ds]
    theta_rad, phi_rad = sops.get_theta_phi(ss,iss,js,ks,rs,vs)
    thetas = np.degrees(theta_rad)

    points = np.array([omega_to_ns, thetas]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(ts[0], ts[-1])
    lc = LineCollection(segments, cmap='plasma', norm=norm)
    # Set the values used for colormapping
    lc.set_array(ts)
    lc.set_linewidth(1.5)
    lc.set_alpha(0.7)
    line = ax.add_collection(lc)

    if cbr:
        fig.colorbar(line, ax=ax, label=r"Time ($10^6P$)")

def plot_traj(data, ax):
    rs = np.stack((data[inds['rx'],::ds],data[inds['ry'],::ds],data[inds['rz'],::ds]), axis=0)
    vs = np.stack((data[inds['vx'],::ds],data[inds['vy'],::ds],data[inds['vz'],::ds]), axis=0)
    ss = np.stack((data[inds['si'],::ds],data[inds['sj'],::ds],data[inds['sk'],::ds]), axis=0)
    iss = np.stack((data[inds['ix'],::ds],data[inds['iy'],::ds],data[inds['iz'],::ds]), axis=0)
    js = np.stack((data[inds['jx'],::ds],data[inds['jy'],::ds],data[inds['jz'],::ds]), axis=0)
    ks = np.stack((data[inds['kx'],::ds],data[inds['ky'],::ds],data[inds['kz'],::ds]), axis=0)

    omega_to_ns = data[inds['omega'],::ds]
    theta_rad, phi_rad = sops.get_theta_phi(ss,iss,js,ks,rs,vs)
    thetas = np.degrees(theta_rad)

    ax.plot(omega_to_ns,thetas, lw=1., color='black', alpha=0.3)
    ax.scatter(omega_to_ns[:1],thetas[:1], s=2, color='tab:red')
    ax.scatter(omega_to_ns[-1:],thetas[-1:], s=2, color='tab:blue')


if __name__=="__main__":

    # Params to change each time
    version = 2.5
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
    if version == 2.4 or version == 2.5:
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

    for k in range(2):
        if together:
            fig, axs = get_fig_axs()
            al = 1.
            lw = 1.
            if k==0: # those that start from 1.2
                axs[1,0].plot([0,10],[90,90],"--",c="tab:blue",alpha=al,lw=lw)
                axs[1,0].legend([r"$\theta = 90^{\circ}$"])
                axs[1,1].plot([0,10],[90,90],"--",c="tab:blue",alpha=al,lw=lw)
                axs[1,1].legend([r"$\theta = 90^{\circ}$"])
                axs[0,0].plot([0,10],[1,1],"--",c="goldenrod",alpha=al,lw=lw)
                axs[0,0].legend([r"1:1 SOR"])
                axs[0,1].plot([0,10],[1,1],"--",c="goldenrod",alpha=al,lw=lw)
                axs[0,1].legend([r"1:1 SOR"])
            else: # trials that start from 2.2
                axs[0,0].plot([0,10],[1,1],"--",c="goldenrod",alpha=al,lw=lw)
                axs[0,1].plot([0,10],[1,1],"--",c="goldenrod",alpha=al,lw=lw)
                axs[0,0].plot([0,10],[2,2],"--",c="tab:purple",alpha=al,lw=lw)
                axs[0,1].plot([0,10],[2,2],"--",c="tab:purple",alpha=al,lw=lw)
                axs[1,0].plot([0,10],[90,90],"--",c="tab:blue",alpha=al,lw=lw)
                axs[1,0].legend([r"$\theta = 90^{\circ}$"])
                axs[1,1].plot([0,10],[90,90],"--",c="tab:blue",alpha=al,lw=lw)
                axs[1,1].legend([r"$\theta = 90^{\circ}$"])

                axs[0,0].legend([r"1:1 SOR",r"2:1 SOR"])
                axs[0,1].legend([r"1:1 SOR",r"2:1 SOR"])
                
            alpha = 0.2
        else:
            alpha = 1.

        # n_errs = 0
        for i in range(n_trials+2):
            trial = i
            if i%2 != k:
                continue
            if i in skip_trials:
                continue

            # 2 example trials that will be plotted at end in diff color and alpha = 1
            if i == 26 or i == 31:
                continue
            if i == n_trials:
                trial = 26
                alpha = 1.
            if i == n_trials + 1:
                trial = 31
                alpha = 1.

            # elif i != 15:
            #     continue

            for j in range(2):

                if not together:
                    fig, axs = get_fig_axs()

                if j == 0: # triax
                    f = open(triax_fs[trial], 'rb')
                else: # oblate
                    f = open(j2_fs[trial], 'rb')

                data = np.load(f)

                # f_j2 = open(j2_fs[i], 'rb')
                # j2_out_data = np.load(f_j2)
                # # j2_out_data = np.zeros_like(triax_out_data)
                if trial == 26 or trial == 31:
                    plot_trial(j==0,fig,axs,data,ds,alpha,inds,clr='tab:red',lw=1.25)
                else:
                    plot_trial(j==0,fig,axs,data,ds,alpha,inds)

                if j == 0:
                    save_name = '3body_trial_'+str(trial)+'_tr_psis_v2.png'
                else:
                    save_name = '3body_trial_'+str(trial)+'_ob_psis_v2.png'

                if not together:
                    plt.savefig(os.path.join(plots_dir,save_name), dpi=300)
                    plt.clf()
                    plt.close(fig)

        # if n_errs > 0:
        #     print(f"Omitting {n_errs} trials with spin rates > 8n")

        if together:
            if k == 0:
                if perturber:
                    sn = '3body_trials_1.2.png'
                else:
                    sn = '2body_trials_1.2.png'
            else:
                if perturber:
                    sn = '3body_trials_2.2.png'
                else:
                    sn = '2body_trials_2.2.png'
            plt.savefig(os.path.join(plots_dir,sn), dpi=300)
            plt.clf()
            plt.close(fig)

    if together:
        # plot trajectories (theta vs omega)
        fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(6.5, 4), sharey=True)
        plt.subplots_adjust(left=0.08, bottom=0.10, right=.98, top=0.92, wspace=0.02, hspace=0.02)
        ax1.set_ylabel(r"$\theta$ ($^{\circ}$)")
        ax1.set_xlabel(r"$\Omega/n$")
        ax2.set_xlabel(r"$\Omega/n$")
        ax1.set_title("Triaxial")
        ax2.set_title("Oblate")

        # ax1.set_ylim(-4,184)
        # ax1.set_xlim(-0.1,2.3)
        # ax2.set_ylim(-4,184)
        # ax2.set_xlim(-0.1,2.3)

        # plot line at 90 degrees
        # ax1.plot([0,2.2],[90,90],"--",c="tab:green",alpha=1.,lw=1.)
        # ax2.plot([0,2.2],[90,90],"--",c="tab:green",alpha=1.,lw=1.)
        
        # ax1.plot([2,2],[0,180],"--",c="darkorange",alpha=1.,lw=1.)
        # ax2.plot([2,2],[0,180],"--",c="darkorange",alpha=1.,lw=1.)

        # ax1.plot([1,1],[0,180],"--",c="gold",alpha=1.,lw=1.)
        # ax2.plot([1,1],[0,180],"--",c="gold",alpha=1.,lw=1.)

        for i in range(n_trials):
            if i in skip_trials:
                continue
            
            f_triax = open(triax_fs[i], 'rb')
            data = np.load(f_triax)

            # plot_varying_color(data,ax1)
            plot_traj(data,ax1)

            f_j2 = open(j2_fs[i], 'rb')
            data = np.load(f_j2)

            # if i == 0:
            #     plot_varying_color(data,ax2,cbr=True)
            # else:
            #     plot_varying_color(data,ax2)
            plot_traj(data,ax2)

        if perturber:
            sn = '3body_trajs.png'
        else:
            sn = '2body_trajs.png'
        plt.savefig(os.path.join(plots_dir,sn), dpi=300)
        plt.clf()
        plt.close(fig)
