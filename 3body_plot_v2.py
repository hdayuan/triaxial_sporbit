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

# returns orbit normal unit vector for orbit of body at index index
def calc_orbit_normal(ps, index):
    r_xyz = np.array([ps[index].x - ps[0].x,ps[index].y - ps[0].y,ps[index].z - ps[0].z]) # here, r is vector from star to planet (opposite from operator code)
    v_xyz = np.array([ps[index].vx,ps[index].vy,ps[index].vz])
    r = np.sqrt(np.dot(r_xyz,r_xyz))
    v = np.sqrt(np.dot(v_xyz,v_xyz))
    r_hat = r_xyz / r
    v_hat = v_xyz / v
    n = np.cross(r_hat, v_hat)
    n_hat = n/np.sqrt(np.dot(n,n))
    return n_hat

# returns (obliquity, phi) of body at index 1 in radians
def get_theta_phi(ps):
    # calculate theta
    s_ijk = np.array([ps[1].params['tt_si'],ps[1].params['tt_sj'],ps[1].params['tt_sk']])
    ijk_xyz = np.array([[ps[1].params['tt_ix'],ps[1].params['tt_iy'],ps[1].params['tt_iz']],[ps[1].params['tt_jx'],ps[1].params['tt_jy'],ps[1].params['tt_jz']],[ps[1].params['tt_kx'],ps[1].params['tt_ky'],ps[1].params['tt_kz']]])
    s_xyz = s_ijk[0]*ijk_xyz[0] + s_ijk[1]*ijk_xyz[1] + s_ijk[2]*ijk_xyz[2]
    s_xyz /= np.sqrt(np.dot(s_xyz,s_xyz))
    n_hat = calc_orbit_normal(ps,1) # orbit normal of triaxial planet
    theta = np.arccos(np.dot(n_hat,s_xyz))
    
    # calculate phi
    n_p_hat = calc_orbit_normal(ps,2) # orbit normal of perturbing planet
    y = np.cross(n_p_hat, n_hat) # unrelated to y basis unit vector
    y_hat = y/np.sqrt(np.dot(y,y))
    x = np.cross(y_hat, n_hat) # unrelated to x basis unit vector
    x_hat = x/np.sqrt(np.dot(x,x))
    phi = np.arctan2(np.dot(s_xyz,y_hat),np.dot(s_xyz,x_hat))

    # range from 0 to 360
    if phi < 0:
        phi = (2*np.pi) + phi
    
    return (theta, phi) 

# returns angle in rad
def get_psi(ps):
    r_vec = np.array([ps[1].x - ps[0].x, ps[1].y - ps[0].y, ps[1].z - ps[0].z])
    r = np.sqrt(np.dot(r_vec,r_vec))
    r_hat = r_vec / r
    i_hat = np.array([ps[1].params['tt_ix'], ps[1].params['tt_iy'], ps[1].params['tt_iz']])
    # j_hat = np.array([ps[1].params['tt_jx'], ps[1].params['tt_jy'], ps[1].params['tt_jz']])
    n_hat = calc_orbit_normal(ps,1)
    # subtact component of i that is not in orbital plane
    i_pl = i_hat - (np.dot(i_hat,n_hat)*n_hat)
    i_pl_hat = i_pl / np.sqrt(np.dot(i_pl,i_pl))
    j_pl_hat = np.cross(n_hat, i_pl_hat)

    i_dot_r = np.dot(i_pl_hat,r_hat)
    j_dot_r = np.dot(j_pl_hat,r_hat)
    psi = np.arctan2(j_dot_r, i_dot_r)

    # range from 0 to 360
    if psi < 0:
        psi = (2*np.pi) + psi

    return psi

# returns angle in rad
def get_sk_angle(ps):
    return np.arccos(ps[1].params['tt_sk'])

def get_fig_axs(nv):
    fig, axs = plt.subplots(nv-1, 2,figsize=(10, 16), sharex=True)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=.98, top=0.92, wspace=0.1, hspace=0.1)
    ylabels = [r"$\omega/n$",r"$\theta$ ($^{\circ}$)",r"$\phi$ ($^{\circ}$)",r"$\psi$ ($^{\circ}$)",r"$\beta$ ($^{\circ}$)"] # ,r"Inclination"]
    for i in range(nv-1):
        axs[i,0].set_ylabel(ylabels[i])
    axs[nv-2,0].set_xlabel("Time (P)")
    axs[nv-2,1].set_xlabel("Time (P)")
    axs[0,0].set_title("Triaxial")
    axs[0,1].set_title("Oblate")

    return fig,axs

# ds is ds
def plot_trial(fig,axs,triax_out_data,j2_out_data,inds,ds,alpha):
    omega_ind,theta_ind,phi_ind,psi_ind,sk_ind,t_ind = inds
    triax_ts = triax_out_data[t_ind,::ds]
    j2_ts = j2_out_data[t_ind,::ds]

    # triax_ts = np.arange(0,3.e7 + 1,10000)
    # j2_ts = np.arange(0,3.e7 + 1,10000)
    # print(np.shape(triax_out_data))

    axs[omega_ind,0].plot(triax_ts,triax_out_data[omega_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[theta_ind,0].plot(triax_ts,triax_out_data[theta_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[phi_ind,0].plot(triax_ts,triax_out_data[phi_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[psi_ind,0].plot(triax_ts,triax_out_data[psi_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[sk_ind,0].plot(triax_ts,triax_out_data[sk_ind,::ds], lw=.5, color='black', alpha=alpha)
    #axs[inc_ind,0].plot(triax_ts,triax_out_data[inc_ind,::ds], lw=.5, color='black', alpha=alpha)

    axs[omega_ind,1].plot(j2_ts,j2_out_data[omega_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[theta_ind,1].plot(j2_ts,j2_out_data[theta_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[phi_ind,1].plot(j2_ts,j2_out_data[phi_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[psi_ind,1].plot(j2_ts,j2_out_data[psi_ind,::ds], lw=.5, color='black', alpha=alpha)
    axs[sk_ind,1].plot(j2_ts,j2_out_data[sk_ind,::ds], lw=.5, color='black', alpha=alpha)
    #axs[inc_ind,1].plot(j2_ts,j2_out_data[inc_ind,::ds], lw=.5, color='black', alpha=alpha)

if __name__=="__main__":

    # Params to change each time
    together = False
    dir_path = "./v2.2_data"
    n_trials = 100
    skip_trials = [28,56,74,75,97] # trials that didn't complete, skip them

    # read data
    nv = 6
    omega_ind = 0
    theta_ind = 1
    phi_ind = 2
    psi_ind = 3
    sk_ind = 4
    # inc_ind = 5
    t_ind = 5
    inds = omega_ind,theta_ind,phi_ind,psi_ind,sk_ind,t_ind

    ds = int(1.e3)

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
        plt.close(fig)
