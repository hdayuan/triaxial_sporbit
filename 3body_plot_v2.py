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

# dot product of many vectors in a single array, where the first dimension is the component
def many_mags(a):
    return np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

def many_dots(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def many_crosses(a,b):
    xs = a[1]*b[2] - a[2]*b[1]
    ys = a[2]*b[0] - a[0]*b[2]
    zs = a[0]*b[1] - a[1]*b[0]
    return np.stack((xs,ys,zs),axis=0)

def many_ijk_to_xyz(s_ijk, i_xyz, j_xyz, k_xyz):
    sii = np.stack((s_ijk[0]*i_xyz[0],s_ijk[0]*i_xyz[1],s_ijk[0]*i_xyz[2]),axis=0)
    sjj = np.stack((s_ijk[1]*j_xyz[0],s_ijk[1]*j_xyz[1],s_ijk[1]*j_xyz[2]),axis=0)
    skk = np.stack((s_ijk[2]*k_xyz[0],s_ijk[2]*k_xyz[1],s_ijk[2]*k_xyz[2]),axis=0)
    s_xyz = sii+sjj+skk
    s_xyz /= many_mags(s_xyz)
    return s_xyz

# returns orbit normal unit vector for orbit of body at index index
def calc_orbit_normal(r_xyz=None,v_xyz=None,incl=np.radians(20.),outer=False):
    if outer:
        l = np.array([0.,-np.sin(incl),np.cos(incl)])
        l_hat = l/np.sqrt(np.dot(l,l))
        l_hats = np.array([[l_hat[j] for i in range(np.shape(r_xyz)[1])] for j in range(3)])
    else:
        l = many_crosses(v_xyz,r_xyz)
        l_hats = l/many_mags(l)
    return l_hats

# returns (obliquity, phi) of body at index 1 in radians
def get_theta_phi(s_ijk,i_xyz,j_xyz,k_xyz,r,v):
    # calculate theta
    s_x = s_ijk[0]*i_xyz[0] + s_ijk[1]*j_xyz[0] + s_ijk[2]*k_xyz[0]
    s_y = s_ijk[0]*i_xyz[1] + s_ijk[1]*j_xyz[1] + s_ijk[2]*k_xyz[1]
    s_z = s_ijk[0]*i_xyz[2] + s_ijk[1]*j_xyz[2] + s_ijk[2]*k_xyz[2]
    s_xyz = np.stack((s_x,s_y,s_z),axis=0)
    s_xyz /= many_mags(s_xyz)

    l_hat = calc_orbit_normal(r_xyz=r,v_xyz=v) # orbit normal of triaxial planet
    theta = np.arccos(many_dots(l_hat,s_xyz))
    
    # calculate phi
    l_p_hat = calc_orbit_normal(r_xyz=r,outer=True) # orbit normal of perturbing planet
    y = many_crosses(l_p_hat, l_hat) # unrelated to y basis unit vector
    y_hat = y/many_mags(y)
    x = many_crosses(y_hat, l_hat) # unrelated to x basis unit vector
    x_hat = x/many_mags(x)
    phi = np.arctan2(many_dots(s_xyz,y_hat),many_dots(s_xyz,x_hat))

    # range from 0 to 360
    phi += (phi < 0).astype(int) * 2*np.pi
    
    return (theta, phi)

# returns angle in rad
def get_psi(rs, iss, js):
    psi = np.arccos(many_dots(rs,iss))
    # range from 0 to 360
    # psi *= ((2 * (many_dots(rs,js) > 0).astype(int)) - 1) # add sign to [0,pi] angle
    # psi += (psi < 0).astype(int) * 2*np.pi # range from 0 to 180
    return psi

# psi is angle between r and {j cross s, or k cross s if s = j}
# psi is positive if in direction of spin, negative if otherwise
def get_psi_v2(rs,i_xyz,j_xyz,k_xyz,s_ijk):
    s_x = s_ijk[0]*i_xyz[0] + s_ijk[1]*j_xyz[0] + s_ijk[2]*k_xyz[0]
    s_y = s_ijk[0]*i_xyz[1] + s_ijk[1]*j_xyz[1] + s_ijk[2]*k_xyz[1]
    s_z = s_ijk[0]*i_xyz[2] + s_ijk[1]*j_xyz[2] + s_ijk[2]*k_xyz[2]
    s_xyz = np.stack((s_x,s_y,s_z),axis=0)
    s_xyz /= many_mags(s_xyz)

    long_axes = many_crosses(j_xyz,s_xyz)
    long_axes += (long_axes == 0.).astype(int) * many_crosses(k_xyz,s_xyz)
    long_axes /= many_mags(long_axes)
    signs = ((many_dots(many_crosses(rs,long_axes), s_xyz) > 0).astype(int)*2) - 1
    proto_psis = np.cos(many_dots(long_axes,rs))
    psis = signs * proto_psis
    return psis

# returns angle in rad
def get_beta(ss):
    beta = np.arccos(ss[2])
    return beta

def get_theta_kl(ks,r,v):
    l_hat = calc_orbit_normal(r_xyz=r,v_xyz=v)
    theta_kl = np.arccos(many_dots(ks,l_hat))
    return theta_kl

def get_theta_primes(triaxial_bool,ss,r,v,i_xyz,j_xyz,k_xyz):
    Re = 4.263e-5 # radius of Earth in AU
    Me = 3.003e-6 # mass of Earth in solar masses
    R_p = 2.*Re # radius of inner planet
    M_p = 4.*Me # mass of inner planet

    if triaxial_bool:
        moment2 = 1.e-5 # 1e-1 # (Ij - Ii) / Ii, < moment3
    else:
        moment2 = 0.
    moment3 = 1.e-3 # 2e-1 # (Ik - Ii) / Ii, > moment2
    k = 0.331
    Ii = k*M_p*R_p*R_p
    Ij = Ii*(1+moment2)
    Ik = Ii*(1+moment3)

    vec = np.stack((Ii*ss[0],Ij*ss[1],Ik*ss[2]),axis=0)
    vec_xyz = np.stack((vec[0]*i_xyz[0] + vec[1]*j_xyz[0] + vec[2]*k_xyz[0],vec[0]*i_xyz[1] + vec[1]*j_xyz[1] + vec[2]*k_xyz[1],vec[0]*i_xyz[2] + vec[1]*j_xyz[2] + vec[2]*k_xyz[2]),axis=0)
    vec_xyz /= many_mags(vec_xyz)
    return many_dots(vec_xyz,calc_orbit_normal(r_xyz=r,v_xyz=v))

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
    rs /= many_mags(rs)
    vs = np.stack((data[inds['vx'],::ds],data[inds['vy'],::ds],data[inds['vz'],::ds]), axis=0)
    ss = np.stack((data[inds['si'],::ds],data[inds['sj'],::ds],data[inds['sk'],::ds]), axis=0)
    iss = np.stack((data[inds['ix'],::ds],data[inds['iy'],::ds],data[inds['iz'],::ds]), axis=0)
    js = np.stack((data[inds['jx'],::ds],data[inds['jy'],::ds],data[inds['jz'],::ds]), axis=0)
    ks = np.stack((data[inds['kx'],::ds],data[inds['ky'],::ds],data[inds['kz'],::ds]), axis=0)
    ts = data[inds['t'],::ds]

    n = np.sqrt(np.dot(vs[:,0],vs[:,0])) / np.sqrt(np.dot(rs[:,0],rs[:,0])) # mean-motion

    omega_to_ns = data[inds['omega'],::ds]
    theta_rad, phi_rad = get_theta_phi(ss,iss,js,ks,rs,vs)
    thetas = np.degrees(theta_rad)
    phis = np.degrees(phi_rad)
    psis = np.degrees(get_psi_v2(rs,iss,js,ks,ss))

    betas = np.degrees(get_beta(ss))
    theta_kls = np.degrees(get_theta_kl(ks,rs,vs))
    s_xyz = many_ijk_to_xyz(ss,iss,js,ks)
    psi_primes = (psis - np.degrees(n*ts/2)) % 360 # np.degrees(np.arctan2(s_xyz[1], s_xyz[0]))
    theta_primes = many_dots(rs,ks) # np.degrees(get_theta_primes(triaxial_bool,ss,rs,vs,iss,js,ks))

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
    together = False
    if perturber:
        dir_name = "v"+str(version)+"_3body_data"
    else:
        dir_name = "v"+str(version)+"_2body_data"
    n_trials = 50
    skip_trials = [32] # trials that didn't complete, skip them
    ds = int(1.e2)
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
            theta_rad, phi_rad = get_theta_phi(ss,iss,js,ks,rs,vs)
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
            theta_rad, phi_rad = get_theta_phi(ss,iss,js,ks,rs,vs)
            thetas = np.degrees(theta_rad)

            ax2.plot(omega_to_ns,thetas, lw=1., color='black', alpha=0.2)

        plt.savefig(os.path.join(plots_dir,'3body_trajs.png'), dpi=300)
        plt.clf()
        plt.close(fig)
