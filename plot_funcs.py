import numpy as np
import so_params as sops
import matplotlib
matplotlib.use('Agg')
# from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

# t_unit is unit of time displayed on x axis (value is in units of orbital periods)
def plot_trial(triaxial_bool,axs,data,lo,hi,ds,alpha,inds,clr='black',lw=0.75, t_unit=1.e6):
    
    # r is vector from planet to star !

    # first dimension of every stacked array is the component of the vector
    # second dimension is the time index
    rs = np.stack((data[inds['rx'],lo:hi:ds],data[inds['ry'],lo:hi:ds],data[inds['rz'],lo:hi:ds]), axis=0)
    rs /= sops.many_mags(rs)
    vs = np.stack((data[inds['vx'],lo:hi:ds],data[inds['vy'],lo:hi:ds],data[inds['vz'],lo:hi:ds]), axis=0)
    ss = np.stack((data[inds['si'],lo:hi:ds],data[inds['sj'],lo:hi:ds],data[inds['sk'],lo:hi:ds]), axis=0)
    iss = np.stack((data[inds['ix'],lo:hi:ds],data[inds['iy'],lo:hi:ds],data[inds['iz'],lo:hi:ds]), axis=0)
    js = np.stack((data[inds['jx'],lo:hi:ds],data[inds['jy'],lo:hi:ds],data[inds['jz'],lo:hi:ds]), axis=0)
    ks = np.stack((data[inds['kx'],lo:hi:ds],data[inds['ky'],lo:hi:ds],data[inds['kz'],lo:hi:ds]), axis=0)
    ts = data[inds['t'],lo:hi:ds] / t_unit

    n = np.sqrt(np.dot(vs[:,0],vs[:,0])) / np.sqrt(np.dot(rs[:,0],rs[:,0])) # mean-motion

    omega_to_ns = data[inds['omega'],lo:hi:ds]
    theta_rad, phi_rad = sops.get_theta_phi(ss,iss,js,ks,rs,vs)
    thetas = np.degrees(theta_rad)
    # phis = np.degrees(phi_rad)
    # psis = np.degrees(sops.get_psi_v2(rs,iss,js,ks,ss))

    betas = np.degrees(sops.get_beta(ss))

    if triaxial_bool:
        col = 0
    else:
        col = 1
    
    axs[0,col].plot(ts,omega_to_ns,color=clr,lw=lw,alpha=alpha)
    axs[1,col].plot(ts,thetas,color=clr,lw=lw,alpha=alpha)
    axs[2,col].plot(ts,betas,color=clr,lw=lw,alpha=alpha)

def get_fig_axs():
    a = 3
    b = 2
    fig, axs = plt.subplots(a, b,figsize=(8, 6), sharex=True)
    plt.subplots_adjust(left=0.08, bottom=0.1, right=.98, top=0.92, wspace=0.1, hspace=0.08)
    ylabels = [r"$\Omega/n$",r"$\theta$ ($^{\circ}$)",r"$\beta$ ($^{\circ}$)"]
    # ylabels = np.array([[r"$\Omega/n$",r"$\theta$ ($^{\circ}$)",r"$\phi$ ($^{\circ}$)",r"$\psi$ ($^{\circ}$)"],[r"$\beta$ ($^{\circ}$)",r"$\theta_{kl}$ ($^{\circ}$)",r"$tan^{-1}(s_y/s_x)$ ($^{\circ}$)",r"$\theta '$ ($^{\circ}$)"]]) # ,r"Inclination"]
    for i in range(b):
        axs[a-1,i].set_xlabel(r"Time ($P$)")
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