import os
import numpy as np
import time
import sys
import so_params as sops
import sim_funcs as smfs

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


if __name__ == "__main__":
    ############## Beginning of params block ###################
    # Params to change each time
    ds = 1
    tf = 1.e4
    n_out = int(1.e3)
    sn = "hamiltonian.png"

    # relatively constant params
    # some constants
    Re = 4.263e-5 # radius of Earth in AU
    Me = 3.003e-6 # mass of Earth in solar masses
    Mj = 9.542e-4 # mass of Jupiter in solar masses

    G = 39.476926421373

    ### SIMULATION PARAMETERS ###
    a = .4 # semi-major axis of inner planet
    Q_tide = 100.
    R_p = 2.*Re # radius of inner planet
    M_p = 4.*Me # mass of inner planet
    # TURN OFF TIDES
    k2 = 0 # 1.5 for uniformly distributed mass
    s_k_angle = np.radians(0.) # angle between s and k
    omega_to_n = 1.5
    theta = np.radians(0.)
    tri = 1.e-5
    obl = 1.e-3
    k = 0.331
    
    # for third body
    a_out = 5. # a of outer planet
    i_out = np.radians(20.) # inclination of outer planet
    M_out = 0

    ################### End of params block ####################
    
    # pre-sim
    A = k*M_p*R_p*R_p
    B = A*(1+tri)
    C = A*(1+obl)

    i, j, k = np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0])
    sim_params = i,j,k,a,Q_tide,R_p,theta,omega_to_n,M_p,k2,tri,obl,s_k_angle,a_out,i_out,M_out
    sim = smfs.create_sim(sim_params,dt_frac=0.01,rand_ijk=False)
    ps = sim.particles

    val_names = ["ix","iy","iz","jx","jy","jz","kx","ky","kz","si","sj","sk","omega","rx","ry","rz","vx","vy","vz","t"] # r is vector from planet to star !
    inds = {val_names[i]:i for i in range(len(val_names))}
    year = ps[1].P
    step = tf/(n_out-1)
    nv = len(val_names)
    data = np.zeros((nv,n_out), dtype=np.float64)

    # run sim
    for i in range(n_out):
        sim.integrate(i*step*year)
        data[inds['ix'],i] = ps[1].params['tt_ix']
        data[inds['iy'],i] = ps[1].params['tt_iy']
        data[inds['iz'],i] = ps[1].params['tt_iz']
        data[inds['jx'],i] = ps[1].params['tt_jx']
        data[inds['jy'],i] = ps[1].params['tt_jy']
        data[inds['jz'],i] = ps[1].params['tt_jz']
        data[inds['kx'],i] = ps[1].params['tt_kx']
        data[inds['ky'],i] = ps[1].params['tt_ky']
        data[inds['kz'],i] = ps[1].params['tt_kz']
        data[inds['si'],i] = ps[1].params['tt_si']
        data[inds['sj'],i] = ps[1].params['tt_sj']
        data[inds['sk'],i] = ps[1].params['tt_sk']
        data[inds['omega'],i] = ps[1].params['tt_omega']/ps[1].n
        data[inds['rx'],i] = ps[0].x - ps[1].x
        data[inds['ry'],i] = ps[0].y - ps[1].y
        data[inds['rz'],i] = ps[0].z - ps[1].z
        data[inds['vx'],i] = ps[1].vx
        data[inds['vy'],i] = ps[1].vy
        data[inds['vz'],i] = ps[1].vz
        data[inds['t'],i] = sim.t / year

    # calculate rotational kinetic energy of spherical planet in synchronous rotation to scale H by
    E_unit = 0.5*A

    # calculate kinetic energy
    pre_T = 0.5*A*data[inds['omega'],::ds]*data[inds['omega'],::ds] / E_unit
    si = data[inds['si'],::ds]
    sj = data[inds['sj'],::ds]
    sk = data[inds['sk'],::ds]
    post_T = si**2 + sj**2*(1 + tri) + sk**2*(1 + obl)
    T = pre_T * post_T

    #############
    # calculate potential energy (calc I_r) (try 1)
    # ix = data[inds['ix'],::ds]
    # iy = data[inds['iy'],::ds]
    # iz = data[inds['iz'],::ds]
    # jx = data[inds['jx'],::ds]
    # jy = data[inds['jy'],::ds]
    # jz = data[inds['jz'],::ds]
    # kx = data[inds['kx'],::ds]
    # ky = data[inds['ky'],::ds]
    # kz = data[inds['kz'],::ds]

    # ijk_basis = np.array([[ix,jx,kx],[iy,jy,ky],[iz,jz,kz]])
    # ijk_bases = np.transpose(ijk_basis, axes=[2, 0, 1])
    # I_ijk = np.array([[A,0,0],[0,B,0],[0,0,C]])

    # r_vecs = np.transpose(np.array([data[inds['rx'],::ds],data[inds['ry'],::ds],data[inds['rz'],::ds]]), axes=[1,0])
    # r_mags = np.sqrt(np.sum(r_vecs*r_vecs, axis=1))
    # r_mags_reshaped = np.stack((r_mags,r_mags,r_mags),axis=-1)
    # r_hats = r_vecs/r_mags_reshaped

    # Ir = np.array([r_hats[i] @ ijk_bases[i] @ I_ijk @ np.array([r_hats[i]]).T for i in range(np.shape(r_hats)[0])]).T[0]
    #############

    # calculate potential energy (calc I_r) (try 2)
    # A*alpha^2 + B*beta^2 + C*gamma^2
    # sx = si*ix + sj*jx + sk*kx
    # sy = si*iy + sj*jy + sk*ky
    # sz = si*iz + sj*jz + sk*kz
    # Ir = A*sx**2 + B*sy**2 + C*sz**2
    #############

    # calculate potential energy (calc I_r) (try 3)
    rs = np.stack((data[inds['rx'],::ds],data[inds['ry'],::ds],data[inds['rz'],::ds]), axis=0)
    r_mags = sops.many_mags(rs)
    r_hats = rs / r_mags
    vs = np.stack((data[inds['vx'],::ds],data[inds['vy'],::ds],data[inds['vz'],::ds]), axis=0)
    ss = np.stack((data[inds['si'],::ds],data[inds['sj'],::ds],data[inds['sk'],::ds]), axis=0)
    iss = np.stack((data[inds['ix'],::ds],data[inds['iy'],::ds],data[inds['iz'],::ds]), axis=0)
    js = np.stack((data[inds['jx'],::ds],data[inds['jy'],::ds],data[inds['jz'],::ds]), axis=0)
    ks = np.stack((data[inds['kx'],::ds],data[inds['ky'],::ds],data[inds['kz'],::ds]), axis=0)
    theta_rad, phi_rad = sops.get_theta_phi(ss,iss,js,ks,r_hats,vs)
    thetas = np.degrees(theta_rad)

    # A*alpha^2 + B*beta^2 + C*gamma^2
    gammas = np.cos(thetas)
    l_hats = sops.calc_orbit_normal(r_xyz=r_hats,v_xyz=vs)
    s_xyz = sops.many_ijk_to_xyz(ss, iss, js, ks)
    s_proj = s_xyz - (sops.many_dots(s_xyz,l_hats)*l_hats)
    s_proj_mags = sops.many_mags(s_proj)
    s_proj_mag_is_zero = (s_proj_mags == 0)
    s_proj_mags = s_proj_mag_is_zero.astype(int) + ((s_proj_mag_is_zero.astype(int)-1)*(-1))*s_proj_mags
    s_proj_hats = s_proj/s_proj_mags
    alphas = sops.many_dots(s_proj_hats,r_hats)
    betas = np.sqrt(1- alphas**2 - gammas**2)

    # gamma x alpha = beta
    betas *= ((sops.many_dots(sops.many_crosses(l_hats, r_hats),s_xyz) < 0).astype(int)*(-2) + 1)
    # is the above necessary?

    Ir = A*alphas**2 + B*betas**2 + C*gammas**2

    #############

    V = (3*G/E_unit/2/(r_mags**3))*(Ir - ((A+B+C)/3))

    H = T + V
    ts = data[inds['t'],::ds]

    fig, axs = plt.subplots(3,1, figsize=(6, 10))
    plt.subplots_adjust(left=0.12, bottom=0.05, right=.98, top=0.95, wspace=0.02, hspace=0.2)
    axs[0,].set_ylabel(r"Hamiltonian")
    axs[0,].set_xlabel(r"Time (P)")
    axs[0,].scatter(ts,H,s=0.5,color='black')
    axs[1,].set_ylabel(r"Kinetic Energy")
    axs[1,].set_xlabel(r"Time (P)")
    axs[1,].scatter(ts,T,s=0.5,color='black')
    axs[2,].set_ylabel(r"Potential Energy")
    axs[2,].set_xlabel(r"Time (P)")
    axs[2,].scatter(ts,V,s=0.5,color='black')

    plt.savefig(sn, dpi=300)
    plt.clf()
    plt.close(fig)






