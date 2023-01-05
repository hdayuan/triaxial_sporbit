import os
import rebound
import reboundx
import numpy as np
import time
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

# make simulation object with given parameters
def create_sim(sim_params,dt_frac,dtheta_offset=0.):
    a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle = sim_params
    if dtheta_offset != 0. and obliquity != 0.:
        print("Error: either obliquity or dtheta_offset must be 0.")
        exit()

    sim = rebound.Simulation()
    sim.integrator = 'whfast'
    sim.units = ('AU', 'yr', 'MSun')
    
    sim.add(m=1.)
    sim.add(m=M_p, a=a)

    rebx = reboundx.Extras(sim)
    triax = rebx.load_operator('triaxial_torque')
    rebx.add_operator(triax)

    # add spin to smaller body
    ps = sim.particles

    if obliquity != 0:
        ps[1].params['tt_ix'] = np.cos(obliquity)
        ps[1].params['tt_iy'] = 0.
        ps[1].params['tt_iz'] = -np.sin(obliquity)
        ps[1].params['tt_jx'] = 0.
        ps[1].params['tt_jy'] = 1.
        ps[1].params['tt_jz'] = 0.
        ps[1].params['tt_kx'] = np.sin(obliquity)
        ps[1].params['tt_ky'] = 0.
        ps[1].params['tt_kz'] = np.cos(obliquity)
    else:
        ps[1].params['tt_ix'] = np.cos(dtheta_offset)
        ps[1].params['tt_iy'] = -np.sin(dtheta_offset)
        ps[1].params['tt_iz'] = 0.
        ps[1].params['tt_jx'] = np.sin(dtheta_offset)
        ps[1].params['tt_jy'] = np.cos(dtheta_offset)
        ps[1].params['tt_jz'] = 0.
        ps[1].params['tt_kx'] = 0.
        ps[1].params['tt_ky'] = 0.
        ps[1].params['tt_kz'] = 1.

    # (2/5)*MR^2
    Ii = (2/5)*M_p*R_p*R_p
    Ij = Ii*(1+moment2)
    Ik = Ii*(1+moment3)

    ps[1].params['tt_Ii'] = Ii
    ps[1].params['tt_Ij'] = Ij
    ps[1].params['tt_Ik'] = Ik

    ps[1].params['tt_si'] = np.sin(s_k_angle)
    ps[1].params['tt_sj'] = 0.
    ps[1].params['tt_sk'] = np.cos(s_k_angle)

    tidal_dt = np.arctan(1./Q_tide) / 2. / ps[1].n # check this / change n to some other frequency?
    omega = omega_to_n*ps[1].n

    ps[1].params['tt_omega'] = omega
    ps[1].params['tt_R'] = R_p
    ps[1].params['tt_k2'] = k2
    ps[1].params['tt_tidal_dt'] = tidal_dt

    sim.dt = dt_frac*np.minimum(ps[1].P, 2*np.pi/omega)

    return sim

# tests convergence order and makes plot
def test_convergence(sim_params, n_dts, dtmax):

    a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle = sim_params
    k2 = 0. # turn off tides
    omega_to_n = 1. # make sure synchronous
    obliquity = 0. # make sure obliquity is 0

    # (2/5)*MR^2
    Ii = (2/5)*M_p*R_p*R_p
    Ij = Ii*(1+moment2)
    Ik = Ii*(1+moment3)

    sim_params = (a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle)
    dtheta = np.radians(0.01)

    dts = dtmax / 2**np.arange(int(n_dts-1), -1, -1) # fraction of min(orbital period, spin period)
    errors = np.zeros(n_dts)
    
    for i in range(n_dts):
        # create simulation
        sim = create_sim(sim_params,dts[i],dtheta_offset=dtheta)
        # run simulation
        sim.integrate(dtmax*a**1.5)
        ps = sim.particles
        rx = ps[0].x - ps[1].x
        ry = ps[0].y - ps[1].y
        rz = ps[0].z - ps[1].z
        r = np.sqrt(rx**2 + ry**2 +rz**2)
        rx /= r
        ry /= r
        rz /= r
        i_dot_r = ps[1].params['tt_ix']*rx + ps[1].params['tt_iy']*ry + ps[1].params['tt_iz']*rz
        j_dot_r = ps[1].params['tt_jx']*rx + ps[1].params['tt_jy']*ry + ps[1].params['tt_jz']*rz
        angle = -np.arctan2(j_dot_r, i_dot_r) # figure out why I need negative sign??
        # only correct for M_star = 1, circular orbit
        freq = np.sqrt(3*sim.G*(Ij-Ii)/Ik/(a**3))
        exact_sol = np.pi - dtheta * np.cos(freq * dtmax*a**1.5)
        errors[i] = np.abs(angle - exact_sol) / exact_sol

    # make plot
    fig, (ax) = plt.subplots(1, 1,figsize=(10, 6))
    # ax.set_yscale('log')
    # ax.plot(dts, final_residuals[n_dts-1] * (dts / dts[n_dts-1])**2, c='y', lw=0.5,
    #         label='2nd order')
    # ax.plot(dts, final_residuals[n_dts-1] * (dts / dts[n_dts-1])**3, c='b', lw=0.5,
    #         label='3rd order')
    ax.loglog(dts, errors[n_dts-1] * (dts / dts[n_dts-1])**4, c='r', lw=1.,
            label='4th order')
    ax.loglog(dts, errors, 'o',c='black', label='Numerical Error')
    ax.legend(fontsize=14)
    ax.set_xlabel("Time step (spin periods)")
    ax.set_ylabel("Error")

    plt.savefig('convergence.png', dpi=300)
    plt.clf()

def test_spin_damp(sim_params, dt_frac, out_step_frac, tf_frac):

    start = time.time()

    a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle = sim_params

    # create simulation
    sim = create_sim(sim_params,dt_frac)

    tf = tf_frac*a**1.5
    out_step = out_step_frac*a**1.5
    n = int(tf / out_step)

    if out_step < sim.dt:
        print("ERROR: output step is smaller than maximum timestep")
        exit()

    ps = sim.particles
    mm = ps[1].n
    ts = np.zeros(n)
    omegas = np.zeros(n)
    ns = np.zeros(n)
    for i in range(n):
        sim.integrate(i*out_step)
        ts[i] = sim.t
        omegas[i] = ps[1].params['tt_omega']
        ns[i] = ps[1].n

    print("Integration time: "+str(time.time() - start)+" seconds")

    # plot
    fig, (ax) = plt.subplots(1, 1,figsize=(10, 6))

    tidal_dt = np.arctan(1./Q_tide) / 2. / mm
    omega0 = omega_to_n*mm
    theta_lag = (omega0 - mm) * tidal_dt

    # in case spin is very large
    max_theta_lag = np.pi/4
    if theta_lag > max_theta_lag:
        theta_lag = max_theta_lag
    if theta_lag < -max_theta_lag:
        theta_lag = -max_theta_lag

    exact_sol = ((omega0 - mm) - ts*(15.*k2*sim.G*R_p**3*np.cos(theta_lag)*np.sin(theta_lag)/(2.*M_p*a**6))) / mm
    ax.plot(ts/(a**1.5), np.array(((omegas)-np.array(ns))/mm), 'ko')
    ax.plot(ts/(a**1.5), exact_sol, color='tab:blue')
    ax.legend(['Numerical', 'Analytical'])
    ax.set_ylabel(r"$\frac{\omega - n}{n}$")
    ax.set_xlabel('Time (orbital periods)')

    plt.savefig('spin_damp_'+str(dt_frac)+'dt.png', dpi=300)
    plt.clf()

def test_obl_damp(sim_params, dt_frac, out_step_frac, tf_frac):
    
    start = time.time()
    a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle = sim_params

    # create simulation
    sim = create_sim(sim_params,dt_frac)

    tf = tf_frac*a**1.5
    out_step = out_step_frac*a**1.5
    n = int(tf / out_step)

    if out_step < sim.dt:
        print("ERROR: output step is smaller than maximum timestep")
        exit()

    ps = sim.particles
    mm = ps[1].n
    ts = np.zeros(n)
    obliquities = np.zeros(n)
    for i in range(n):
        sim.integrate(i*out_step)
        ts[i] = sim.t
        obliquities[i] = np.abs(np.arccos(ps[1].params['tt_kz']))

    print("Integration time: "+str(time.time() - start)+" seconds")

    # plot
    fig, (ax) = plt.subplots(1, 1,figsize=(10, 6))

    ax.plot(ts/a**1.5, np.degrees(np.array(obliquities)), 'ko', label='Num')
    ax.set_ylabel("Obliquity (deg)")
    ax.set_xlabel('Time (orbital periods)')

    # ax.legend(['Numerical', 'Analytical'])

    plt.savefig('obl_damp_'+str(dt_frac)+'dt.png', dpi=300)
    plt.clf()

# def test_chandler():
    # PLOT PRECESSION
    # ax1.plot(times[:]/a**1.5, kxs, marker='o')
    # ax1.set_ylabel(r"$k_x$")
    # ax2.plot(times[:]/a**1.5, omega_xs, marker='o')
    # ax2.set_ylabel(r"$k_y$")
    # ax2.set_xlabel('Time (orbital periods)')

# main function
if __name__ == '__main__':

    # choose simulation params:
    a = .1
    Q_tide = .01
    R_p = 1.e-4 # ~ 20 earth radii
    obliquity = np.radians(1.)
    omega_to_n = 1.5 # omega / n
    M_p = 1.e-4 # in units of primary body's mass (~ 2 earth masses)
    k2 = 1.5 # 1.5 for uniformly distributed mass
    moment2 = 0 # 1e-2 # (Ij - Ii) / Ii, < moment3
    moment3 = 0 # 2e-2 # (Ik - Ii) / Ii, > moment2
    s_k_angle = np.radians(0.) # angle between s and k

    sim_params = (a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle)

    dt_frac = 0.01 # fraction of min(orbital period, spin period)
    tf = 100. # number of orbital periods
    step = 0.05 # fraction on orbital periods

    if sys.argv[1] == '-co':
        moment2 = 1e-2 # (Ij - Ii) / Ii, < moment3
        moment3 = 2e-2 # (Ik - Ii) / Ii, > moment2
        sim_params = (a,Q_tide,R_p,0,omega_to_n,M_p,k2,moment2,moment3,s_k_angle)
        dtmax = 0.1
        test_convergence(sim_params,10,dtmax)

    elif sys.argv[1] == '-sp':
        test_spin_damp(sim_params,dt_frac,step,tf)

    elif sys.argv[1] == '-ob':
        test_obl_damp(sim_params,dt_frac,step,tf)
