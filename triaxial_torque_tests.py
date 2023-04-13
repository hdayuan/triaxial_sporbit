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
plt.rc('lines', lw=1.55)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

# make simulation object with given parameters
def create_sim(sim_params,dt_frac,dtheta_offset=0.,one_body=False):
    a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle = sim_params
    if dtheta_offset != 0. and obliquity != 0.:
        print("Error: either obliquity or dtheta_offset must be 0.")
        exit()

    sim = rebound.Simulation()
    sim.integrator = 'whfast'
    sim.units = ('AU', 'yr', 'MSun')
    
    if one_body:
        sim.add(m=M_p)
        i = 0

    else:
        sim.add(m=1.)
        sim.add(m=M_p, a=a)
        i = 1

    rebx = reboundx.Extras(sim)
    triax = rebx.load_operator('triaxial_torque')
    rebx.add_operator(triax)

    # add spin to smaller body
    ps = sim.particles

    if obliquity != 0:
        ps[i].params['tt_ix'] = np.cos(obliquity)
        ps[i].params['tt_iy'] = 0.
        ps[i].params['tt_iz'] = -np.sin(obliquity)
        ps[i].params['tt_jx'] = 0.
        ps[i].params['tt_jy'] = 1.
        ps[i].params['tt_jz'] = 0.
        ps[i].params['tt_kx'] = np.sin(obliquity)
        ps[i].params['tt_ky'] = 0.
        ps[i].params['tt_kz'] = np.cos(obliquity)
    else:
        ps[i].params['tt_ix'] = np.cos(dtheta_offset)
        ps[i].params['tt_iy'] = -np.sin(dtheta_offset)
        ps[i].params['tt_iz'] = 0.
        ps[i].params['tt_jx'] = np.sin(dtheta_offset)
        ps[i].params['tt_jy'] = np.cos(dtheta_offset)
        ps[i].params['tt_jz'] = 0.
        ps[i].params['tt_kx'] = 0.
        ps[i].params['tt_ky'] = 0.
        ps[i].params['tt_kz'] = 1.

    # (2/5)*MR^2
    Ii = (2/5)*M_p*R_p*R_p
    Ij = Ii*(1+moment2)
    Ik = Ii*(1+moment3)

    ps[i].params['tt_Ii'] = Ii
    ps[i].params['tt_Ij'] = Ij
    ps[i].params['tt_Ik'] = Ik

    ps[i].params['tt_si'] = np.sin(s_k_angle)
    ps[i].params['tt_sj'] = 0.
    ps[i].params['tt_sk'] = np.cos(s_k_angle)

    if one_body:
        tidal_dt = np.arctan(1./Q_tide) / 2.
        omega = omega_to_n*2*np.pi

        ps[i].params['tt_omega'] = omega
        ps[i].params['tt_R'] = R_p
        ps[i].params['tt_k2'] = k2
        ps[i].params['tt_tidal_dt'] = tidal_dt

        sim.dt = dt_frac*2*np.pi/omega
    else:
        tidal_dt = np.arctan(1./Q_tide) / 2. / ps[1].n # check this / change n to some other frequency?
        omega = omega_to_n*ps[1].n

        ps[i].params['tt_omega'] = omega
        ps[i].params['tt_R'] = R_p
        ps[i].params['tt_k2'] = k2
        ps[i].params['tt_tidal_dt'] = tidal_dt

        if omega == 0:
            sim.dt = dt_frac*ps[1].P
        else:
            sim.dt = dt_frac*np.minimum(ps[1].P, 2*np.pi/omega)

    # print(np.degrees((omega-ps[1].n)*tidal_dt))

    return sim

# tests convergence order and makes plot
# add option to plot the oscillations / make its own function
def test_convergence(sim_params, n_dts, dtmax, tf_small, tf_big, plot_angle=False, dtheta=np.radians(0.1),n_data=51):

    a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle = sim_params
    k2 = 0. # turn off tides
    omega_to_n = 1. # make sure synchronous
    obliquity = 0. # make sure obliquity is 0

    # (2/5)*MR^2
    Ii = (2/5)*M_p*R_p*R_p
    Ij = Ii*(1+moment2)
    Ik = Ii*(1+moment3)

    sim_params = (a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle)

    if plot_angle:
        sim = create_sim(sim_params,dtmax,dtheta_offset=dtheta)
        ps = sim.particles

        out_step = (tf_big / (n_data-1))*a**1.5

        ts = np.zeros(n_data)
        angles = np.zeros(n_data)
        # run simulation
        for i in range(n_data):
            sim.integrate(i*out_step)
            ts[i] = sim.t
            rx = ps[0].x - ps[1].x
            ry = ps[0].y - ps[1].y
            rz = ps[0].z - ps[1].z
            r = np.sqrt(rx**2 + ry**2 +rz**2)
            rx /= r
            ry /= r
            rz /= r
            i_dot_r = ps[1].params['tt_ix']*rx + ps[1].params['tt_iy']*ry + ps[1].params['tt_iz']*rz
            j_dot_r = ps[1].params['tt_jx']*rx + ps[1].params['tt_jy']*ry + ps[1].params['tt_jz']*rz
            pi_angle = np.arctan2(j_dot_r, i_dot_r)
            if pi_angle > 0:
                angles[i] = pi_angle - np.pi
            else:
                angles[i] = np.pi + pi_angle


        # analytical solution
        ts_fine = np.linspace(0,tf_big,200)
        # only correct for M_star = 1, circular orbit
        freq = np.sqrt(3*sim.G*(Ij-Ii)/Ik/(a**3))
        angles_true = np.degrees(dtheta * np.cos(freq * ts_fine*a**1.5))

        # plot
        fig, ax = plt.subplots(1, 1,figsize=(12, 4))
        plt.subplots_adjust(left=0.10, bottom=0.15, right=.98, top=0.95, wspace=0.05, hspace=0.)
        ax.set_ylabel(r"$\psi$ ($^{\circ}$)", fontsize=20)
        ax.set_xlabel(r"Time ($P$)", fontsize=20)
        ax.plot(ts_fine, angles_true, c="tab:red",lw=1.5)
        ax.plot(ts/(a**1.5), np.degrees(angles), 'ko')

        plt.savefig('oscillation_'+str(dtmax)+'dt.png', dpi=300)
        plt.clf()

    else:
        dts = dtmax / 2**np.arange(int(n_dts-1), -1, -1) # fraction of min(orbital period, spin period)
        errors = np.zeros(n_dts)
        tfs = [tf_small*a**1.5,tf_big*a**1.5]

        # make plot
        fig, ax = plt.subplots(1, 2,figsize=(14, 6),sharey=True)
        plt.subplots_adjust(left=0.08, bottom=0.15, right=.98, top=0.9, wspace=0.05, hspace=0.)
        # ax.set_yscale('log')
        # ax.plot(dts, final_residuals[n_dts-1] * (dts / dts[n_dts-1])**2, c='y', lw=0.5,
        #         label='2nd order')
        # ax.plot(dts, final_residuals[n_dts-1] * (dts / dts[n_dts-1])**3, c='b', lw=0.5,
        #         label='3rd order')
        ax[0].set_title(r"$t_f/P_s=$ "+str(tf_small))
        ax[1].set_title(r"$t_f/P_s=$ "+str(tf_big))
        ax[0].set_ylabel("Error", fontsize=20)

        for j in range(2):
            for i in range(n_dts):
                # create simulation
                sim = create_sim(sim_params,dts[i],dtheta_offset=dtheta)
                # run simulation
                sim.integrate(tfs[j])
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
                pi_angle = np.arctan2(j_dot_r, i_dot_r)
                if pi_angle > 0:
                    angle = pi_angle - np.pi
                else:
                    angle = np.pi + pi_angle
                # only correct for M_star = 1, circular orbit
                freq = np.sqrt(3*sim.G*(Ij-Ii)/Ik/(a**3))
                exact_sol = dtheta * np.cos(freq * tfs[j]) # figure out why I need negative sign??
                # ##
                # if i == 0:
                #     fig, (ax) = plt.subplots(1, 1,figsize=(10, 6))
                #     ts = np.linspace(0, 10, 100)
                #     ax.plot(ts, dtheta * np.cos(freq * ts*a**1.5))
                #     plt.savefig('test.png', dpi=300)
                #     plt.clf()
                # ##
                errors[i] = np.abs((angle - exact_sol) / exact_sol)
            
            ax[j].loglog(dts, errors[n_dts-1] * (dts / dts[n_dts-1])**4, c='tab:red', lw=1.5,
                    label='4th order')
            ax[j].loglog(dts, errors, 'o',c='black', label='Numerical Error')
            ax[j].legend(fontsize=20)
            ax[j].set_xlabel(r"$\Delta t/P_s$", fontsize=20)

        
        plt.savefig('convergence.png', dpi=300)
        plt.clf()

def test_spin_damp(sim_params, dt_frac, n_data, tf_frac, dual=False):

    start = time.time()

    a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle = sim_params

    # create simulation
    sim = create_sim(sim_params,dt_frac)

    out_step = (tf_frac / (n_data-1))*a**1.5

    if out_step < sim.dt:
        print("ERROR: output step is smaller than maximum timestep")
        exit()

    ps = sim.particles
    mm = ps[1].n
    ts = np.zeros(n_data)
    omegas = np.zeros(n_data)
    ns = np.zeros(n_data)
    for i in range(n_data):
        sim.integrate(i*out_step)
        ts[i] = sim.t
        omegas[i] = ps[1].params['tt_omega']
        ns[i] = ps[1].n

    print("Integration time: "+str(time.time() - start)+" seconds")

    # plot
    if dual:
        fig, (ax1,ax2) = plt.subplots(2, 1,figsize=(5, 6),sharex=True)
        plt.subplots_adjust(left=0.20, bottom=0.1, right=.98, top=0.98, wspace=0.05, hspace=0.05)
        tidal_dt = np.arctan(1./Q_tide) / 2. / mm
        omega0 = omega_to_n*mm
        theta_lag = (omega0 - mm) * tidal_dt

        # in case spin is very large
        max_theta_lag = np.pi/4
        if theta_lag > max_theta_lag:
            print("WARNING: Max tidal angle lag exceeded!")
            theta_lag = max_theta_lag
        if theta_lag < -max_theta_lag:
            print("WARNING: Min tidal angle lag exceeded!")
            theta_lag = -max_theta_lag

        ts_fine = np.linspace(0,tf_frac,200)
        
        omega_dot = -15.*k2*sim.G*R_p**3*np.cos(theta_lag)*np.sin(theta_lag)/(2.*M_p*a**6)
        exact_sol = (omega0 + ts_fine*a**1.5*omega_dot) / mm

        ax1.plot(ts_fine, exact_sol, color="tab:red",lw=1.5)
        ax1.plot(ts/(a**1.5), np.array(omegas/mm), 'ko')
        ax1.legend(['Analytical','Numerical'])
        ax1.set_ylabel(r"$\Omega/n$",fontsize=20)

        # run second simulation
        start = time.time()

        omega_to_n = 2 - omega_to_n
        sim_params = a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle
        # create simulation
        sim = create_sim(sim_params,dt_frac)

        if out_step < sim.dt:
            print("ERROR: output step is smaller than maximum timestep")
            exit()

        ps = sim.particles
        mm = ps[1].n
        ts = np.zeros(n_data)
        omegas = np.zeros(n_data)
        ns = np.zeros(n_data)
        for i in range(n_data):
            sim.integrate(i*out_step)
            ts[i] = sim.t
            omegas[i] = ps[1].params['tt_omega']
            ns[i] = ps[1].n

        print("Integration time: "+str(time.time() - start)+" seconds")
        
        tidal_dt = np.arctan(1./Q_tide) / 2. / mm
        omega0 = omega_to_n*mm
        theta_lag = (omega0 - mm) * tidal_dt

        # in case spin is very large
        max_theta_lag = np.pi/4
        if theta_lag > max_theta_lag:
            print("WARNING: Max tidal angle lag exceeded!")
            theta_lag = max_theta_lag
        if theta_lag < -max_theta_lag:
            print("WARNING: Min tidal angle lag exceeded!")
            theta_lag = -max_theta_lag
        
        omega_dot = -15.*k2*sim.G*R_p**3*np.cos(theta_lag)*np.sin(theta_lag)/(2.*M_p*a**6)
        exact_sol = (omega0 + ts_fine*a**1.5*omega_dot) / mm

        ax2.plot(ts_fine, exact_sol, color="tab:red",lw=1.5)
        ax2.plot(ts/(a**1.5), np.array(omegas/mm), 'ko')
        ax2.set_ylabel(r"$\Omega/n$", fontsize=20)

        ax2.set_xlabel(r'Time ($P$)',fontsize=20)

        plt.savefig('spin_damp_'+str(dt_frac)+'dt.png', dpi=300)
        plt.clf()
    else:
        fig, (ax) = plt.subplots(1, 1,figsize=(10, 6))
        plt.subplots_adjust(left=0.10, bottom=0.15, right=.98, top=0.95, wspace=0.05, hspace=0.)

        tidal_dt = np.arctan(1./Q_tide) / 2. / mm
        omega0 = omega_to_n*mm
        theta_lag = (omega0 - mm) * tidal_dt

        # in case spin is very large
        max_theta_lag = np.pi/4
        if theta_lag > max_theta_lag:
            print("WARNING: Max tidal angle lag exceeded!")
            theta_lag = max_theta_lag
        if theta_lag < -max_theta_lag:
            print("WARNING: Min tidal angle lag exceeded!")
            theta_lag = -max_theta_lag

        ts_fine = np.linspace(0,tf_frac,200)
        
        omega_dot = -15.*k2*sim.G*R_p**3*np.cos(theta_lag)*np.sin(theta_lag)/(2.*M_p*a**6)
        exact_sol = (omega0 + ts_fine*a**1.5*omega_dot) / mm

        ax.plot(ts_fine, exact_sol, color="tab:red",lw=1.5)
        ax.plot(ts/(a**1.5), np.array(omegas/mm), 'ko')
        ax.legend(['Analytical','Numerical'])
        ax.set_ylabel(r"$\Omega/n$",fontsize=20)
        ax.set_xlabel(r'Time ($P$)',fontsize=20)

        plt.savefig('spin_damp_'+str(dt_frac)+'dt.png', dpi=300)
        plt.clf()

    # # theoretical spin damp from su 2022
    # t_tide = (8/15)*Q_tide/k2/mm * M_p*(a/R_p)**3
    # om_dot = (1/t_tide)*(2*(mm-omega0))
    # su_sol = (omega0 + ts_fine*a**1.5*om_dot) / mm
    # ax.plot(ts_fine, su_sol, color='green',lw=1.5)
    # #
    # print(np.mean((omega_to_n - su_sol[1:])/(omega_to_n - exact_sol[1:])))

# add analytical solution to obliquity
def test_obl_damp(sim_params, dt_frac, n_data, tf_frac, dual=False):
    
    start = time.time()
    a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle = sim_params

    # create simulation
    sim = create_sim(sim_params,dt_frac)

    out_step = (tf_frac / (n_data-1))*a**1.5

    if out_step < sim.dt:
        print("ERROR: output step is smaller than maximum timestep")
        exit()

    ps = sim.particles
    mm = ps[1].n
    ts = np.zeros(n_data)
    obliquities = np.zeros(n_data)
    for i in range(n_data):
        sim.integrate(i*out_step)
        ts[i] = sim.t
        obliquities[i] = np.abs(np.arccos(ps[1].params['tt_kz']))

    print("Integration time: "+str(time.time() - start)+" seconds")

    if dual:
        fig, (ax1,ax2) = plt.subplots(2, 1,figsize=(5, 6),sharex=True)
        plt.subplots_adjust(left=0.20, bottom=0.1, right=.98, top=0.98, wspace=0.05, hspace=0.05)
        tidal_dt = np.arctan(1./Q_tide) / 2. / mm
        omega0 = omega_to_n*mm
        theta_lag = (omega0 - mm) * tidal_dt

        # in case spin is very large
        max_theta_lag = np.pi/4
        if theta_lag > max_theta_lag:
            print("WARNING: Max tidal angle lag exceeded!")
            theta_lag = max_theta_lag
        if theta_lag < -max_theta_lag:
            print("WARNING: Min tidal angle lag exceeded!")
            theta_lag = -max_theta_lag

        ts_fine = np.linspace(0,tf_frac,200)
        t_tide = (8/15)*Q_tide/k2/mm * M_p*(a/R_p)**3
        theta_dot = -np.sin(obliquity)/t_tide * ((2/omega_to_n)-np.cos(obliquity))
        exact_sol = np.degrees(obliquity + (ts_fine*a**1.5 * theta_dot))
        ax1.plot(ts_fine, exact_sol, color="tab:red",lw=1.5)
        ax1.plot(ts/a**1.5, np.degrees(np.array(obliquities)), 'ko', label='Num')
        ax1.set_ylabel(r"$\theta$ ($^{\circ}$)",fontsize=20)
        ax1.legend(['Analytical','Numerical'])

        # run second simulation
        start = time.time()

        omega_to_n = 4 - omega_to_n
        sim_params = a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle
        # create simulation
        sim = create_sim(sim_params,dt_frac)

        if out_step < sim.dt:
            print("ERROR: output step is smaller than maximum timestep")
            exit()

        ps = sim.particles
        mm = ps[1].n
        ts = np.zeros(n_data)
        obliquities = np.zeros(n_data)
        for i in range(n_data):
            sim.integrate(i*out_step)
            ts[i] = sim.t
            obliquities[i] = np.abs(np.arccos(ps[1].params['tt_kz']))

        print("Integration time: "+str(time.time() - start)+" seconds")
        
        tidal_dt = np.arctan(1./Q_tide) / 2. / mm
        omega0 = omega_to_n*mm
        theta_lag = (omega0 - mm) * tidal_dt

        print(theta_lag/(2*np.pi))

        # in case spin is very large
        max_theta_lag = np.pi/4
        if theta_lag > max_theta_lag:
            print("WARNING: Max tidal angle lag exceeded!")
            theta_lag = max_theta_lag
        if theta_lag < -max_theta_lag:
            print("WARNING: Min tidal angle lag exceeded!")
            theta_lag = -max_theta_lag
        
        theta_dot = -np.sin(obliquity)/t_tide * ((2/omega_to_n)-np.cos(obliquity))
        exact_sol = np.degrees(obliquity + (ts_fine*a**1.5 * theta_dot))
        ax2.plot(ts_fine, exact_sol, color="tab:red",lw=1.5)
        ax2.plot(ts/a**1.5, np.degrees(np.array(obliquities)), 'ko', label='Num')
        ax2.set_ylabel(r"$\theta$ ($^{\circ}$)",fontsize=20)

        ax2.set_xlabel(r'Time ($P$)',fontsize=20)

        plt.savefig('obl_damp_'+str(dt_frac)+'dt.png', dpi=300)
        plt.clf()
    
    else:
        # plot
        fig, (ax) = plt.subplots(1, 1,figsize=(10, 6))
        plt.subplots_adjust(left=0.10, bottom=0.15, right=.98, top=0.95, wspace=0.05, hspace=0.)

        ts_fine = np.linspace(0,tf_frac,200)
        t_tide = (8/15)*Q_tide/k2/mm * M_p*(a/R_p)**3
        theta_dot = -np.sin(obliquity)/t_tide * ((2/omega_to_n)-np.cos(obliquity))
        exact_sol = np.degrees(obliquity + (ts_fine*a**1.5 * theta_dot))
        ax.plot(ts_fine, exact_sol, color="tab:red",lw=1.5)
        ax.plot(ts/a**1.5, np.degrees(np.array(obliquities)), 'ko', label='Num')
        ax.set_ylabel(r"$\theta$ ($^{\circ}$)",fontsize=20)
        ax.set_xlabel(r'Time ($P$)',fontsize=20)
        ax.legend(['Numerical', 'Analytical'])

        plt.savefig('obl_damp_'+str(dt_frac)+'dt.png', dpi=300)
        plt.clf()

def test_chandler(sim_params, dt_frac, n_data, tf_frac):

    ind = 0

    start = time.time()

    a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle = sim_params

    spin_period = 1./omega_to_n # given omega = omega_to_n*2*pi (see create_sim())

    # create simulation
    sim = create_sim(sim_params,dt_frac,one_body=True)

    out_step = (tf_frac / (n_data-1))*spin_period

    if out_step < sim.dt:
        print("ERROR: output step is smaller than timestep")
        exit()

    ps = sim.particles

    omega0 = ps[ind].params['tt_omega']
    si0 = ps[ind].params['tt_si']
    sj0 = ps[ind].params['tt_sj']
    sk0 = ps[ind].params['tt_sk']

    ts = np.zeros(n_data)
    sis = np.zeros(n_data)
    sjs = np.zeros(n_data)
    sks = np.zeros(n_data)
    omegas = np.zeros(n_data)
    for i in range(n_data):
        sim.integrate(i*out_step)
        ts[i] = sim.t
        sis[i] = ps[ind].params['tt_si']
        sjs[i] = ps[ind].params['tt_sj']
        sks[i] = ps[ind].params['tt_sk']
        omegas[i] = ps[ind].params['tt_omega']

    print("Integration time: "+str(time.time() - start)+" seconds")

    # Plot
    fig, ax = plt.subplots(4, 1,figsize=(10, 10),sharex=True)
    plt.subplots_adjust(left=0.1, bottom=0.08, right=.98, top=0.98, wspace=0., hspace=0.05)
    ax[3].set_xlabel(r'Time ($P_s$)', fontsize=20)

    freq = moment3*omega0*sk0
    ts_fine = np.linspace(0,tf_frac,200)
    si_true = si0*np.cos(freq*ts_fine)
    sj_true = si0*np.sin(freq*ts_fine)
    sk_true = sk0*np.ones_like(ts_fine)
    omega_true = omega0*np.ones_like(ts_fine)

    ax[0].plot(ts_fine, si_true, c="tab:red",lw=1.5)
    ax[0].plot(ts/spin_period, sis, 'o', c='black')
    ax[0].set_ylabel(r"$s_i$", fontsize=20)

    ax[1].plot(ts_fine, sj_true, c="tab:red",lw=1.5)
    ax[1].plot(ts/spin_period, sjs, 'o', c='black')
    ax[1].set_ylabel(r"$s_j$", fontsize=20)

    ax[2].plot(ts_fine, sk_true, c="tab:red",lw=1.5)
    ax[2].plot(ts/spin_period, sks, 'o', c='black')
    ax[2].set_ylabel(r"$s_k$", fontsize=20)
    ax[2].set_ylim(ymin=sk0 - 0.05,ymax=sk0 + 0.05)

    ax[3].plot(ts_fine, omega_true/omega0, c="tab:red",lw=1.5)
    ax[3].plot(ts/spin_period, omegas/omega0, 'o', c='black')
    ax[3].set_ylabel(r"$\Omega / \Omega_0$", fontsize=20)
    ax[3].set_ylim(ymin=0., ymax=2.)

    plt.savefig('chandler_'+str(dt_frac)+'dt.png', dpi=300)
    plt.clf()

# main function
if __name__ == '__main__':

    # choose simulation params:
    a = .1
    Q_tide = 10.
    R_p = 1.e-4 # ~ 2 earth radii
    obliquity = 0.
    omega_to_n = 1. # omega / n
    M_p = 1.e-4 # in units of primary body's mass (~ 2 earth masses)
    k2 = 1.5 # 1.5 for uniformly distributed mass
    moment2 = 1e-1 # (Ij - Ii) / Ii, < moment3
    moment3 = 2e-1 # (Ik - Ii) / Ii, > moment2
    s_k_angle = np.radians(0.) # angle between s and k

    dt_frac = 0.05 # fraction of min(orbital period, spin period)
    tf = 100. # number of orbital periods
    step = 0.05 # fraction of orbital periods

    if sys.argv[1] == '-convergence':
        # change params as needed
        k2 = 0.

        # create sim_params
        sim_params = (a,Q_tide,R_p,0,omega_to_n,M_p,k2,moment2,moment3,s_k_angle)

        dtmax = 0.05
        tf_small = dtmax
        tf_big = 10.

        test_convergence(sim_params,8,dtmax,tf_small,tf_big)

    elif sys.argv[1] == '-oscillation':
        # change params as needed
        k2 = 0.

        # create sim_params
        sim_params = (a,Q_tide,R_p,0,omega_to_n,M_p,k2,moment2,moment3,s_k_angle)

        dtmax = 0.05
        tf_small = dtmax
        tf_big = 10.
        nd = 51

        dtheta_offset = np.radians(1.)

        test_convergence(sim_params,8,dtmax,tf_small,tf_big,plot_angle=True,dtheta=dtheta_offset,n_data=nd)

    elif sys.argv[1] == '-spin':
        # change params as needed
        # turn off triaxial torque
        moment2 = 0
        moment3 = 0
        omega_to_n = 1.5

        # create sim_params
        sim_params = (a,Q_tide,R_p,0,omega_to_n,M_p,k2,moment2,moment3,s_k_angle)

        dt_frac = 0.05 # fraction of min(orbital period, spin period)
        tf = 1000. # number of orbital periods
        n_data = 20 # number of data points

        test_spin_damp(sim_params,dt_frac,n_data,tf,dual=True)

    elif sys.argv[1] == '-obliquity':
        # change params as needed
        # turn off triaxial torque
        moment2 = 0
        moment3 = 0
        obliquity = np.radians(10.)
        omega_to_n = 1.5

        # create sim_params
        sim_params = (a,Q_tide,R_p,obliquity,omega_to_n,M_p,k2,moment2,moment3,s_k_angle)

        dt_frac = 0.05 # fraction of min(orbital period, spin period)
        tf = 1000. # number of orbital periods
        n_data = 20 # number of data points

        test_obl_damp(sim_params,dt_frac,n_data,tf,dual=True)

    elif sys.argv[1] == '-chandler':
        # omega_to_n is just omega/(2pi) here because only 1 body
        # dt_frac and tf is fraction of spin period because no orbit

        # change params as needed
        moment2 = 0. # (Ij - Ii) / Ii, < moment3
        moment3 = 1e-1 # (Ik - Ii) / Ii, > moment2
        s_k_angle = np.radians(10.)
        omega_to_n = 1.

        # create sim_params
        sim_params = (a,Q_tide,R_p,0,omega_to_n,M_p,k2,moment2,moment3,s_k_angle)

        tf = 50.
        n_data = 35 # number of data points

        test_chandler(sim_params,dt_frac,n_data,tf)
