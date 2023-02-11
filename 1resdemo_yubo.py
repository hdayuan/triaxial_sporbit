'''
For fiducial parameters, Eqs (3-4) of Su & Lai predict precession rates:

spin-orbit period (yr):
>>> 2 * pi / (0.3 / (2 * 0.25) * Msun / (4 Mearth) * ((2 Rearth) / (0.1 AU))^3 * (G * Msun / (0.1 AU)^3)^(1/2) * 1.2) / yr
853.629035

orbit-orbit (yr):
>>> 2 * pi / (3 * Mjup / (4 * Msun) * ((0.1 AU) / (0.6 AU))^3 * (G * Msun / (0.1 AU)^3)^(1/2) * yr)
9543.457683

Thus, we should run the integration for ~3e4 yr
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)
plt.rc('figure', figsize=(8.0, 8.0), dpi=300)

import rebound
import reboundx
import time
import os, pickle, lzma

RSUN = 1 / 215.097 # AU
REARTH = 4.259e-5 # AU
MEARTH = 3.002e-6 # Msun
MJUP = 0.954e-3 # Msun
RJUP = 0.000467

def plot_cs2(
    # inner SE params
    m1=4 * MEARTH, r1=2 * REARTH,
        a1=0.1, e1=0, I1=0, Omega1=0, omega1=0, f1=0,
        P1_mult=0.8, # spin period (in units of orbital period)
        theta1=np.radians(10), # obliquity
        phi1=np.radians(0), # azimuthal angle of spin axis
        k1=0.25, # moment of inertia I = k1 * m1 * r1**2
        k2_1=0.3, # hydrostatic Love number: J2 = k2 / 3 * (Omega / Omega)_c^2
                # Top of page 2 of Su & Lai 2022
        Q1=1e2, # tidal Q
    # outer CJ params
    m2=MJUP, r2=RJUP,
        a2=0.6, e2=0, I2=np.radians(20), Omega2=0, omega2=0, f2=0,
    mstar=1, # host star mass (Msun)
    tmax=1e4, # end time of integration
    ntimes=1000, # number of points to integrate over
    dt_mult=0.2, # time step in units of inner planet period
    # seems not to have an effect?
    plot_fn='testo',
):
    sim = rebound.Simulation()
    sim.units = ('AU', 'Msun', 'yr')
    n1 = np.sqrt(sim.G * mstar / a1**3)
    spin1 = n1 / P1_mult

    # initialize rebound simulation & add particles & add tides/spin
    sim.add(m=mstar, r=RSUN)
    sim.add(m=m1, r=r1, a=a1, inc=I1, Omega=Omega1, omega=omega1, f=f1)
    sim.add(m=m2, r=r2, a=a2, inc=I2, Omega=Omega2, omega=omega2, f=f2)
    ps = sim.particles[1: ] # planet particles
    sim.integrator = 'whfast'
    sim.dt = ps[0].P * dt_mult
    sim.move_to_com()
    rebx = reboundx.Extras(sim)
    sf = rebx.load_force('tides_spin')
    rebx.add_force(sf)

    ps[0].params['Omega'] = rebound.spherical_to_xyz(
        magnitude=spin1, theta=theta1, phi=phi1)
    ps[0].params['I'] = k1 * ps[0].m * ps[0].r**2
    rebx.initialize_spin_ode(sf)
    ps[0].params['k2'] = k2_1
    ps[0].params['tau'] = 1 / (2 * Q1 * n1)

    # use "pickle" library to save multiple variables at once
    # lzma is a compression library to save space & speed up loads
    # you can probably copy as is
    pkl_fn = '%s.pkl' % plot_fn
    if not os.path.exists(pkl_fn):
        print('Running %s' % pkl_fn)
        # param definitions
        # integrate
        start_time = time.time()
        times = np.linspace(0, tmax, ntimes)
        Omega_planet = np.zeros((3, ntimes))
        orbit_normal = np.zeros((3, ntimes))
        lout_hat = np.zeros((3, ntimes))
        for i, t in enumerate(times):
            Omega_planet[:, i] = ps[0].params['Omega']
            orbit_normal[:, i] = ps[0].hvec
            lout_hat[:, i] = np.array([
                np.sin(ps[1].inc) * np.sin(ps[1].Omega),
                -np.sin(ps[1].inc) * np.cos(ps[1].Omega),
                np.cos(ps[1].inc)
            ])
            sim.integrate(t)
        print('Integration for %s took %.3f seconds'
              % (plot_fn, time.time() - start_time))
        with lzma.open(pkl_fn, 'wb') as f:
            pickle.dump((times, Omega_planet, orbit_normal, lout_hat), f)
    else:
        with lzma.open(pkl_fn, 'rb') as f:
            print('Loading %s' % pkl_fn)
            times, Omega_planet, orbit_normal, lout_hat = pickle.load(f)

    # postprocess to plot obliquity & spin rate
    def vec_dot(x, y):
        '''
        dot product of two timeseries: (3 x N) . (3 x N) => (N)
        maybe there's a better way to do this...
        '''
        num_el = np.shape(x)[1]
        z = np.zeros(num_el)
        for i in range(num_el):
            z[i] = np.dot(x[:, i], y[:, i])
        return z
    def vec_cross(x, y):
        ''' cross product of two time series '''
        num_el = np.shape(x)[1]
        z = np.zeros((3, num_el))
        for i in range(num_el):
            z[:, i] = np.cross(x[:, i], y[:, i])
        return z
    def vec_mag(x):
        return np.sqrt(vec_dot(x, x))

    # calculate some derived quantities to plot
    spin_rate = vec_mag(Omega_planet)
    s_hat = Omega_planet / spin_rate
    orbit_rate = vec_mag(orbit_normal)
    l_hat = orbit_normal / orbit_rate
    cos_obl = vec_dot(s_hat, l_hat)
    # calculate phi using the sign convention described after Eq (5) of SL22
    y_vec = vec_cross(lout_hat, l_hat)
    x_vec = vec_cross(y_vec, l_hat)
    x_hat = x_vec / vec_mag(x_vec)
    y_hat = y_vec / vec_mag(y_vec)
    s_x = vec_dot(s_hat, x_hat)
    s_y = vec_dot(s_hat, y_hat)
    phi = np.degrees(np.unwrap(np.arctan2(s_y, s_x)))

    def reg_arccos(x):
        ''' cosine complains when |x| > 1, so regularize values first '''
        return np.arccos(np.minimum(np.maximum(x, -1), 1))
    obl_d = np.degrees(reg_arccos(cos_obl))

    # plotting
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(8, 8),
        sharex=True)
    ax1.plot(times, obl_d)
    ax2.plot(times, spin_rate / n1)
    ax3.plot(times, phi)
    # # DEBUG
    # ax1.plot(times, s_hat[2], 'r', alpha=0.7)
    # ax2.plot(times, s_hat[0], 'r', alpha=0.7)
    # ax3.plot(times, s_hat[1], 'r', alpha=0.7)
    # ax1.plot(times, l_hat[2], 'k-.', alpha=0.7)
    # ax2.plot(times, l_hat[0], 'k-.', alpha=0.7)
    # ax3.plot(times, l_hat[1], 'k-.', alpha=0.7)
    # ax1.plot(times, lout_hat[2], 'g:', alpha=0.7)
    # ax2.plot(times, lout_hat[0], 'g:', alpha=0.7)
    # ax3.plot(times, lout_hat[1], 'g:', alpha=0.7)

    ax3.set_xlabel(r'$t$ [yr]')
    ax1.set_ylabel(r'$\theta$')
    ax2.set_ylabel(r'$\Omega / n$')
    ax3.set_ylabel(r'$\phi$')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    plt.savefig(plot_fn)
    plt.clf()

    plt.scatter(phi % 360, obl_d, c='b', s=2)
    plt.scatter(phi[0] % 360, obl_d[0], c='r')
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\theta$')
    plt.savefig(plot_fn + '_scatter')
    plt.close()

def make_phase_portrait(tmax=3e3):
    '''
    I was having trouble finding the resonance center, so I made this plot, and
    I think it's somewhat helpful to see to help visualize the resonance
    '''
    phis = np.arange(0, 360, 60)
    obls = np.arange(60, 100, 10)
    for phi in phis:
        for obl in obls:
            ret = plot_cs2(tmax=3e3, P1_mult=1, theta1=np.radians(obl),
                     phi1=np.radians(phi),
                     plot_fn='resportrait_%d_%d' % (phi, obl), Q1=np.inf)

            phi_d, obl_d = ret
            plt.scatter(phi_d % 360, obl_d, c='b', s=2)
            plt.scatter(phi_d[0] % 360, obl_d[0], c='r')

    plt.xlim(0, 360)
    plt.savefig('resportrait')
    plt.close()

if __name__ == '__main__':
    # this one should just have small obliquity oscillations and damp to zero
    plot_cs2(tmax=1e4, theta1=np.radians(30), plot_fn='secj_nores', Q1=1e3)

    # make a phase portrait of the spin evolution to find the resonant behavior
    # in the absence of dissipation (purely instructive)
    # make_phase_portrait()

    # Use Eq 40 to start at a tidally stable Cassini State
    # Use Eq 37 to start at the point where \dot{Omega} = 0
    theta1 = np.radians(76)
    P1_mult = (1 + np.cos(theta1)**2) / (2 * np.cos(theta1))
    # resonant case: notice the lack of obliquity damping
    # NB: there's a 90 degree offset in definition of phi compared to what I use
    # / define above... not sure why
    ret = plot_cs2(tmax=1e4, P1_mult=P1_mult, theta1=theta1,
                   phi1=np.radians(270), plot_fn='secj_res', Q1=1e3)
