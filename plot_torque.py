import os
import rebound
import reboundx
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

tf = 200
step = 0.05
n = int(tf / step)

def run(dt, dtheta_offset=np.radians(1.), to_plot=True):
    start = time.time()

    sim = rebound.Simulation()
    sim.integrator = 'whfast'
    sim.units = ('AU', 'yr', 'MSun')
    sim.dt = dt
    sim.add(m=1.)

    # change offset to 0 if using force
    v = 6.286207389817359
    d_theta = v * sim.dt
    # x_val = 1. # np.cos(d_theta)
    # y_val = 0. # -np.sin(d_theta)
    # vx_val = 0. # v*np.sin(d_theta)
    # vy_val = v # v*np.cos(d_theta)
    x_val = np.cos(d_theta)
    y_val = -np.sin(d_theta)
    vx_val = v*np.sin(d_theta)
    vy_val = v*np.cos(d_theta)

    sim.add(m=0.00001, x=x_val,y=y_val,vx=vx_val,vy=vy_val)
    # sim.add(m=0.001, a=1.)

    rebx = reboundx.Extras(sim)
    triax = rebx.load_operator('triaxial_torque') # change if force / operator
    rebx.add_operator(triax) # change if force / operator

    # add spin to smaller body
    ps = sim.particles

    ps[1].params['tt_ix'] = np.cos(dtheta_offset)
    ps[1].params['tt_iy'] = np.sin(dtheta_offset)
    ps[1].params['tt_iz'] = 0.
    ps[1].params['tt_jx'] = -np.sin(dtheta_offset)
    ps[1].params['tt_jy'] = np.cos(dtheta_offset)
    ps[1].params['tt_jz'] = 0.
    ps[1].params['tt_kx'] = 0.
    ps[1].params['tt_ky'] = 0.
    ps[1].params['tt_kz'] = 1.

    # (2/5)*MR^2
    Ii = 1.e-13
    Ij = Ii + 1.e-14
    Ik = Ii + 2.e-14

    ps[1].params['tt_Ii'] = Ii
    ps[1].params['tt_Ij'] = Ij
    ps[1].params['tt_Ik'] = Ik

    ps[1].params['tt_si'] = 0.
    ps[1].params['tt_sj'] = 0.
    ps[1].params['tt_sk'] = 1.

    Q = 1.e-20
    tidal_dt = 1 / Q / ps[1].n
    omega = 2*np.pi / ps[1].P
    ps[1].params['tt_omega'] = omega
    ps[1].params['tt_R'] = 0.001
    ps[1].params['tt_k2'] = 0.5
    ps[1].params['tt_tidal_dt'] = tidal_dt

    filename = 'test_torque_out_%.10fdt' % sim.dt
    f = open(filename + '.txt', 'w')

    times = []
    angs = []
    rxs = []
    ixs = []

    for i in range(n):
        sim.integrate(i*step)
        rx = ps[0].x - ps[1].x
        ry = ps[0].y - ps[1].y
        rz = ps[0].z - ps[1].z
        r = np.sqrt(rx**2 + ry**2 +rz**2)
        rx /= r
        ry /= r
        rz /= r
        i_dot_r = ps[1].params['tt_ix']*rx + ps[1].params['tt_iy']*ry + ps[1].params['tt_iz']*rz
        j_dot_r = ps[1].params['tt_jx']*rx + ps[1].params['tt_jy']*ry + ps[1].params['tt_jz']*rz
        ang = np.arctan2(j_dot_r, i_dot_r)
        angs.append(ang)
        f.write(str(ang)+'\t')
        f.write(str(ps[1].params['tt_omega'])+'\t')
        f.write(str(ps[1].params['tt_ix'])+'\t')
        f.write(str(ps[1].params['tt_iy'])+'\t')
        f.write(str(ps[1].params['tt_iz'])+'\t')
        f.write(str(ps[1].params['tt_jx'])+'\t')
        f.write(str(ps[1].params['tt_jy'])+'\t')
        f.write(str(ps[1].params['tt_jz'])+'\t')
        f.write(str(ps[1].params['tt_kx'])+'\t')
        f.write(str(ps[1].params['tt_ky'])+'\t')
        f.write(str(ps[1].params['tt_kz'])+'\t')
        f.write(str(rx)+'\t')
        f.write(str(ry)+'\t')
        f.write(str(rz)+'\t')
        f.write(str(sim.t)+'\n')
        times.append(sim.t)

        rxs.append(rx)
        ixs.append(ps[1].params['tt_ix'])

    times = np.array(times)
    if to_plot:
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(8, 8),
            sharex=True)

        freq = np.sqrt(3*sim.G*(Ij-Ii)/Ik)
        exact_sol = np.pi - dtheta_offset * np.cos(freq * times)

        ax1.plot(times[1:], np.unwrap(angs)[1:], 'ko', label='Num')
        ax1.plot(times, exact_sol, label='Exact')
        ax1.set_title('%.10f' % dt)
        ax1.set_ylabel('theta')
        ax2.plot(times[1: ], (np.unwrap(angs) - exact_sol)[1: ])
        ax2.set_ylabel('Residuals')
        ax2.set_xlabel('Time')
        plt.savefig(filename + '.png', dpi=300)
        plt.clf()

    f.close()
    print('Running for %.10f took %.7f s' % (dt, time.time() - start))

def zero_offset_assert():
    '''
    assert that with zero offset, we get very small oscillations
    '''
    dt = 1e-4 # some small dt

    run(dt, dtheta_offset=0, to_plot=False)
    outf = 'test_torque_out_%.10fdt.txt' % dt
    f = open(outf, 'r')
    allLines = f.readlines()

    angles = np.zeros(n)
    for i in range(n):
        datum = allLines[i].split()
        angles[i] = float(datum[0])
    angles = angles[1: ] # remove leading element

    assert np.min(angles) - np.max(angles) < 1e-3
    print('Passed zero torque test case!!')

if __name__ == '__main__':
    zero_offset_assert()

    dtmax = step
    dts = dtmax / 2**np.arange(4, -1, -1)
    # dts = np.array([step, step / 2])
    outfs = []
    for dt in dts:
        outf = 'test_torque_out_%.10fdt.txt' % dt
        run(dt)
        outfs.append(outf)

    nv = 15

    angle = 0
    omega = 1
    t = nv - 1
    ##################################

    f = open(outfs[0], 'r')
    allLines = f.readlines()

    data = np.zeros((n, nv))

    for i in range(n):
        datum = allLines[i].split()
        for j in range(nv):
            data[i,j] = float(datum[j])

    true_vals = np.unwrap(data[ :, angle])[1: ]

    rms = np.zeros(len(outfs)-1)
    rms_an = np.zeros(len(outfs)-1)
    means = np.zeros(len(outfs)-1)
    mins = np.zeros(len(outfs)-1)
    maxes = np.zeros(len(outfs)-1)

    for i in range(1,len(outfs)):
        f = open(outfs[i], 'r')
        allLines = f.readlines()

        data = np.zeros((n, nv))

        for j in range(n):
            datum = allLines[j].split()
            for k in range(nv):
                data[j,k] = float(datum[k])

        ang = np.unwrap(data[ :, angle])[1: ]

        rms[i-1] = np.sqrt(np.sum((ang-true_vals)**2)) / len(ang)
        # rms[i-1] = np.abs(ang-true_vals)[-1]
        means[i-1] = np.mean(ang)
        mins[i-1] = np.min(ang[1: ])
        maxes[i-1] = np.max(ang[1: ])

    fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(8, 8),sharex=True)
    ax1.loglog(dts[1:], rms, 'go', label='RMS (data)')
    ax1.set_yscale('log')
    ylims = ax1.get_ylim()
    ax1.plot(dts[1: ], rms[3] * (dts[1: ] / dts[4]), c='r', lw=0.5,
             label='1-order')
    ax1.plot(dts[1: ], rms[3] * (dts[1: ] / dts[4])**2, c='y', lw=0.5,
             label='2-order')
    ax1.plot(dts[1: ], rms[3] * (dts[1: ] / dts[4])**3, c='b', lw=0.5,
             label='3-order')
    ax1.legend(fontsize=12)
    ax1.set_ylim(ylims)

    ax2.plot(dts[1: ], means, 'b')
    ax2.plot(dts[1: ], mins, 'g--', lw=0.5)
    ax2.plot(dts[1: ], maxes, 'g--', lw=0.5)
    ax2.set_ylabel('Min/Mean/Max')

    ax1.set_ylabel('RMS (radians)')
    ax2.set_xlabel('dt (years)')
    ax2.set_xscale('log')
    ax1.axvline(step, c='k')
    ax2.axvline(step, c='k')

    plt.tight_layout()
    plt.savefig('torque', dpi=300)
