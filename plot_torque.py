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

tf = 10
step = 0.01
n = int(tf / step)

def run(dt):
    to_plot = False
    start = time.time()

    sim = rebound.Simulation()
    sim.integrator = 'whfast'
    sim.units = ('AU', 'yr', 'MSun')
    sim.dt = dt
    sim.add(m=1.)

    # change offset to 0 if using force
    v = 6.286207389817359
    d_theta = v * sim.dt + np.radians(1)
    # x_val = 1. # np.cos(d_theta)
    # y_val = 0. # -np.sin(d_theta)
    # vx_val = 0. # v*np.sin(d_theta)
    # vy_val = v # v*np.cos(d_theta)
    x_val = np.cos(d_theta)
    y_val = -np.sin(d_theta)
    vx_val = v*np.sin(d_theta)
    vy_val = v*np.cos(d_theta)

    sim.add(m=0.001, x=x_val,y=y_val,vx=vx_val,vy=vy_val)
    # sim.add(m=0.001, a=1.)

    rebx = reboundx.Extras(sim)
    triax = rebx.load_operator('triaxial_torque') # change if force / operator
    rebx.add_operator(triax) # change if force / operator

    # add spin to smaller body
    ps = sim.particles
    angle = 0.*np.pi/180.


    ps[1].params['tt_ix'] = np.cos(angle)
    ps[1].params['tt_iy'] = np.sin(angle)
    ps[1].params['tt_iz'] = 0.
    ps[1].params['tt_jx'] = -np.sin(angle)
    ps[1].params['tt_jy'] = np.cos(angle)
    ps[1].params['tt_jz'] = 0.
    ps[1].params['tt_kx'] = 0.
    ps[1].params['tt_ky'] = 0.
    ps[1].params['tt_kz'] = 1.

    ps[1].params['tt_Ii'] = 1.
    ps[1].params['tt_Ij'] = 2.
    ps[1].params['tt_Ik'] = 3.

    ps[1].params['tt_si'] = 0.
    ps[1].params['tt_sj'] = 0.
    ps[1].params['tt_sk'] = 1.

    ps[1].params['tt_omega'] = 2*np.pi / ps[1].P

    filename = 'test_torque_out_%.10fdt' % sim.dt
    f = open(filename + '.txt', 'w')

    #############################################################

    # sim.integrate(2*sim.dt)
    # rx = ps[0].x - ps[1].x
    # ry = ps[0].y - ps[1].y
    # rz = ps[0].z - ps[1].z
    # r = np.sqrt(rx**2 + ry**2 +rz**2)
    # rx /= r
    # ry /= r
    # rz /= r
    # i_dot_r = ps[1].params['tt_ix']*rx + ps[1].params['tt_iy']*ry + ps[1].params['tt_iz']*rz
    # j_dot_r = ps[1].params['tt_jx']*rx + ps[1].params['tt_jy']*ry + ps[1].params['tt_jz']*rz
    # f.write(str(np.arctan(j_dot_r/i_dot_r))+'\t')
    # f.write(str(ps[1].params['tt_omega'])+'\t')
    # f.write(str(ps[1].params['tt_ix'])+'\t')
    # f.write(str(ps[1].params['tt_iy'])+'\t')
    # f.write(str(ps[1].params['tt_iz'])+'\t')
    # f.write(str(ps[1].params['tt_jx'])+'\t')
    # f.write(str(ps[1].params['tt_jy'])+'\t')
    # f.write(str(ps[1].params['tt_jz'])+'\t')
    # f.write(str(ps[1].params['tt_kx'])+'\t')
    # f.write(str(ps[1].params['tt_ky'])+'\t')
    # f.write(str(ps[1].params['tt_kz'])+'\t')
    # f.write(str(sim.t)+'\n')

    ixprev = None
    iyprev = None
    izprev = None
    jxprev = None
    jyprev = None
    jzprev = None

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

        ixprev = ps[1].params['tt_ix']
        iyprev = ps[1].params['tt_iy']
        izprev = ps[1].params['tt_iz']
        jxprev = ps[1].params['tt_jx']
        jyprev = ps[1].params['tt_jy']
        jzprev = ps[1].params['tt_jz']
        rxs.append(rx)
        ixs.append(ps[1].params['tt_ix'])

    if to_plot:
        plt.plot(times, np.unwrap(angs), 'ko')
        # plt.ylim(np.pi - 0.2, np.pi + 0.2)
        plt.title('%.10f' % dt)
        plt.xlabel('Time')
        plt.ylabel('theta')
        plt.savefig(filename + '.png', dpi=300)
        plt.clf()

    f.close()
    print('Running for %.10f took %.7f s' % (dt, time.time() - start))

if __name__ == '__main__':
    dtmax = step
    dts = dtmax / 2**np.arange(10, -1, -1)
    # dts = [step]
    outfs = []
    for dt in dts:
        outf = 'test_torque_out_%.10fdt.txt' % dt
        run(dt)
        outfs.append(outf)

    nv = 3

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
    means = np.zeros(len(outfs)-1)
    stdevs = np.zeros(len(outfs)-1)

    for i in range(1,len(outfs)):
        f = open(outfs[i], 'r')
        allLines = f.readlines()

        data = np.zeros((n, nv))

        for j in range(n):
            datum = allLines[j].split()
            for k in range(nv):
                data[j,k] = float(datum[k])

        ang = np.unwrap(data[ :, angle])[1: ]
        rms[i-1] = np.sqrt(np.sum((ang-true_vals)**2))
        means[i-1] = np.mean(ang)
        stdevs[i-1] = np.std(ang)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        sharex=True)
    ax1.loglog(dts[1:], rms, 'go')
    ax1.set_yscale('log')
    ax1.plot(dts[1: ], rms[1] * (dts[1: ] / dts[1]))

    ax2.plot(dts[1: ], means, 'b')
    ax2.plot(dts[1: ], means + stdevs, 'g--', lw=0.5)
    ax2.plot(dts[1: ], means - stdevs, 'g--', lw=0.5)
    ax2.set_ylabel('Mean / std')

    # ax2.plot(dts[1: ], np.abs(means - means[0]), 'go')
    # ax2.set_ylim(bottom=0)
    # ax2.set_ylabel('Mean - Mean0')

    ax1.set_ylabel('RMS (radians)')
    ax2.set_xlabel('dt (years)')
    ax2.set_xscale('log')
    ax1.axvline(step, c='k')
    ax2.axvline(step, c='k')

    plt.tight_layout()
    plt.savefig('torque', dpi=300)
