import os
import rebound
import reboundx
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
# import multiprocessing as mp

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=11)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

# make simulation object with given parameters
# theta = obliquity, phi = azimuthal angle, 
# phi = 0 corresponds to initial condition where planet's k axis is tilting directly away from star
def create_sim(sim_params,dt_frac=0.025):
    a,Q_tide,R_p,theta,omega_to_n,M_p,k2,moment2,moment3,s_k_angle,a_out,i_out,M_out = sim_params

    sim = rebound.Simulation()
    sim.integrator = 'whfast'
    sim.units = ('AU', 'yr', 'MSun')

    sim.add(m=1.)
    sim.add(m=M_p, a=a)
    sim.add(m=M_out, a=a_out, inc=i_out)

    rebx = reboundx.Extras(sim)
    triax = rebx.load_operator('triaxial_torque')
    rebx.add_operator(triax)

    # add spin to smaller body
    ps = sim.particles

    ps[1].params['tt_ix'] = np.cos(theta) # + ((np.sin(phi)**2) * (1-np.cos(theta)))
    ps[1].params['tt_iy'] = 0. # -np.sin(phi)*np.cos(phi)*(1-np.cos(theta))
    ps[1].params['tt_iz'] = -np.sin(theta)# -np.cos(phi)*np.sin(theta)
    ps[1].params['tt_jx'] = 0. #-np.sin(phi)*np.cos(phi)*(1-np.cos(theta))
    ps[1].params['tt_jy'] = 1. #np.cos(theta) + ((np.cos(phi)**2) * (1-np.cos(theta)))
    ps[1].params['tt_jz'] = 0. #-np.sin(phi)*np.sin(theta)
    ps[1].params['tt_kx'] = np.sin(theta) # *np.cos(phi)
    ps[1].params['tt_ky'] = 0. # np.sin(theta)*np.sin(phi)
    ps[1].params['tt_kz'] = np.cos(theta)

    k = 0.331
    Ii = k*M_p*R_p*R_p
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

    if omega == 0:
        sim.dt = dt_frac*ps[1].P
    else:
        sim.dt = dt_frac*np.minimum(ps[1].P, 2*np.pi/omega)

    return sim

def calc_orbit_normal(r_xyz=None,v_xyz=None):
    l = np.cross(v_xyz,r_xyz)
    l_hat = l/np.sqrt(np.dot(l,l))
    return l_hat

# returns (obliquity, phi) of body at index 1 in radians
def get_theta(s_ijk,i_xyz,j_xyz,k_xyz,r,v):
    # calculate theta
    s_x = s_ijk[0]*i_xyz[0] + s_ijk[1]*j_xyz[0] + s_ijk[2]*k_xyz[0]
    s_y = s_ijk[0]*i_xyz[1] + s_ijk[1]*j_xyz[1] + s_ijk[2]*k_xyz[1]
    s_z = s_ijk[0]*i_xyz[2] + s_ijk[1]*j_xyz[2] + s_ijk[2]*k_xyz[2]
    s_xyz = np.stack((s_x,s_y,s_z),axis=0)
    s_xyz /= np.sqrt(np.dot(s_xyz,s_xyz))

    l_hat = calc_orbit_normal(r_xyz=r,v_xyz=v) # orbit normal of triaxial planet
    theta = np.arccos(np.dot(l_hat,s_xyz))
    
    return theta

def get_fig_axs(nw,nh,sharex=False,sharey=False,share_titles=False,share_xlab=False,share_ylab=False,titles=None,xlabels=None,ylabels=None):
    fig, axs = plt.subplots(nh, nw,figsize=(6*nw, 4*nh), sharex=sharex,sharey=sharey)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=.98, top=0.92, wspace=0.2, hspace=0.05)
    for i in range(nw):
        for j in range(nh):
            if (not (titles is None)) and (not (share_titles and j != 0)):
                axs[j,i].set_title(titles[j,i])
            if (not (xlabels is None)) and (not (share_xlab and j != nh-1)):
                axs[j,i].set_xlabel(xlabels[j,i])
            if (not (ylabels is None)) and (not (share_ylab and i != 0)):
                axs[j,i].set_ylabel(ylabels[j,i])
    # ylabels = np.array([[r"$\Omega/n$",r"$\theta$ ($^{\circ}$)",r"$\phi$ ($^{\circ}$)",r"$\psi$ ($^{\circ}$)"],[r"$\beta$ ($^{\circ}$)",r"$\theta_{kl}$ ($^{\circ}$)",r"$tan^{-1}(s_y/s_x)$ ($^{\circ}$)",r"$\theta '$ ($^{\circ}$)"]]) # ,r"Inclination"]

    return fig,axs

def calc_om_dot(ts,omegas):
    min_delta_t = n/4
    n = len(omegas)
    sorted_inds = np.argsort(omegas)

    # try mins
    indi = sorted_inds[0]
    i = 1
    while np.abs(sorted_inds[i] - indi) < min_delta_t:
        if i == n-1:
            break
        i += 1
    indf = sorted_inds[i]
    dist = np.abs(indf - indi)

    # if not min_delta_t apart, try maxes:
    if dist < min_delta_t:
        indii = sorted_inds[-1]
        i = -2
        while np.abs(sorted_inds[i] - indii) < min_delta_t:
            if i == 0:
                break
            i -= 1
        indff = sorted_inds[i]
        if np.abs(indff - indii) > dist:
            indi = indii
            indf == indff

    if indf < indi:
        temp = indi
        indi = indf
        indf = temp

    delta_t = ts[indf] - ts[indi]
    delta_omega = omegas[indf] - omegas[indi]
    return delta_omega/delta_t

def integrate_sim(sim_params,tf,theta_bool=False):

    # make sim
    sim = create_sim(sim_params)
    ps = sim.particles
    year = ps[1].P
    n = ps[1].n
    omegas = np.zeros(tf)
    ts = np.arange(int(tf))
    for i in range(tf):
        sim.integrate(i*year)
        omegas[i] = ps[1].params['tt_omega']

    return calc_om_dot(ts,omegas)

    # if theta_bool:
    #     s_ijk = np.array([ps[1].params['tt_si'],ps[1].params['tt_sj'],ps[1].params['tt_sk']])
    #     i_xyz = np.array([ps[1].params['tt_ix'],ps[1].params['tt_iy'],ps[1].params['tt_iz']])
    #     j_xyz = np.array([ps[1].params['tt_jx'],ps[1].params['tt_jy'],ps[1].params['tt_jz']])
    #     k_xyz = np.array([ps[1].params['tt_kx'],ps[1].params['tt_ky'],ps[1].params['tt_kz']])
    #     r_xyz = np.array([ps[0].x - ps[1].x,ps[0].y - ps[1].y,ps[0].z - ps[1].z])
    #     v_xyz = np.array([ps[1].vx,ps[1].vy,ps[1].vz])
        
    #     theta0 = get_theta(s_ijk,i_xyz,j_xyz,k_xyz,r_xyz,v_xyz)
    # else:
    #     omega0 = ps[1].params['tt_omega']

    # avrg_interval = tf / 2

    # na = 20
    # omegas = np.zeros(na)
    # step = year * avrg_interval / na
    # for i in range(na):
    #     sim.integrate(i*step)
    #     omegas[i] = ps[1].params['tt_omega']
    # omega0 = np.mean(omegas)

    # sim.integrate((tf-avrg_interval)*year)
    
    # for i in range(na):
    #     sim.integrate((tf-avrg_interval)*year + i*step)
    #     omegas[i] = ps[1].params['tt_omega']

    # if theta_bool:
    #     s_ijk = np.array([ps[1].params['tt_si'],ps[1].params['tt_sj'],ps[1].params['tt_sk']])
    #     i_xyz = np.array([ps[1].params['tt_ix'],ps[1].params['tt_iy'],ps[1].params['tt_iz']])
    #     j_xyz = np.array([ps[1].params['tt_jx'],ps[1].params['tt_jy'],ps[1].params['tt_jz']])
    #     k_xyz = np.array([ps[1].params['tt_kx'],ps[1].params['tt_ky'],ps[1].params['tt_kz']])
    #     r_xyz = np.array([ps[0].x - ps[1].x,ps[0].y - ps[1].y,ps[0].z - ps[1].z])
    #     v_xyz = np.array([ps[1].vx,ps[1].vy,ps[1].vz])
    #     return (get_theta(s_ijk,i_xyz,j_xyz,k_xyz,r_xyz,v_xyz) - theta0) / sim.t
    
    # else:
    #     return ((np.mean(omegas) - omega0)/n) / (tf-avrg_interval)
    
# 2d plot (omega dot vs omega for a few theta values)
def plot_2d():
    omega_lo = 0.5
    omega_hi = 3.0
    n_omegas = 200
    omegas = np.linspace(omega_lo,omega_hi,n_omegas)

    thetas = np.radians(np.array([90.])) # np.radians(np.array([5.,50.,120.,130.]))
    n_thetas = np.shape(thetas)[0]

    tf = 5.

    # some constants
    Re = 4.263e-5 # radius of Earth in AU
    Me = 3.003e-6 # mass of Earth in solar masses
    Mj = 9.542e-4 # mass of Jupiter in solar masses

    ### SIMULATION PARAMETERS ###
    # fixed parameters
    a = .4 # semi-major axis of inner planet
    Q_tide = 100.
    R_p = 2.*Re # radius of inner planet
    M_p = 4.*Me # mass of inner planet
    k2 = 1.5 # 1.5 for uniformly distributed mass
    s_k_angle = np.radians(0.) # angle between s and k
    a_out = 5. # a of outer planet
    i_out = np.radians(20.) # inclination of outer planet
    M_out = Mj # mass of outer planet
    # What to do about these?
    moment2 = 1.e-5 # 1e-1 # (Ij - Ii) / Ii, < moment3
    moment3 = 1.e-3 # 2e-1 # (Ik - Ii) / Ii, > moment2
    
    omega_dots = np.zeros((2,n_thetas,n_omegas)) # first dimension corresponds to triax (0) or oblate (1)

    for i in range(n_thetas):
        theta = thetas[i]
        for j in range(n_omegas):
            omega_to_n = omegas[j]

            ### RUN SIMULATION ###
            sim_params = a,Q_tide,R_p,theta,omega_to_n,M_p,k2,moment2,moment3,s_k_angle,a_out,i_out,M_out
            omega_dots[0,i,j] = integrate_sim(sim_params,tf)

            ### Re-RUN SIMULATION with same parameters, except just j2 ###
            moment2 = 0.
            sim_params = a,Q_tide,R_p,theta,omega_to_n,M_p,k2,moment2,moment3,s_k_angle,a_out,i_out,M_out
            omega_dots[1,i,j] = integrate_sim(sim_params,tf)

    # plot results
    fig, axs = plt.subplots(1, 2,figsize=(8, 4), sharex=True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=.98, top=0.92, wspace=0.2, hspace=0.05)
    axs[0].set_xlabel(r"$\Omega/n$")
    axs[1].set_xlabel(r"$\Omega/n$")
    axs[0].set_ylabel(r"$d\Omega/dt$ (rad/year$^2$)")
    
    axs[0].set_title("Triaxial")
    axs[1].set_title("Oblate")

    cs = ["r","g","b","orange"]
    for i in range(n_thetas):
        axs[0].plot(omegas, omega_dots[0,i],lw=0.5,c=cs[i])
    
    axs[0].legend([r"$\theta =$ "+str(int(np.degrees(thetas[i]))) for i in range(n_thetas)])
    
    for i in range(n_thetas):
        axs[1].plot(omegas, omega_dots[0,i],lw=0.5,c=cs[i])
    
    axs[1].legend([r"$\theta =$ "+str(int(np.degrees(thetas[i]))) for i in range(n_thetas)])

    plt.savefig("omega_dot.png", dpi=300)
    plt.clf()
    plt.close(fig)

# 3d plot (colored omega dot on omega and theta axes)
def plot_3d(theta_bool):
    omega_lo = 1.95
    omega_hi = 2.05
    n_omegas = 30
    omegas = np.linspace(omega_lo,omega_hi,n_omegas)

    theta_lo = 0.
    theta_hi = 180.
    n_thetas = 40
    thetas = np.linspace(theta_lo,theta_hi,n_thetas)

    omega_grid, theta_grid = np.meshgrid(omegas,thetas)

    tf = 200.

    # some constants
    Re = 4.263e-5 # radius of Earth in AU
    Me = 3.003e-6 # mass of Earth in solar masses
    Mj = 9.542e-4 # mass of Jupiter in solar masses

    ### SIMULATION PARAMETERS ###
    # fixed parameters
    a = .4 # semi-major axis of inner planet
    Q_tide = 100.
    R_p = 2.*Re # radius of inner planet
    M_p = 4.*Me # mass of inner planet
    k2 = 1.5 # 1.5 for uniformly distributed mass
    s_k_angle = np.radians(0.) # angle between s and k
    a_out = 5. # a of outer planet
    i_out = np.radians(20.) # inclination of outer planet
    M_out = Mj # mass of outer planet
    # What to do about these?
    moment2 = 1.e-5 # (Ij - Ii) / Ii, < moment3
    moment3 = 1.e-3 # (Ik - Ii) / Ii, > moment2
    
    omega_dots = np.zeros((2,n_thetas,n_omegas)) # first dimension corresponds to triax (0) or oblate (1)

    for i in range(n_omegas):
        for j in range(n_thetas):
            omega_to_n = omega_grid[j,i]
            theta = np.radians(theta_grid[j,i])

            ### RUN SIMULATION ###
            sim_params = a,Q_tide,R_p,theta,omega_to_n,M_p,k2,moment2,moment3,s_k_angle,a_out,i_out,M_out
            omega_dots[0,j,i] = integrate_sim(sim_params,tf,theta_bool=theta_bool)

            ### Re-RUN SIMULATION with same parameters, except just j2 ###
            moment2 = 0.
            sim_params = a,Q_tide,R_p,theta,omega_to_n,M_p,k2,moment2,moment3,s_k_angle,a_out,i_out,M_out
            omega_dots[1,j,i] = integrate_sim(sim_params,tf,theta_bool=theta_bool)

    # omega_dots = np.minimum(omega_dots,np.zeros_like(omega_dots))

    # plot results
    fig, axs = plt.subplots(1, 2,figsize=(8, 4), sharex=True,sharey=True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=.95, top=0.92, wspace=0.1, hspace=0.05)
    axs[0].set_xlabel(r"$\Omega/n$")
    axs[1].set_xlabel(r"$\Omega/n$")
    axs[0].set_ylabel(r"$\theta$ ($^{\circ}$)")
    
    axs[0].set_title("Triaxial")
    axs[1].set_title("Oblate")

    val = np.maximum(np.max(omega_dots[0]),-np.min(omega_dots[0])) # (np.max(omega_dots[0]) - np.min(omega_dots[0]))/2.
    norm = mpl.colors.Normalize(vmin=-val, vmax=val)
    axs[0].pcolormesh(omega_grid,theta_grid,omega_dots[0],norm=norm,cmap='coolwarm',shading='auto')
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap='coolwarm'), ax=axs[0])

    val = np.maximum(np.max(omega_dots[1]),-np.min(omega_dots[1]))
    # val = (np.max(omega_dots[1]) - np.min(omega_dots[1]))/2.
    norm = mpl.colors.Normalize(vmin=-val, vmax=val)
    axs[1].pcolormesh(omega_grid,theta_grid,omega_dots[1],norm=norm,cmap='coolwarm',shading='auto')
    if theta_bool:
        lab = r"$d\theta/dt$ (rad/year$^2$)"
    else:
        lab = r"$d\Omega/dt$ ($n/P^2$)"
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap='coolwarm'), ax=axs[1],label=lab)

    plt.savefig("omega_dot.png", dpi=300)
    plt.clf()
    plt.close(fig)

def plot_spin_profile():

    start = time.time()
    
    omega_to_n = 2.01
    theta = np.radians(70.)

    tf = 10000.
    nout = 1000
    step = tf/nout

    # some constants
    Re = 4.263e-5 # radius of Earth in AU
    Me = 3.003e-6 # mass of Earth in solar masses
    Mj = 9.542e-4 # mass of Jupiter in solar masses

    ### SIMULATION PARAMETERS ###
    # fixed parameters
    a = .4 # semi-major axis of inner planet
    Q_tide = 100.
    R_p = 2.*Re # radius of inner planet
    M_p = 4.*Me # mass of inner planet
    k2 = 1.5 # 1.5 for uniformly distributed mass
    s_k_angle = np.radians(0.) # angle between s and k
    a_out = 5. # a of outer planet
    i_out = np.radians(20.) # inclination of outer planet
    M_out = Mj # mass of outer planet
    # What to do about these?
    moment2 = 1.e-5 # 1e-1 # (Ij - Ii) / Ii, < moment3
    moment3 = 1.e-3 # 2e-1 # (Ik - Ii) / Ii, > moment2
    
    ts = np.zeros((2,nout+1)) # first dimension corresponds to triax (0) or oblate (1)
    omegas = np.zeros((2,nout+1))
    thetas = np.zeros((2,nout+1))
    betas = np.zeros((2,nout+1))

    for i in range(2):
        if i == 1:
            moment2 = 0.
        sim_params = a,Q_tide,R_p,theta,omega_to_n,M_p,k2,moment2,moment3,s_k_angle,a_out,i_out,M_out
        sim = create_sim(sim_params)
        ps = sim.particles
        year = ps[1].P
        n = ps[1].n
        for j in range(nout+1):
            sim.integrate(step*j*year)

            s_ijk = np.array([ps[1].params['tt_si'],ps[1].params['tt_sj'],ps[1].params['tt_sk']])
            i_xyz = np.array([ps[1].params['tt_ix'],ps[1].params['tt_iy'],ps[1].params['tt_iz']])
            j_xyz = np.array([ps[1].params['tt_jx'],ps[1].params['tt_jy'],ps[1].params['tt_jz']])
            k_xyz = np.array([ps[1].params['tt_kx'],ps[1].params['tt_ky'],ps[1].params['tt_kz']])
            r_xyz = np.array([ps[0].x - ps[1].x,ps[0].y - ps[1].y,ps[0].z - ps[1].z])
            v_xyz = np.array([ps[1].vx,ps[1].vy,ps[1].vz])

            ts[i,j] = sim.t/year
            omegas[i,j] = ps[1].params['tt_omega']/n
            thetas[i,j] = get_theta(s_ijk,i_xyz,j_xyz,k_xyz,r_xyz,v_xyz)
            betas[i,j] = np.arccos(s_ijk[2])

    print(time.time()-start)

    # plot results
    titles = np.array([["triaxial","oblate"]])
    xlabs = np.array([["Time (P)","Time (P)"],["Time (P)","Time (P)"],["Time (P)","Time (P)"]])
    ylabs = np.array([[r"$\Omega/n$",r"$\Omega/n$"],[r"$\theta$ ($^{\circ}$)",r"$\theta$ ($^{\circ}$)"],[r"$\beta$ ($^{\circ}$)",r"$\beta$ ($^{\circ}$)"]])
    fig, axs = get_fig_axs(2,3,sharex=True,share_titles=True,share_xlab=True,share_ylab=True,titles=titles,xlabels=xlabs,ylabels=ylabs)
    
    for i in range(2):
        axs[0,i].scatter(ts[i],omegas[i],color="black",s=0.5)
        axs[1,i].scatter(ts[i],np.degrees(thetas[i]),color="black",s=0.5)
        axs[2,i].scatter(ts[i],np.degrees(betas[i]),color="black",s=0.5)
    
    plt.savefig("omega_profile.png", dpi=300)
    plt.clf()
    plt.close(fig)
    
if __name__=="__main__":
    start = time.time()
    # plot_2d()
    # plot_3d(False)
    plot_spin_profile()

    print(time.time()-start)
