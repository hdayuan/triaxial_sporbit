import os
import rebound
import reboundx
import numpy as np
import time
import sys



# make simulation object with given parameters
# theta = obliquity, phi = azimuthal angle, 
# phi = 0 corresponds to initial condition where planet's k axis is tilting directly away from star
def create_sim(sim_params,dt_frac=0.05):
    a,Q_tide,R_p,theta,phi,omega_to_n,M_p,k2,moment2,moment3,s_k_angle = sim_params

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
    ps[1].params['tt_ix'] = np.cos(theta) # + ((np.sin(phi)**2) * (1-np.cos(theta)))
    ps[1].params['tt_iy'] = 0. # -np.sin(phi)*np.cos(phi)*(1-np.cos(theta))
    ps[1].params['tt_iz'] = -np.sin(theta)# -np.cos(phi)*np.sin(theta)
    ps[1].params['tt_jx'] = 0. #-np.sin(phi)*np.cos(phi)*(1-np.cos(theta))
    ps[1].params['tt_jy'] = 1. #np.cos(theta) + ((np.cos(phi)**2) * (1-np.cos(theta)))
    ps[1].params['tt_jz'] = 0. #-np.sin(phi)*np.sin(theta)
    ps[1].params['tt_kx'] = np.sin(theta) # *np.cos(phi)
    ps[1].params['tt_ky'] = 0. # np.sin(theta)*np.sin(phi)
    ps[1].params['tt_kz'] = np.cos(theta)

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

    if omega == 0:
        sim.dt = dt_frac*ps[1].P
    else:
        sim.dt = dt_frac*np.minimum(ps[1].P, 2*np.pi/omega)

    return sim

def run_sim(trial_num, tf=1000., n_out=201):

    start = time.time()

    step = tf // (n_out-1)

    # parameters
    a = .1
    Q_tide = 1.
    R_p = 1.e-4 # ~ 2 earth radii
    M_p = 1.e-4 # in units of primary body's mass (~ 2 earth masses)
    k2 = 1.5 # 1.5 for uniformly distributed mass
    s_k_angle = np.radians(0.) # angle between s and k
    # What to do about these?
    moment2 = 1e-1 # (Ij - Ii) / Ii, < moment3
    moment3 = 2e-1 # (Ik - Ii) / Ii, > moment2
    # vary these
    theta = np.pi*np.random.default_rng().uniform()
    print(f"Obliquity: {theta}")
    phi = 2*np.pi*np.random.default_rng().uniform()
    omega_to_n = np.random.default_rng().uniform()*2 # 2 because otherwise obliquity is excited # (1+(np.pi/2/np.arctan(1/Q_tide)))
    print(f"spin rate / n: {omega}")

    # make sim
    sim_params = a,Q_tide,R_p,theta,phi,omega_to_n,M_p,k2,moment2,moment3,s_k_angle
    sim = create_sim(sim_params)
    ps = sim.particles

    # make output directory and file
    dir_path = "./2body_equi_data"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_path = os.path.join(dir_path,"trial_"+str(trial_num)+".txt")
    f = open(file_path, "w")

    # want to plot omega, theta, and phi, so write those to file
    # also write time
    year = ps[1].P
    mm = ps[1].n
    for i in range(n_out):
        sim.integrate(i*step*year)
        t = sim.t / year
        omega = ps[1].params['tt_omega']/mm
        s_ijk = np.array([ps[1].params['tt_si'],ps[1].params['tt_sj'],ps[1].params['tt_sk']])
        ijk_xyz = np.array([[ps[1].params['tt_ix'],ps[1].params['tt_iy'],ps[1].params['tt_iz']],[ps[1].params['tt_jx'],ps[1].params['tt_jy'],ps[1].params['tt_jz']],[ps[1].params['tt_kx'],ps[1].params['tt_ky'],ps[1].params['tt_kz']]])
        s_xyz = s_ijk[0]*ijk_xyz[0] + s_ijk[1]*ijk_xyz[1] + s_ijk[2]*ijk_xyz[2]
        theta = np.degrees(np.arccos(s_xyz[2]))
        phi = np.degrees(np.arctan2(s_xyz[1],s_xyz[0]))

        f.write(str(t)+'\t')
        f.write(str(omega)+'\t')
        f.write(str(theta)+'\t')
        f.write(str(phi)+'\n')
    
    f.close()
    return time.time() - start

# main function
if __name__ == '__main__':
    n_trials = 10
    for i in range(n_trials):
        print("Integration Time: "+str(run_sim(i)))