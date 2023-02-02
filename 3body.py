import os
import rebound
import reboundx
import numpy as np
import time
import multiprocessing as mp

# make simulation object with given parameters
# theta = obliquity, phi = azimuthal angle, 
# phi = 0 corresponds to initial condition where planet's k axis is tilting directly away from star
def create_sim(sim_params,dt_frac=0.01):
    a,Q_tide,R_p,theta,phi,omega_to_n,M_p,k2,moment2,moment3,s_k_angle,a_out,i_out,M_out = sim_params

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

# returns orbit normal unit vector for orbit of body at index index
def calc_orbit_normal(ps, index):
    r_xyz = np.array([ps[index].x - ps[0].x,ps[index].y - ps[0].y,ps[index].z - ps[0].z]) # here, r is vector from star to planet (opposite from operator code)
    v_xyz = np.array([ps[index].vx,ps[index].vy,ps[index].vz])
    r = np.sqrt(np.dot(r_xyz,r_xyz))
    v = np.sqrt(np.dot(v_xyz,v_xyz))
    r_hat = r_xyz / r
    v_hat = v_xyz / v
    return np.cross(r_hat, v_hat)

# returns (obliquity, phi) of body at index index in degrees
def get_theta_phi_deg(ps, index):
    s_ijk = np.array([ps[index].params['tt_si'],ps[index].params['tt_sj'],ps[index].params['tt_sk']])
    ijk_xyz = np.array([[ps[index].params['tt_ix'],ps[index].params['tt_iy'],ps[index].params['tt_iz']],[ps[index].params['tt_jx'],ps[index].params['tt_jy'],ps[index].params['tt_jz']],[ps[index].params['tt_kx'],ps[index].params['tt_ky'],ps[index].params['tt_kz']]])
    s_xyz = s_ijk[0]*ijk_xyz[0] + s_ijk[1]*ijk_xyz[1] + s_ijk[2]*ijk_xyz[2]
    theta = np.degrees(np.arccos(np.dot(calc_orbit_normal(ps,index),s_xyz)))
    # following definition of phi assumes inclination is 0
    phi = np.degrees(np.arctan2(s_xyz[1],s_xyz[0])) 
    return (theta, phi)

# returns spin rate of body at index index in units of mean motion
def get_omega_to_n(ps, index):
    return ps[index].params['tt_omega']/ps[index].n 

def run_sim(trial_num, tf=10000., n_out=200):

    start = time.time()

    # some constants
    Re = 4.263e-5 # radius of Earth in AU
    Me = 3.003e-6 # mass of Earth in solar masses
    Mj = 9.542e-4 # mass of Jupiter in solar masses

    ### SIMULATION PARAMETERS ###
    # fixed parameters
    a = .4 # semi-major axis of inner planet
    Q_tide = 10.
    R_p = 2.*Re # radius of inner planet
    M_p = 4.*Me # mass of inner planet
    k2 = 1.5 # 1.5 for uniformly distributed mass
    s_k_angle = np.radians(0.) # angle between s and k
    a_out = 5. # a of outer planet
    i_out = np.radians(20.) # inclination of outer planet
    M_out = Mj # mass of outer planet
    # What to do about these?
    moment2 = 0. # 1e-1 # (Ij - Ii) / Ii, < moment3
    moment3 = 0. # 2e-1 # (Ik - Ii) / Ii, > moment2

    # variable params
    theta = np.pi*np.random.default_rng().uniform()
    phi = 2*np.pi*np.random.default_rng().uniform()
    omega_to_n = 2.*np.random.default_rng().uniform() # 2 because otherwise obliquity is excited # (1+(np.pi/2/np.arctan(1/Q_tide)))
    
    ### RUN SIMULATION ###
    print(f"Trial {trial_num} initiated with obliquity {np.degrees(theta)} degrees and spin {omega_to_n} mean-motions.")

    # make sim
    sim_params = a,Q_tide,R_p,theta,phi,omega_to_n,M_p,k2,moment2,moment3,s_k_angle,a_out,i_out,M_out
    sim = create_sim(sim_params)
    ps = sim.particles

    # make output directory and file
    dir_path = "./3bd_20i_sphere_10Q_0.01dt"
    if trial_num == 0:
        if os.path.exists(dir_path):
            print("Error: Directory already exists") 
        os.mkdir(dir_path)
    else:
        while (not os.path.exists(dir_path)):
            time.sleep(2)
            
    file_path = os.path.join(dir_path,"trial_"+str(trial_num)+".txt")
    f = open(file_path, "w")

    # want to plot omega, theta, and phi, so write those to file
    # also write time
    step = tf / (n_out-1)
    year = ps[1].P
    for i in range(n_out):
        sim.integrate(i*step*year)
        t = sim.t / year
        omega = get_omega_to_n(ps,1)
        theta, phi = get_theta_phi_deg(ps,1)

        f.write(str(t)+'\t')
        f.write(str(omega)+'\t')
        f.write(str(theta)+'\t')
        f.write(str(phi)+'\n')
    
    f.close()
    int_time = time.time() - start
    hrs = int_time // 3600
    mins = (int_time % 3600) // 60
    secs = int((int_time % 3600) % 60)
    print(f"Trial {trial_num} completed in {hrs} hours {mins} minutes {secs} seconds.")
    return 

# main function
if __name__ == '__main__':
    n_trials = 30
    with mp.Pool(processes=30) as pool:
        int_times = pool.map(run_sim, range(n_trials))
