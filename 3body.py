import os
import rebound
import reboundx
import numpy as np
import time
import multiprocessing as mp

def get_rand_ijk():
    proto_i = np.array([np.random.default_rng().uniform(),np.random.default_rng().uniform(),np.random.default_rng().uniform()])
    i = proto_i/np.sqrt(np.dot(proto_i,proto_i))

    cont_bool = True
    while (cont_bool):
        proto_j = np.array([np.random.default_rng().uniform(),np.random.default_rng().uniform(),np.random.default_rng().uniform()])
        pre_j = proto_j - (np.dot(proto_j,i)*i)
        if (not np.all(pre_j == 0)):
            cont_bool = False

    j = pre_j / np.sqrt(np.dot(pre_j,pre_j))
    proto_k = np.cross(i,j)
    k = proto_k / np.sqrt(np.dot(proto_k,proto_k))

    return (i,j,k)

# make simulation object with given parameters
# theta = obliquity, phi = azimuthal angle, 
# phi = 0 corresponds to initial condition where planet's k axis is tilting directly away from star
def create_sim(sim_params,dt_frac=0.025,rand_ijk=True):
    i,j,k,a,Q_tide,R_p,theta,omega_to_n,M_p,k2,moment2,moment3,s_k_angle,a_out,i_out,M_out = sim_params

    sim = rebound.Simulation()
    sim.integrator = 'whfast'
    sim.units = ('AU', 'yr', 'MSun')

    sim.add(m=1.)
    sim.add(m=M_p, a=a)
    if M_out > 0:
        sim.add(m=M_out, a=a_out, inc=i_out)

    rebx = reboundx.Extras(sim)
    triax = rebx.load_operator('triaxial_torque')
    rebx.add_operator(triax)

    # add spin to smaller body
    ps = sim.particles

    if not rand_ijk:
        ps[1].params['tt_ix'] = np.cos(theta) # + ((np.sin(phi)**2) * (1-np.cos(theta)))
        ps[1].params['tt_iy'] = 0. # -np.sin(phi)*np.cos(phi)*(1-np.cos(theta))
        ps[1].params['tt_iz'] = -np.sin(theta)# -np.cos(phi)*np.sin(theta)
        ps[1].params['tt_jx'] = 0. #-np.sin(phi)*np.cos(phi)*(1-np.cos(theta))
        ps[1].params['tt_jy'] = 1. #np.cos(theta) + ((np.cos(phi)**2) * (1-np.cos(theta)))
        ps[1].params['tt_jz'] = 0. #-np.sin(phi)*np.sin(theta)
        ps[1].params['tt_kx'] = np.sin(theta) # *np.cos(phi)
        ps[1].params['tt_ky'] = 0. # np.sin(theta)*np.sin(phi)
        ps[1].params['tt_kz'] = np.cos(theta)

    else:
        ps[1].params['tt_ix'] = i[0]
        ps[1].params['tt_iy'] = i[1]
        ps[1].params['tt_iz'] = i[2]
        ps[1].params['tt_jx'] = j[0]
        ps[1].params['tt_jy'] = j[1]
        ps[1].params['tt_jz'] = j[2]
        ps[1].params['tt_kx'] = k[0]
        ps[1].params['tt_ky'] = k[1]
        ps[1].params['tt_kz'] = k[2]

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

def integrate_sim(dir_path,sim_params,trial_num_dec,tf,step,rand_ijk=True):
    start = time.time()
    print(f"Trial {trial_num_dec} initiated",flush=True)

    # make sim
    sim = create_sim(sim_params,rand_ijk=rand_ijk)
    ps = sim.particles

    # want to plot omega, theta, phi, psi, eccentricity, and inclination so save those to array
    # also write time
    year = ps[1].P
    n_out = int((tf // step) + 1)
    if len(ps) == 3:
        val_names = ["ix","iy","iz","jx","jy","jz","kx","ky","kz","si","sj","sk","omega","rx","ry","rz","vx","vy","vz","rpx","rpy","rpz","t"] # r is vector from planet to star !
    else:
        val_names = ["ix","iy","iz","jx","jy","jz","kx","ky","kz","si","sj","sk","omega","rx","ry","rz","vx","vy","vz","t"] # r is vector from planet to star !
    inds = {val_names[i]:i for i in range(len(val_names))}
    nv = len(val_names)
    out_data = np.zeros((nv,n_out), dtype=np.float32)

    for i in range(n_out):
        sim.integrate(i*step*year)

        out_data[inds['ix'],i] = ps[1].params['tt_ix']
        out_data[inds['iy'],i] = ps[1].params['tt_iy']
        out_data[inds['iz'],i] = ps[1].params['tt_iz']
        out_data[inds['jx'],i] = ps[1].params['tt_jx']
        out_data[inds['jy'],i] = ps[1].params['tt_jy']
        out_data[inds['jz'],i] = ps[1].params['tt_jz']
        out_data[inds['kx'],i] = ps[1].params['tt_kx']
        out_data[inds['ky'],i] = ps[1].params['tt_ky']
        out_data[inds['kz'],i] = ps[1].params['tt_kz']
        out_data[inds['si'],i] = ps[1].params['tt_si']
        out_data[inds['sj'],i] = ps[1].params['tt_sj']
        out_data[inds['sk'],i] = ps[1].params['tt_sk']
        out_data[inds['omega'],i] = ps[1].params['tt_omega']/ps[1].n
        out_data[inds['rx'],i] = ps[0].x - ps[1].x
        out_data[inds['ry'],i] = ps[0].y - ps[1].y
        out_data[inds['rz'],i] = ps[0].z - ps[1].z
        out_data[inds['vx'],i] = ps[1].vx
        out_data[inds['vy'],i] = ps[1].vy
        out_data[inds['vz'],i] = ps[1].vz
        if len(ps) == 3:
            out_data[inds['rpx'],i] = ps[0].x - ps[2].x
            out_data[inds['rpy'],i] = ps[0].y - ps[2].y
            out_data[inds['rpz'],i] = ps[0].z - ps[2].z
        out_data[inds['t'],i] = sim.t / year

    file_path = os.path.join(dir_path,"trial_"+str(trial_num_dec)+".npy")
    
    with open(file_path, 'wb') as f:
        np.save(f, out_data)
    
    int_time = time.time() - start
    hrs = int_time // 3600
    mins = (int_time % 3600) // 60
    secs = int((int_time % 3600) % 60)
    print(f"Trial {trial_num_dec} completed in {hrs} hours {mins} minutes {secs} seconds.", flush=True) 

def run_sim(trial_num, tf=1.e7, out_step=1.e3, perturber=True, rand_ijk=False, n_trials=50, version=2.4):

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
    if perturber:
        M_out = Mj # mass of outer planet
    else:
        M_out = 0
    # What to do about these?
    moment2 = 1.e-5 # 1e-1 # (Ij - Ii) / Ii, < moment3
    moment3 = 1.e-3 # 2e-1 # (Ik - Ii) / Ii, > moment2

    # max_omega = 4. # 2 because otherwise obliquity is excited # (1+(np.pi/2/np.arctan(1/Q_tide)))
    # omega_to_n = max_omega*np.random.default_rng().uniform()
    omegas = [1.2,2.2]
    omega_to_n = omegas[trial_num % len(omegas)]

    # variable params
    if rand_ijk:
        theta = 0.
    else:
        thetas = np.linspace(0,180,int(n_trials / len(omegas)))
        theta = thetas[trial_num//2]

    # generate random i,j,k
    if rand_ijk:
        i,j,k = get_rand_ijk()
    else:
        i, j, k = np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0])

    # make output directory and file
    if perturber:
        dir_path = "./data/v"+str(version)+"_3body_data"
    else:
        dir_path = "./data/v"+str(version)+"_2body_data"
    if trial_num == 0:
        if os.path.exists(dir_path):
            print("Error: Directory already exists")
            exit()
        os.mkdir(dir_path)
    else:
        while (not os.path.exists(dir_path)):
            time.sleep(1)

    ### RUN SIMULATION ###
    sim_params = i,j,k,a,Q_tide,R_p,theta,omega_to_n,M_p,k2,moment2,moment3,s_k_angle,a_out,i_out,M_out
    integrate_sim(dir_path,sim_params,trial_num,tf,out_step,rand_ijk=rand_ijk)

    ### Re-RUN SIMULATION with same parameters, except just j2 ###
    trial_num_2 = trial_num + 0.1
    moment2 = 0.
    sim_params = i,j,k,a,Q_tide,R_p,theta,omega_to_n,M_p,k2,moment2,moment3,s_k_angle,a_out,i_out,M_out
    integrate_sim(dir_path,sim_params,trial_num_2,tf,out_step,rand_ijk=rand_ijk)

# main function
if __name__ == '__main__':
    # to change params, see keyword args in run_sim()
    n_trials = 50 # if I change this, make sure to also change in keyword args in run_sim()
    start = time.time()
    with mp.Pool(processes=n_trials) as pool:
        pool.map(run_sim, range(n_trials))
    
    tot_time = time.time() - start
    hrs = tot_time // 3600
    mins = (tot_time % 3600) // 60
    secs = int((tot_time % 3600) % 60)
    print(f"Total Runtime: {hrs} hours {mins} minutes {secs} seconds", flush=True)
