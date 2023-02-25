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
    n = np.cross(r_hat, v_hat)
    n_hat = n/np.sqrt(np.dot(n,n))
    return n_hat

# returns (obliquity, phi) of body at index 1 in radians
def get_theta_phi(ps):
    # calculate theta
    s_ijk = np.array([ps[1].params['tt_si'],ps[1].params['tt_sj'],ps[1].params['tt_sk']])
    ijk_xyz = np.array([[ps[1].params['tt_ix'],ps[1].params['tt_iy'],ps[1].params['tt_iz']],[ps[1].params['tt_jx'],ps[1].params['tt_jy'],ps[1].params['tt_jz']],[ps[1].params['tt_kx'],ps[1].params['tt_ky'],ps[1].params['tt_kz']]])
    s_xyz = s_ijk[0]*ijk_xyz[0] + s_ijk[1]*ijk_xyz[1] + s_ijk[2]*ijk_xyz[2]
    s_xyz /= np.sqrt(np.dot(s_xyz,s_xyz))
    n_hat = calc_orbit_normal(ps,1) # orbit normal of triaxial planet
    theta = np.arccos(np.dot(n_hat,s_xyz))
    
    # calculate phi
    n_p_hat = calc_orbit_normal(ps,2) # orbit normal of perturbing planet
    y = np.cross(n_p_hat, n_hat) # unrelated to y basis unit vector
    y_hat = y/np.sqrt(np.dot(y,y))
    x = np.cross(y_hat, n_hat) # unrelated to x basis unit vector
    x_hat = x/np.sqrt(np.dot(x,x))
    phi = np.arctan2(np.dot(s_xyz,y_hat),np.dot(s_xyz,x_hat))

    # range from 0 to 360
    if phi < 0:
        phi = (2*np.pi) + phi
    
    return (theta, phi)

# returns spin rate of body at index 1 in units of mean motion
def get_omega_to_n(ps):
    return ps[1].params['tt_omega']/ps[1].n 

# returns angle in rad
def get_psi(ps):
    r_vec = np.array([ps[1].x - ps[0].x, ps[1].y - ps[0].y, ps[1].z - ps[0].z])
    r = np.sqrt(np.dot(r_vec,r_vec))
    r_hat = r_vec / r
    i_hat = np.array([ps[1].params['tt_ix'], ps[1].params['tt_iy'], ps[1].params['tt_iz']])
    # j_hat = np.array([ps[1].params['tt_jx'], ps[1].params['tt_jy'], ps[1].params['tt_jz']])
    n_hat = calc_orbit_normal(ps,1)
    # subtact component of i that is not in orbital plane
    i_pl = i_hat - (np.dot(i_hat,n_hat)*n_hat)
    i_pl_hat = i_pl / np.sqrt(np.dot(i_pl,i_pl))
    j_pl_hat = np.cross(n_hat, i_pl_hat)

    i_dot_r = np.dot(i_pl_hat,r_hat)
    j_dot_r = np.dot(j_pl_hat,r_hat)
    psi = np.arctan2(j_dot_r, i_dot_r)

    # range from 0 to 360
    if psi < 0:
        psi = (2*np.pi) + psi

    return psi

# returns angle in rad
def get_sk_angle(ps):
    return np.arccos(ps[1].params['tt_sk'])

def integrate_sim(dir_path,sim_params,trial_num_dec,tf,nv,inds,step):
    start = time.time()
    print(f"Trial {trial_num_dec} initiated",flush=True)

    # make sim
    sim = create_sim(sim_params)
    ps = sim.particles

    # want to plot omega, theta, phi, psi, eccentricity, and inclination so save those to array
    # also write time
    year = ps[1].P
    n_out = int((tf // step) + 1)
    nv = 6
    out_data = np.zeros((nv,n_out), dtype=np.float32)
    omega_ind,theta_ind,phi_ind,psi_ind,sk_ind,t_ind = inds

    for i in range(n_out):
        sim.integrate(i*step*year)
        t = sim.t / year
        omega = get_omega_to_n(ps)
        theta, phi = get_theta_phi(ps)
        psi = get_psi(ps)
        sk_angle = get_sk_angle(ps)

        out_data[omega_ind,i] = omega
        out_data[theta_ind,i] = theta
        out_data[phi_ind,i] = phi
        out_data[psi_ind,i] = psi
        out_data[sk_ind,i] = sk_angle
        out_data[t_ind,i] = t

    # convert all angles to degrees
    out_data[theta_ind] = np.degrees(out_data[theta_ind])
    out_data[phi_ind] = np.degrees(out_data[phi_ind])
    out_data[psi_ind] = np.degrees(out_data[psi_ind])
    out_data[sk_ind] = np.degrees(out_data[sk_ind])

    file_path = os.path.join(dir_path,"trial_"+str(trial_num_dec)+".npy")
    
    with open(file_path, 'wb') as f:
        np.save(f, out_data)
    
    int_time = time.time() - start
    hrs = int_time // 3600
    mins = (int_time % 3600) // 60
    secs = int((int_time % 3600) % 60)
    print(f"Trial {trial_num_dec} completed in {hrs} hours {mins} minutes {secs} seconds.", flush=True) 

def run_sim(trial_num, tf=2.e7, out_step=50.):

    # some constants
    Re = 4.263e-5 # radius of Earth in AU
    Me = 3.003e-6 # mass of Earth in solar masses
    Mj = 9.542e-4 # mass of Jupiter in solar masses

    ### SIMULATION PARAMETERS ###
    # fixed parameters
    a = .4 # semi-major axis of inner planet
    Q_tide = 300.
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

    # variable params
    theta = 0. # np.pi*np.random.default_rng().uniform()
    # max_omega = 4. # 2 because otherwise obliquity is excited # (1+(np.pi/2/np.arctan(1/Q_tide)))
    # omega_to_n = max_omega*np.random.default_rng().uniform()
    omegas = [1.2,2.2]
    omega_to_n = omegas[trial_num % 2]

    # generate random i,j,k
    i, j, k = get_rand_ijk()

    # make output directory and file
    dir_path = "./data/v2.2_data"
    if trial_num == 0:
        if os.path.exists(dir_path):
            print("Error: Directory already exists")
            exit()
        os.mkdir(dir_path)
    else:
        while (not os.path.exists(dir_path)):
            time.sleep(1)

    # output format params
    nv = 6
    omega_ind = 0
    theta_ind = 1
    phi_ind = 2
    psi_ind = 3
    sk_ind = 4
    # inc_ind = 5
    t_ind = 5
    inds = (omega_ind,theta_ind,phi_ind,psi_ind,sk_ind,t_ind)

    ### RUN SIMULATION ###
    sim_params = i,j,k,a,Q_tide,R_p,theta,omega_to_n,M_p,k2,moment2,moment3,s_k_angle,a_out,i_out,M_out
    integrate_sim(dir_path,sim_params,trial_num,tf,nv,inds,out_step)

    ### Re-RUN SIMULATION with same parameters, except just j2 ###
    trial_num_2 = trial_num + 0.1
    moment2 = 0.
    sim_params = i,j,k,a,Q_tide,R_p,theta,omega_to_n,M_p,k2,moment2,moment3,s_k_angle,a_out,i_out,M_out
    integrate_sim(dir_path,sim_params,trial_num_2,tf,nv,inds,out_step)

# main function
if __name__ == '__main__':
    n_trials = 100
    start = time.time()
    with mp.Pool(processes=n_trials) as pool:
        pool.map(run_sim, range(n_trials))
    
    tot_time = time.time() - start
    hrs = tot_time // 3600
    mins = (tot_time % 3600) // 60
    secs = int((tot_time % 3600) % 60)
    print(f"Total Runtime: {hrs} hours {mins} minutes {secs} seconds", flush=True)
