import os
import numpy as np
import time
import scipy.stats as stats
import multiprocessing as mp

def calc_om_dot_lr(ts,omegas):
    result = stats.linregress(ts,omegas)
    return result.slope

def mp_calc_om_dot(trial_num):
    om_dots = np.zeros(2) # 0 for triaxial, 1 for oblate
    for k in range(2):
        if k==1:
            trial_num_dec = trial_num + .1
        else:
            trial_num_dec = int(trial_num)

        file_path = os.path.join(dir_path,"trial_"+str(trial_num_dec)+".npy")
        f = open(file_path, 'rb')
        data = np.load(f)
        om_dots[k] = calc_om_dot_lr(data[1],data[0])

    return om_dots

if __name__=="__main__":
    start = time.time()
    # tf=300.
    # out_step=1.
    perturber=False
    omega_lo = 0.
    omega_hi = 3.
    n_omegas = 900
    theta_lo = 0.
    theta_hi = 180.
    n_thetas = 360
    if perturber:
        dir = "3body_data_"+str(n_thetas)+":"+str(theta_lo)+"-"+str(theta_hi)
        dir_path = "./data/grid/"+dir
    else:
        dir = "2body_data_"+str(n_thetas)+":"+str(theta_lo)+"-"+str(theta_hi)
        dir_path = "./data/grid/"+dir

    omega_dots = np.zeros((2,n_thetas,n_omegas)) # first dimension corresponds to triax (0) or oblate (1)

    with mp.Pool() as pool:
        om_dots = pool.map(mp_calc_om_dot, range(n_omegas*n_thetas))

    for i in range(n_thetas):
        for j in range(n_omegas):
            trial_num = i*n_omegas + j
            omega_dots[0,i,j] = om_dots[trial_num][0]
            omega_dots[1,i,j] = om_dots[trial_num][1]

    file_path = os.path.join(dir_path,"grid_data.npy")
    
    with open(file_path, 'wb') as f:
        np.save(f, omega_dots)

    print(time.time()-start)