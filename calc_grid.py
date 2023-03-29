import os
import numpy as np
import time
import scipy.stats as stats
import multiprocessing as mp

def calc_om_dot_v2(ts,omegas,tnd):
    buffer = 10
    ds = 2
    max_nex = 16
    ts = ts[::ds]
    omegas = omegas[::ds]
    n_data = len(omegas)
    d_omegas = omegas[1:] - omegas[:-1]

    # test for roughly linear
    if np.all(d_omegas >= 0) or np.all(d_omegas <= 0):
        slope = stats.linregress(ts,omegas).slope
    
    # otherwise assume sinusoidal
    else:
        squared_d_oms = d_omegas*d_omegas
        sorted_ds = np.argsort(squared_d_oms)
        min_inds = []
        max_inds = []
        min_count = 0
        max_count = 0
        for i in range(len(d_omegas)):
            if min_count >= max_nex // 2 and max_count >= max_nex // 2:
                break
            ind = sorted_ds[i]
            if ind == 0 or ind == len(d_omegas) - 1:
                continue
            
            extreme = np.mean(omegas[ind:ind+2])

            if omegas[ind-1] > extreme and omegas[ind+2] > extreme and min_count < max_nex // 2:
                # then this is a local minimum
                min_inds.append(ind)
                min_count += 1

            elif omegas[ind-1] < extreme and omegas[ind+2] < extreme and max_count < max_nex // 2:
                # then this is a local maximum
                max_inds.append(ind)
                max_count += 1

            else:
                lo = ind - 10
                if  lo < 0:
                    lo = 0
                hi = ind + 12
                if hi > n_data:
                    hi = n_data

                left_avrg = np.mean(omegas[lo:ind])
                right_avrg = np.mean(omegas[ind+2:hi])

                if left_avrg > extreme and right_avrg > extreme and min_count < max_nex // 2:
                    # then this is a local minimum
                    min_inds.append(ind)
                    min_count += 1

                elif left_avrg < extreme and right_avrg < extreme and max_count < max_nex // 2:
                    # then this is a local maximum
                    max_inds.append(ind)
                    max_count += 1
        
        min_inds = np.array(min_inds)
        max_inds = np.array(max_inds)
        if min_count >= 2:
            mins = np.array([np.mean(omegas[i:i+2]) for i in min_inds])
            t_mins = np.array([np.mean(ts[i:i+2]) for i in min_inds])
            min_slope = stats.linregress(t_mins,mins).slope
            min_bool = True
        else:
            min_bool = False

        if max_count >= 2:
            maxs = np.array([np.mean(omegas[i:i+2]) for i in max_inds])
            t_maxs = np.array([np.mean(ts[i:i+2]) for i in max_inds])
            max_slope = stats.linregress(t_maxs,maxs).slope
            max_bool = True
        else:
            max_bool = False

        if min_bool and max_bool:
            slope = (min_slope + max_slope) / 2.
        elif min_bool:
            slope = min_slope
        elif max_bool:
            slope = max_slope
        else:
            slope = stats.linregress(ts,omegas).slope
            print(f"Warning: Not enough extremes for trial {tnd}")

    return slope

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
        om_dots[k] = calc_om_dot_v2(data[1],data[0],trial_num_dec)

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