import os
import numpy as np
import time
import scipy.stats as stats
import multiprocessing as mp
import so_params as sops

beta_bool = True # if false, theta vs omega, if true, beta vs omega, use all theta valiables for beta
theta_fix = float(60.) # degrees
short_bool = True
# tf=300.
# out_step=1.
version = 2
perturber=False
omega_lo = float(1.95)
omega_hi = float(2.05)
n_omegas = 100
theta_lo = float(0.)
theta_hi = float(90.)
n_thetas = 90
proto_dir = "./data/grid/"
if beta_bool:
    proto_dir = "./data/grid/beta_ss"+str(theta_fix)+"th_"
if perturber:
    if version == 1:
        dir = "./data/grid/3body_data_"+str(n_thetas)+":"+str(theta_lo)+"-"+str(theta_hi)
    elif version == 2:
        dir = proto_dir+"3body_"+str(n_thetas)+"."+str(theta_lo)+"-"+str(theta_hi)+"_"+str(n_omegas)+"."+str(omega_lo)+"-"+str(omega_hi)
else:
    if version == 1:
        dir = "./data/grid/2body_data_"+str(n_thetas)+":"+str(theta_lo)+"-"+str(theta_hi)
    elif version == 2:
        dir = proto_dir+"2body_"+str(n_thetas)+"."+str(theta_lo)+"-"+str(theta_hi)+"_"+str(n_omegas)+"."+str(omega_lo)+"-"+str(omega_hi)
dir_path = dir

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
                lo = ind - buffer
                if  lo < 0:
                    lo = 0
                hi = ind + 2 + buffer
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
            # print(f"Warning: Not enough extremes for trial {tnd}")

    return slope

def calc_om_dot_simple(ts,omegas,tnd):
    delta_omega = omegas[-1]-omegas[0]
    delta_t = ts[-1]-ts[0]
    return delta_omega/delta_t

def mp_calc_om_dot(trial_num):
    ds = 1
    om_th_dots = np.zeros((2,2)) # first dimension corresponds to triax (0) or oblate (1)
    # second dimension corresponds to omega (0) or theta (1)

    val_names = ["ix","iy","iz","jx","jy","jz","kx","ky","kz","si","sj","sk","omega","rx","ry","rz","vx","vy","vz","t"] # r is vector from planet to star !
    inds = {val_names[i]:i for i in range(len(val_names))}
    
    for k in range(2):
        if k==1:
            trial_num_dec = trial_num + .1
        else:
            trial_num_dec = int(trial_num)

        file_path = os.path.join(dir_path,"trial_"+str(trial_num_dec)+".npy")
        f = open(file_path, 'rb')
        data = np.load(f)
        
        rs = np.stack((data[inds['rx'],::ds],data[inds['ry'],::ds],data[inds['rz'],::ds]), axis=0)
        rs /= sops.many_mags(rs)
        vs = np.stack((data[inds['vx'],::ds],data[inds['vy'],::ds],data[inds['vz'],::ds]), axis=0)
        ss = np.stack((data[inds['si'],::ds],data[inds['sj'],::ds],data[inds['sk'],::ds]), axis=0)
        iss = np.stack((data[inds['ix'],::ds],data[inds['iy'],::ds],data[inds['iz'],::ds]), axis=0)
        js = np.stack((data[inds['jx'],::ds],data[inds['jy'],::ds],data[inds['jz'],::ds]), axis=0)
        ks = np.stack((data[inds['kx'],::ds],data[inds['ky'],::ds],data[inds['kz'],::ds]), axis=0)
        ts = data[inds['t'],::ds]

        n = np.sqrt(np.dot(vs[:,0],vs[:,0])) / np.sqrt(np.dot(rs[:,0],rs[:,0])) # mean-motion

        omegas = data[inds['omega'],::ds]
        theta_rad, phi_rad = sops.get_theta_phi(ss,iss,js,ks,rs,vs)
        thetas = np.degrees(theta_rad)

        if short_bool:
            om_th_dots[k,0] = calc_om_dot_simple(ts,omegas,trial_num_dec)
            om_th_dots[k,1] = calc_om_dot_simple(ts,thetas,trial_num_dec)
        else:
            om_th_dots[k,0] = calc_om_dot_v2(ts,omegas,trial_num_dec)
            om_th_dots[k,1] = calc_om_dot_v2(ts,thetas,trial_num_dec)
        # if om_th_dots[k,0] > 1.e-8 or om_th_dots[k,0] < -1.e-8:
        #     print(trial_num_dec)
        # if om_th_dots[k,1] > 1.e-5 or om_th_dots[k,1] < -1.e-5:
        #     print(trial_num_dec)

    return om_th_dots

if __name__=="__main__":
    start = time.time()

    omega_theta_dots = np.zeros((2,2,n_thetas,n_omegas)) # first dimension corresponds to triax (0) or oblate (1)
    # second dimension corresponds to omega (0) or theta (1)

    with mp.Pool() as pool:
        dots = pool.map(mp_calc_om_dot, range(n_omegas*n_thetas))

    for i in range(n_thetas):
        for j in range(n_omegas):
            trial_num = i*n_omegas + j
            omega_theta_dots[0,0,i,j] = dots[trial_num][0,0]
            omega_theta_dots[1,0,i,j] = dots[trial_num][1,0]
            omega_theta_dots[0,1,i,j] = dots[trial_num][0,1]
            omega_theta_dots[1,1,i,j] = dots[trial_num][1,1]

    file_path = os.path.join(dir_path,"grid_data.npy")
    
    with open(file_path, 'wb') as f:
        np.save(f, omega_theta_dots)

    print(time.time()-start)