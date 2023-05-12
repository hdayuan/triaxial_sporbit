import os
import numpy as np
import time
import sys
import so_params as sops
import plot_funcs as pfs
import scipy.stats as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)


def calc_om_dot_v2(ts,omegas,ax):
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
        ax.plot(ts[min_inds],omegas[min_inds],'o',markersize=6,color='tab:blue')
        ax.plot(ts[max_inds],omegas[max_inds],'o',markersize=6,color='tab:blue')
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

        ax.plot(ts,omegas[0]+(slope*ts),'--',color='tab:red',lw=1.5)
        ax.plot(ts,2.01695+(slope*ts),'--',color='tab:red',lw=1.5)

    return slope

def plt_om_dot_demo():
    trial = int(14105)
    lo = 0
    hi = 1001
    ds = 2

    data_dir="./data"
    plots_dir = "./plots"
    subdir = "grid/2body_180.0.0-180.0_200.1.75-2.25"

    val_names = ["ix","iy","iz","jx","jy","jz","kx","ky","kz","si","sj","sk","omega","rx","ry","rz","vx","vy","vz","t"] # r is vector from planet to star !
    inds = {val_names[i]:i for i in range(len(val_names))}

    fig, ax = plt.subplots(1, 1,figsize=(6, 2), sharex=True)
    plt.subplots_adjust(left=0.10, bottom=0.2, right=.98, top=0.98, wspace=0.15, hspace=0.08)
    ylabels = [r"$\Omega/n$"]
    ax.set_xlabel(r"Time ($P$)")
    ax.set_ylabel(ylabels[0])

    file = "trial_"+str(trial)+".npy"
    path = os.path.join(data_dir,subdir,file)

    f = open(path, 'rb')
    data = np.load(f)
    omegas = data[inds['omega'],lo:hi]
    ts = data[inds['t'],lo:hi]

    ax.plot(ts[::],omegas[::],c='black',lw=1.)
    calc_om_dot_v2(ts,omegas,ax)

    plt.savefig(os.path.join(plots_dir,subdir,"trial_"+str(trial)+"_omdot_demo.png"), dpi=300)
    plt.clf()
    plt.close(fig)

def plt_om_th_be():
    trial = int(14100)

    data_dir="./data"
    plots_dir = "./plots"
    subdir = "grid/2body_180.0.0-180.0_200.1.75-2.25"

    val_names = ["ix","iy","iz","jx","jy","jz","kx","ky","kz","si","sj","sk","omega","rx","ry","rz","vx","vy","vz","t"] # r is vector from planet to star !
    inds = {val_names[i]:i for i in range(len(val_names))}

    fig, axs = pfs.get_fig_axs()

    for i in range(2):
        if i == 0:
            file = "trial_"+str(trial)+".npy"
        else:
            file = "trial_"+str(trial+0.1)+".npy"
        path = os.path.join(data_dir,subdir,file)

        f = open(path, 'rb')
        data = np.load(f)
        pfs.plot_trial(i==0,axs,data,0,-1,1,1,inds,t_unit=1.)

    plt.savefig(os.path.join(plots_dir,subdir,"trial_"+str(trial)+".png"), dpi=300)
    plt.clf()
    plt.close(fig)

if __name__=="__main__":
    # plt_om_th_be()

    plt_om_dot_demo()

    