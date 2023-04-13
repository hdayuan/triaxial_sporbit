import os
import numpy as np
import time
import sys
import so_params as sops
import plot_funcs as pfs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)


if __name__=="__main__":

    trial = int(562)

    data_dir="./data"
    plots_dir = "./plots"
    subdir = "grid/2body_40.0.0-180.0_40.1.97-2.0"

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
        pfs.plot_trial(i==0,axs,data,0,250,1,1,inds,t_unit=1.)

    plt.savefig(os.path.join(plots_dir,subdir,"trial_"+str(trial)+".png"), dpi=300)
    plt.clf()
    plt.close(fig)