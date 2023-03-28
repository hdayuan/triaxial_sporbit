import os
# import rebound
# import reboundx
import numpy as np
# import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats

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

def calc_om_dot_lr(ts,omegas,tnd,plots_dir):
    result = stats.linregress(ts,omegas)
    if omegas[0] > 1.998:
        plt.scatter(ts,omegas,color='black',s=0.5)
        plt.plot(ts,result.slope*ts + omegas[0],color="red")
        plt.savefig(os.path.join(plots_dir,"trial_"+str(tnd)+".png"), dpi=300)
        plt.clf()

    return result.slope

def calc_om_dot_v2(ts,omegas,tnd,plots_dir):
    n_data = len(omegas)
    d_omegas = omegas[1:] - omegas[:-1]
    dd_omegas = d_omegas[1:] - d_omegas[:-1]

    # test for roughly linear
    if np.all(d_omegas >= 0) or np.all(d_omegas <= 0):
        slope = stats.linregress(ts,omegas).slope
    
    # otherwise assume sinusoidal
    else:
        squared_d_oms = d_omegas*d_omegas
        sorted_ds = np.argsort(squared_d_oms)
        min_inds = []
        max_inds = []
        count = 0
        for i in range(len(d_omegas)):
            if count >= n_data//50:
                break
            ind = sorted_ds[i]
            if ind == 0 or ind == len(d_omegas) - 1:
                continue

            if dd_omegas[ind-1] > 0 and dd_omegas[ind] > 0:
                # then this is a local minimum
                min_inds.append(ind)
                count += 1
                continue

            if dd_omegas[ind-1] < 0 and dd_omegas[ind] < 0:
                # then this is a local maximum
                max_inds.append(ind)
                count += 1

        if len(min_inds) == 0 and len(max_inds) == 0:
            print(tnd)
            plt.scatter(ts,omegas,color='black',s=0.5)
            plt.savefig(os.path.join(plots_dir,"trial_"+str(tnd)+".png"), dpi=300)
            plt.clf()
            return 0
        
        if len(min_inds) > len(max_inds):
            inds = np.array(min_inds)
        else:
            inds = np.array(max_inds)

        indi = np.min(inds)
        indf = np.max(inds)
        slope = (np.mean(omegas[indf:indf+1]) - np.mean(omegas[indi:indi+1])) / (ts[indf]-ts[indi])
    
    if omegas[0] > 1.998:
        plt.scatter(ts,omegas,color='black',s=0.5)
        plt.plot(ts,slope*ts + omegas[0],color="red")
        plt.savefig(os.path.join(plots_dir,"trial_"+str(tnd)+".png"), dpi=300)
        plt.clf()

    return slope


def calc_om_dot(ts,omegas):
    n = len(omegas)
    min_delta_t = n/4
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

if __name__=="__main__":
    # tf=300.
    # out_step=1.
    from_file=False
    perturber=False
    omega_lo = 1.98
    omega_hi = 2.0
    n_omegas = 20
    theta_lo = 0.
    theta_hi = 180.
    n_thetas = 40
    if perturber:
        dir = "3body_"+str(n_thetas)+"."+str(theta_lo)+"-"+str(theta_hi)+"_"+str(n_omegas)+"."+str(omega_lo)+"-"+str(omega_hi)
    else:
        dir = "2body_"+str(n_thetas)+"."+str(theta_lo)+"-"+str(theta_hi)+"_"+str(n_omegas)+"."+str(omega_lo)+"-"+str(omega_hi)
    dir_path = "./data/grid/"+dir

    plots_dir = os.path.join("plots","grid",dir)
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    # n_out = int((tf // out_step) + 1)
    omegas = np.linspace(omega_lo, omega_hi, n_omegas)
    thetas = np.linspace(theta_lo,theta_hi,n_thetas)
    omega_grid, theta_grid = np.meshgrid(omegas,thetas)

    if not from_file:
        omega_dots = np.zeros((2,n_thetas,n_omegas)) # first dimension corresponds to triax (0) or oblate (1)

        for i in range(n_thetas):
            for j in range(n_omegas):
                trial_num = i*n_omegas + j
                for k in range(2):
                    if k==1:
                        trial_num_dec = trial_num + .1
                    else:
                        trial_num_dec = int(trial_num)

                    file_path = os.path.join(dir_path,"trial_"+str(trial_num_dec)+".npy")
                    f = open(file_path, 'rb')
                    data = np.load(f)
                    omega_dots[k,i,j] = calc_om_dot_v2(data[1],data[0],trial_num_dec,plots_dir)
                    # if omega_dots[k,i,j] > 0: #(omega_grid[i,j] <= 2 and omega_grid[i,j] > 1.995):
                    #     plt.scatter(data[1],data[0],color='black',s=0.5)
                    #     plt.savefig(os.path.join(plots_dir,"trial_"+str(trial_num_dec)+".png"), dpi=300)
                    #     plt.clf()

    else:
        file_path = os.path.join(dir_path,"grid_data.npy")
        f = open(file_path, 'rb')
        omega_dots = np.load(f)


    # plot results
    fig, axs = plt.subplots(2, 1,figsize=(8, 8), sharex=True,sharey=True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=.95, top=0.92, wspace=0.1, hspace=0.1)
    # axs[0].set_xlabel(r"$\Omega/n$")
    axs[1].set_xlabel(r"$\Omega/n$")
    axs[0].set_ylabel(r"$\theta$ ($^{\circ}$)")
    axs[1].set_ylabel(r"$\theta$ ($^{\circ}$)")
    
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
    lab = r"$d\Omega/dt$ ($n/P^2$)"
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap='coolwarm'), ax=axs[1],label=lab)

    plt.savefig(os.path.join(plots_dir,"omega_dot.png"), dpi=300)
    plt.clf()
    plt.close(fig)