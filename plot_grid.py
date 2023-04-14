import os
# import rebound
# import reboundx
import numpy as np
import so_params as sops
# import time
import scipy.stats as stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
plt.rc('lines', lw=2.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

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
    # if omegas[0] > 1.998:
    #     plt.scatter(ts,omegas,color='black',s=0.5)
    #     plt.plot(ts,result.slope*ts + omegas[0],color="tab:red")
    #     plt.savefig(os.path.join(plots_dir,"trial_"+str(tnd)+".png"), dpi=300)
    #     plt.clf()

    return result.slope

def calc_om_dot_v2(ts,omegas,tnd,plots_dir):
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

        plt.scatter(ts,omegas,color='black',s=0.5)
        plt.plot(ts,slope*ts + omegas[0],color="tab:red")
        plt.savefig(plots_dir+"/trial_"+str(tnd)+".png", dpi=300)
        plt.clf()
    
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

            # plt.scatter(ts,omegas,color='black',s=0.5)
            # plt.plot(ts,slope*ts + omegas[0],color="tab:red")
            # plt.scatter([ts[i] for i in min_inds],[omegas[i] for i in min_inds],color="blue")
            # plt.scatter([ts[i] for i in max_inds],[omegas[i] for i in max_inds],color="blue")
            # plt.savefig(os.path.join(plots_dir,"trial_"+str(tnd)+".png"), dpi=300)
            # plt.clf()
        theta_lo = 0.
        theta_hi = 180.
        n_thetas = 40
        n_omegas = 20
        thetas = np.linspace(theta_lo,theta_hi,n_thetas)
        theta = thetas[int(tnd) % n_omegas]
        # if slope > 1.e-6 or slope < -1.e-6 or (omegas[0] >= 2 and omegas[0] < 2.01 and theta >=50 and theta < 75):
        plt.scatter(ts,omegas,color='black',s=0.5)
        plt.plot(ts,slope*ts + omegas[0],color="tab:red")
        plt.scatter([ts[i] for i in min_inds],[omegas[i] for i in min_inds],color="blue")
        plt.scatter([ts[i] for i in max_inds],[omegas[i] for i in max_inds],color="blue")
        plt.savefig(plots_dir+"/trial_"+str(tnd)+".png", dpi=300)
        plt.clf()

    # if omegas[0] > 1.998:
    #     plt.scatter(ts,omegas,color='black',s=0.5)
    #     plt.plot(ts,slope*ts + omegas[0],color="tab:red")
    #     plt.savefig(os.path.join(plots_dir,"trial_"+str(tnd)+".png"), dpi=300)
    #     plt.clf()
    # if slope < -2.e-5 or slope > 1.e-5:
    #     plt.scatter(ts,omegas,color='black',s=0.5)
    #     plt.plot(ts,slope*ts + omegas[0],color="tab:red")
    #     plt.savefig(os.path.join(plots_dir,"trial_"+str(tnd)+".png"), dpi=300)
    #     plt.clf()

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

def calc_om_dot_simple(ts,omegas):
    return (omegas[-1]-omegas[0]) / (ts[-1]-ts[0])

if __name__=="__main__":
    beta_bool = False # if false, theta vs omega, if true, beta vs omega, use all theta valiables for beta
    theta_fix = float(120.) # degrees
    short_bool = False
    # tf=300.
    # out_step=1.
    from_file=True
    perturber=False
    version = 2 # 1 for 3body_data_..., 2 for 3body_...
    omega_lo = float(1.75)
    omega_hi = float(2.25)
    n_omegas = 200
    theta_lo = float(0.)
    theta_hi = float(180.)
    n_thetas = 180
    if beta_bool:
        if short_bool:
            proto_dir = "beta_s"+str(theta_fix)+"th_"
        else:
            proto_dir = "beta_"+str(theta_fix)+"th_"
    else:
        if short_bool:
            proto_dir = "ss_"
        else:
            proto_dir = ""
    if perturber:
        if version == 1:
            dir = "3body_data_"+str(n_thetas)+":"+str(theta_lo)+"-"+str(theta_hi)
        elif version == 2:
            dir = proto_dir+"3body_"+str(n_thetas)+"."+str(theta_lo)+"-"+str(theta_hi)+"_"+str(n_omegas)+"."+str(omega_lo)+"-"+str(omega_hi)
    else:
        if version == 1:
            dir = "2body_data_"+str(n_thetas)+":"+str(theta_lo)+"-"+str(theta_hi)
        elif version == 2:
            dir = proto_dir+"2body_"+str(n_thetas)+"."+str(theta_lo)+"-"+str(theta_hi)+"_"+str(n_omegas)+"."+str(omega_lo)+"-"+str(omega_hi)
    dir_path = "./data/grid/"+dir

    plots_dir = os.path.join("plots","grid",dir)
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    # n_out = int((tf // out_step) + 1)
    omegas = np.linspace(omega_lo, omega_hi, n_omegas)
    thetas = np.linspace(theta_lo,theta_hi,n_thetas)
    omega_grid, theta_grid = np.meshgrid(omegas,thetas)

    if not from_file:
        ds = 1
        omega_dots = np.zeros((2,n_thetas,n_omegas)) # first dimension corresponds to triax (0) or oblate (1)
        theta_dots = np.zeros((2,n_thetas,n_omegas)) # first dimension corresponds to triax (0) or oblate (1)
        val_names = ["ix","iy","iz","jx","jy","jz","kx","ky","kz","si","sj","sk","omega","rx","ry","rz","vx","vy","vz","t"] # r is vector from planet to star !
        inds = {val_names[i]:i for i in range(len(val_names))}
        for i in range(n_thetas):
            for j in range(n_omegas):
                trial_num = i*n_omegas + j
                for k in range(2):
                    if k==1:
                        trial_num_dec = trial_num + .1
                    else:
                        trial_num_dec = int(trial_num)

                    file_path = os.path.join(dir_path,"trial_"+str(trial_num_dec)+".npy")
                    if (not os.path.exists(file_path)):
                        continue
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

                    omega_dots[k,i,j] = calc_om_dot_simple(ts,omegas)
                    theta_dots[k,i,j] = calc_om_dot_simple(ts,thetas)
                    # calc_om_dot_v2(ts,omegas,trial_num_dec,plots_dir)

                    # omega_dots[k,i,j] = calc_om_dot_v2(data[1],data[0],trial_num_dec,plots_dir)
                    # if omega_dots[k,i,j] > 0: #(omega_grid[i,j] <= 2 and omega_grid[i,j] > 1.995):
                    #     plt.scatter(data[1],data[0],color='black',s=0.5)
                    #     plt.savefig(os.path.join(plots_dir,"trial_"+str(trial_num_dec)+".png"), dpi=300)
                    #     plt.clf()

    else:
        file_path = os.path.join(dir_path,"grid_data.npy")
        f = open(file_path, 'rb')
        omega_theta_dots = np.load(f)
        omega_dots = omega_theta_dots[:,0,:,:]
        theta_dots = omega_theta_dots[:,1,:,:]


    # crop
    # lo = int((6/60)*n_omegas)
    # hi = int((14/60)*n_omegas)
    # omega_dots = omega_dots[:,:,lo:hi]
    # theta_dots = theta_dots[:,:,lo:hi]
    # omega_grid = omega_grid[:,lo:hi]
    # theta_grid = theta_grid[:,lo:hi]
    
    # omega_dots = omega_dots[:,:,n_omegas//6:5*n_omegas//6 + 1]
    # omega_grid = omega_grid[:,n_omegas//6:5*n_omegas//6 + 1]
    # theta_grid = theta_grid[:,n_omegas//6:5*n_omegas//6 + 1]

    # plot results
    for i in range(2):
        if i == 1:
            omega_dots = theta_dots

        if i==1:
            fig, axs = plt.subplots(2, 1,figsize=(5, 7), sharex=True,sharey=True)
            plt.subplots_adjust(left=0.08, bottom=0.09, right=0.88, top=0.95, wspace=0., hspace=0.15)
        else:
            fig, axs = plt.subplots(2, 1,figsize=(5, 7), sharex=True,sharey=True)
            plt.subplots_adjust(left=0.13, bottom=0.09, right=0.93, top=0.95, wspace=0., hspace=0.15)
        # axs[0].set_xlabel(r"$\Omega/n$")
        axs[1].set_xlabel(r"$\Omega/n$")
        if beta_bool:
            axs[0].set_ylabel(r"$\beta$ ($^{\circ}$)")
            axs[1].set_ylabel(r"$\beta$ ($^{\circ}$)")
            axs[0].set_title(r"Triaxial, $\theta=$ "+str(theta_fix)+r"$^{\circ}$")
            axs[1].set_title(r"Oblate, $\theta=$ "+str(theta_fix)+r"$^{\circ}$")
        else:
            if i==0:
                axs[0].set_ylabel(r"$\theta$ ($^{\circ}$)")
                axs[1].set_ylabel(r"$\theta$ ($^{\circ}$)")
            axs[0].set_title("Triaxial")
            axs[1].set_title("Oblate")

        lab = r"$d\Omega/dt$ ($n/P$)"
        if i == 1:
            lab = r"$d\theta/dt$ ($^{\circ}/P$)"

        if i== 1:
            val = 0.80*np.maximum(np.max(omega_dots[1]),-np.min(omega_dots[1]))
        else:
            val = 0.80*np.maximum(np.max(omega_dots[0]),-np.min(omega_dots[0]))
        if i==1:
            val = 5.e-5
        # val = np.maximum(np.max(omega_dots[0]),-np.min(omega_dots[0])) # (np.max(omega_dots[0]) - np.min(omega_dots[0]))/2.
        norm = mpl.colors.Normalize(vmin=-val, vmax=val)
        axs[0].pcolormesh(omega_grid,theta_grid,omega_dots[0],norm=norm,cmap='coolwarm',shading='auto')
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap='coolwarm'), ax=axs[0],label=lab)
        
        val = 0.80*np.maximum(np.max(omega_dots[1]),-np.min(omega_dots[1]))
        if i == 1:
            val = 5.e-5
        # val = (np.max(omega_dots[1]) - np.min(omega_dots[1]))/2.
        norm = mpl.colors.Normalize(vmin=-val, vmax=val)
        axs[1].pcolormesh(omega_grid,theta_grid,omega_dots[1],norm=norm,cmap='coolwarm',shading='auto')
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap='coolwarm'), ax=axs[1],label=lab)

        if i == 1:
            save_name = "theta_dot.png"
        else:
            save_name = "omega_dot.png"
        plt.savefig(os.path.join(plots_dir,save_name), dpi=300)
        plt.clf()
        plt.close(fig)
