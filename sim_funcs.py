import rebound
import reboundx
import numpy as np

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