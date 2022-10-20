import rebound
import reboundx
import numpy as np
import time

start = time.time()

sim = rebound.Simulation()
sim.integrator = "whfast"
sim.units = ('AU', 'yr', 'MSun')
sim.dt = 0.0125 / 64
sim.add(m=1.)

v = 6.286207389817359
d_theta = v*sim.dt
x_val = np.cos(d_theta)
y_val = -np.sin(d_theta)
vx_val = v*np.sin(d_theta)
vy_val = v*np.cos(d_theta)

sim.add(m=0.001, x=x_val,y=y_val,vx=vx_val,vy=vy_val)
# sim.add(m=0.001, a=1.)

rebx = reboundx.Extras(sim)
triax = rebx.load_operator("triaxial_torque")
rebx.add_operator(triax)

# add spin to smaller body
ps = sim.particles
angle = 1.*np.pi/180.
ps[1].params["tt_ix"] = np.cos(angle)
ps[1].params["tt_iy"] = np.sin(angle)
ps[1].params["tt_iz"] = 0.
ps[1].params["tt_jx"] = -np.sin(angle)
ps[1].params["tt_jy"] = np.cos(angle)
ps[1].params["tt_jz"] = 0.
ps[1].params["tt_kx"] = 0.
ps[1].params["tt_ky"] = 0.
ps[1].params["tt_kz"] = 1.

ps[1].params["tt_Ii"] = 1.
ps[1].params["tt_Ij"] = 2.
ps[1].params["tt_Ik"] = 3.

ps[1].params["tt_si"] = 0.
ps[1].params["tt_sj"] = 0.
ps[1].params["tt_sk"] = 1.

ps[1].params["tt_omega"] = 2*np.pi / ps[1].P

f = open("test_torque_out_"+str(sim.dt)+"dt.txt", "w")

#############################################################

# sim.integrate(2*sim.dt)
# rx = ps[0].x - ps[1].x
# ry = ps[0].y - ps[1].y
# rz = ps[0].z - ps[1].z
# r = np.sqrt(rx**2 + ry**2 +rz**2)
# rx /= r
# ry /= r
# rz /= r
# i_dot_r = ps[1].params["tt_ix"]*rx + ps[1].params["tt_iy"]*ry + ps[1].params["tt_iz"]*rz
# j_dot_r = ps[1].params["tt_jx"]*rx + ps[1].params["tt_jy"]*ry + ps[1].params["tt_jz"]*rz
# f.write(str(np.arctan(j_dot_r/i_dot_r))+"\t")
# f.write(str(ps[1].params['tt_omega'])+"\t")
# f.write(str(ps[1].params['tt_ix'])+"\t")
# f.write(str(ps[1].params['tt_iy'])+"\t")
# f.write(str(ps[1].params['tt_iz'])+"\t")
# f.write(str(ps[1].params['tt_jx'])+"\t")
# f.write(str(ps[1].params['tt_jy'])+"\t")
# f.write(str(ps[1].params['tt_jz'])+"\t")
# f.write(str(ps[1].params['tt_kx'])+"\t")
# f.write(str(ps[1].params['tt_ky'])+"\t")
# f.write(str(ps[1].params['tt_kz'])+"\t")
# f.write(str(sim.t)+"\n")

for i in range(1000):
    sim.integrate(i*0.1)
    rx = ps[0].x - ps[1].x
    ry = ps[0].y - ps[1].y
    rz = ps[0].z - ps[1].z
    r = np.sqrt(rx**2 + ry**2 +rz**2)
    rx /= r
    ry /= r
    rz /= r
    i_dot_r = ps[1].params["tt_ix"]*rx + ps[1].params["tt_iy"]*ry + ps[1].params["tt_iz"]*rz
    j_dot_r = ps[1].params["tt_jx"]*rx + ps[1].params["tt_jy"]*ry + ps[1].params["tt_jz"]*rz
    f.write(str(np.arctan(j_dot_r/i_dot_r))+"\t")
    f.write(str(ps[1].params['tt_omega'])+"\t")
    f.write(str(ps[1].params['tt_ix'])+"\t")
    f.write(str(ps[1].params['tt_iy'])+"\t")
    f.write(str(ps[1].params['tt_iz'])+"\t")
    f.write(str(ps[1].params['tt_jx'])+"\t")
    f.write(str(ps[1].params['tt_jy'])+"\t")
    f.write(str(ps[1].params['tt_jz'])+"\t")
    f.write(str(ps[1].params['tt_kx'])+"\t")
    f.write(str(ps[1].params['tt_ky'])+"\t")
    f.write(str(ps[1].params['tt_kz'])+"\t")
    f.write(str(sim.t)+"\n")

f.close()

print(time.time() - start)