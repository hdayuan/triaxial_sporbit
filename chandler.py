import rebound
import reboundx
import numpy as np
import time

start = time.time()

sim = rebound.Simulation()
sim.integrator = "whfast"
sim.units = ('AU', 'yr', 'MSun')
sim.dt = 0.001
sim.add(m=1.)

rebx = reboundx.Extras(sim)
triax = rebx.load_operator("triaxial_torque")
rebx.add_operator(triax)

# add spin
ps = sim.particles
ps[0].params["tt_ix"] = 1.
ps[0].params["tt_iy"] = 0.
ps[0].params["tt_iz"] = 0.
ps[0].params["tt_jx"] = 0.
ps[0].params["tt_jy"] = 1.
ps[0].params["tt_jz"] = 0.
ps[0].params["tt_kx"] = 0.
ps[0].params["tt_ky"] = 0.
ps[0].params["tt_kz"] = 1.

ps[0].params["tt_Ii"] = 1.
ps[0].params["tt_Ij"] = 1.
ps[0].params["tt_Ik"] = 2.

ps[0].params["tt_si"] = 0.
ps[0].params["tt_sj"] = 0.5
ps[0].params["tt_sk"] = np.sqrt(3) / 2

ps[0].params["tt_omega"] = 4.*2*np.pi

f = open("chandler_rk_out.txt", "w")

for i in range(1000):
	sim.integrate(i*0.01) # in time units, not dt units
	f.write(str(sim.particles[0].params["tt_si"])+"\t")
	f.write(str(sim.particles[0].params["tt_sj"])+"\t")
	f.write(str(sim.particles[0].params["tt_sk"])+"\t")
	f.write(str(sim.t)+"\n")

f.close()

print(time.time() - start)
