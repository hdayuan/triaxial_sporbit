import rebound
import reboundx
import numpy as np

sim = rebound.Simulation()
sim.integrator = "whfast"
sim.units = ('AU', 'yr', 'MSun')
# sim.dt = 
sim.add(m=1.)
sim.add(a=1., m=0.01)

rebx = reboundx.Extras(sim)
triax = rebx.load_operator("triaxial_torque")
rebx.add_operator(triax)

# add spin to smaller body
ps = sim.particles
ps[1].params["tt_ix"] = 1.
ps[1].params["tt_iy"] = 0.
ps[1].params["tt_iz"] = 0.
ps[1].params["tt_jx"] = 0.
ps[1].params["tt_jy"] = 1.
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

ps[1].params["tt_omega"] = 4.*2*np.pi

f = open("test_out.txt", "w")

for i in range(1000):
	sim.integrate(i*sim.dt) # in time units, not dt units
	f.write(str(sim.particles[1].params["tt_ix"])+"\t")
	f.write(str(sim.particles[1].params["tt_iy"])+"\t")
	f.write(str(sim.particles[1].params["tt_iz"])+"\t")
	f.write(str(sim.particles[1].params["tt_jx"])+"\t")
	f.write(str(sim.particles[1].params["tt_jy"])+"\t")
	f.write(str(sim.particles[1].params["tt_jz"])+"\t")
	f.write(str(sim.t)+"\n")

f.close()
