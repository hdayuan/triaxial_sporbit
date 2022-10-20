import rebound
import reboundx

sim = rebound.Simulation()
sim.add(m=1.)
sim.add(a=1., m=0.01)

rebx = reboundx.Extras(sim)
stark = rebx.load_force("stark_force")
rebx.add_force(stark)

f = open("stark_force_out.txt", "w")

for i in range(100):
	sim.integrate(1000*i)
	f.write(str(sim.particles[0].vx)+"\n")

f.close()
