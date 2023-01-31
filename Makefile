compile:
	./recompile.sh

conv:
	python3 triaxial_torque_tests.py -convergence
	open *.png

spin:
	python3 triaxial_torque_tests.py -spin
	open *.png

obli:
	python3 triaxial_torque_tests.py -obliquity
	open *.png

chan:
	python3 triaxial_torque_tests.py -chandler
	open *.png

osci:
	python3 triaxial_torque_tests.py -oscillation
	open *.png

plot_torque:
	python3 plot_torque.py
	open *.png
	
clean:
	rm -rf *.png
