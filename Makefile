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

equi:
	python3 two_body_equi_conv.py

eq_plot:
	python3 2body_plot.py
	open *.png

plot_torque:
	python3 plot_torque.py
	open *.png

clean:
	rm -rf *.png

trash_data:
	rm -rf 2body_equi_data
