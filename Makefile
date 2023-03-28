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

3body:
	python3 3body.py > output.txt 2> errors.txt

plt_omdot:
	python3 plot_sp_dot.py

3bd_plot:
	python3 3body_plot_v2.py

grid:
	python3 sp_dot_grid.py > output.txt 2> errors.txt

plot_grid:
	python3 plot_grid.py

calc_grid:
	python3 calc_grid.py

eq_plot:
	python3 2body_plot.py
	open *.png

plot_torque:
	python3 plot_torque.py
	open *.png

clean:
	rm -rf *.png
	rm -rf *.txt
