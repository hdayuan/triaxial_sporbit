compile:
	./recompile.sh

converge:
	python3 triaxial_torque_tests.py -co
	open *.png

spin:
	python3 triaxial_torque_tests.py -sp
	open *.png

obl:
	python3 triaxial_torque_tests.py -ob
	open *.png

plot_torque:
	python3 plot_torque.py
	open *.png
	
clean:
	rm -rf *.txt
	rm -rf *.png
