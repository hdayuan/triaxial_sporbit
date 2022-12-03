compile:
	./recompile.sh

run:
	python3 plot_torque.py

plot:
	open *.png

clean:
	rm -rf *.txt
	rm -rf *.png
