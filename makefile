all: process

2D_Wave.o: 2D_Wave.c
	gcc -c 2D_Wave.c 

png_util.o: png_util.c
	gcc -l lpng16 -c png_util.c

process: 2D_Wave.o png_util.o
	gcc -O3 -o process_wave -lm -l png16 2D_Wave.o png_util.o

test: process
	./process ./images/cube.png test.png

clean:
	rm *.o
	rm process_wave
