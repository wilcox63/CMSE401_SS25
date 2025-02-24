all: process process_omp process_serial

main_process.o: main_process.c
	gcc -c main_process.c 

png_util.o: png_util.c
	gcc -l lpng16 -c png_util.c

main_process_serial.o: main_process_serial.c
	gcc -c main_process_serial.c

main_process_omp.o: main_process_omp.c
	gcc -c -fopenmp main_process_omp.c

process: main_process.o png_util.o
	gcc -O3 -o process -lm -l png16 main_process.o png_util.o

process_serial: main_process_serial.o png_util.o
	gcc -O3 -o process_serial -lm -l png16 main_process_serial.o png_util.o

process_omp: main_process_omp.o png_util.o
	gcc -O3 -fopenmp -o process_omp -lm -l png16 main_process_omp.o png_util.o

test: process
	./process ./images/cube.png test.png

clean:
	rm *.o
	rm process 
	rm process_serial
	rm process_omp
