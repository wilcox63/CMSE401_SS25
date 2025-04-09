CC = mpicc

CFLAGS = -O2 -Wall

LIBS = -lm



all: monte_pi submit



monte_pi: mpi_monte_carlo.c

	$(CC) $(CFLAGS) -o monte_pi mpi_monte_carlo.c $(LIBS)

submit: monte_pi
	sbatch submit.sh


clean:

	rm -f monte_pi *.o
