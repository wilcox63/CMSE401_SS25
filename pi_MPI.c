#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

static long num_steps = 100000;
double step;

int main(int argc, char* argv[]) {
    int i, rank, size;
    double x, pi, local_sum = 0.0, global_sum;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    step = 1.0 / (double) num_steps;
    
    // Each process computes its part
    for (i = rank; i < num_steps; i += size) {
        x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }
    
    // Reduce all local sums to get the global sum
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        pi = global_sum * step;
        printf("Computed PI = %.16f\n", pi);
    }
    
    MPI_Finalize();
    return 0;
}
