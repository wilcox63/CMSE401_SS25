#include <stdlib.h>

#include <stdio.h>

#include <mpi.h>



#define sqr(x) ((x)*(x))

long random(void);



double dboard(int darts) {

    double x_coord, y_coord, pi, r;

    int score = 0;

    long rd;

    unsigned long cconst = RAND_MAX;



    for (int n = 0; n < darts; n++) {

        rd = random();

        r = (double)rd / cconst;

        x_coord = (2.0 * r) - 1.0;

        r = (double)random() / cconst;

        y_coord = (2.0 * r) - 1.0;



        if ((sqr(x_coord) + sqr(y_coord)) <= 1.0)

            score++;

    }



    pi = 4.0 * (double)score / (double)darts;

    return pi;

}



int main(int argc, char **argv) {

    int rank, size;

    int total_darts = 1000000;

    int darts_per_proc;

    double pi, local_pi, start, end;



    MPI_Init(&argc, &argv);           // Step 2: Initialize MPI

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);



    darts_per_proc = total_darts / size;



    start = MPI_Wtime();              // Start timing



    local_pi = dboard(darts_per_proc);



    // Step 3: Reduce to Master

    MPI_Reduce(&local_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);



    end = MPI_Wtime();                // End timing



    if (rank == 0) {

        pi = pi / size;

        printf("Estimated PI: %.16f\n", pi);

        printf("Execution Time: %f seconds\n", end - start); // Step 4

    }



    MPI_Finalize();                   // Step 2: Finalize MPI

    return 0;

}


