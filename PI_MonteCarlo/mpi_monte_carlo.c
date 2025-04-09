#include <stdlib.h>

#include <stdio.h>

#include <mpi.h>

#include <time.h>

#include <math.h>



#define sqr(x) ((x)*(x))



double dboard(int darts)

{

    double x_coord,       /* x coordinate, between -1 and 1  */

           y_coord,       /* y coordinate, between -1 and 1  */

           pi,            /* pi  */

           r;             /* random number between 0 and 1  */

    int score,            /* number of darts that hit circle */

        n;

    unsigned long cconst; /* used to convert integer random number to double between 0 and 1 */



    cconst = RAND_MAX;

    score = 0;



    /* "throw darts at board" */

    for (n = 1; n <= darts; n++) {

        /* generate random numbers for x and y coordinates */

        r = (double)rand()/cconst;

        x_coord = (2.0 * r) - 1.0;

        r = (double)rand()/cconst;

        y_coord = (2.0 * r) - 1.0;



        /* if dart lands in circle, increment score */

        if ((sqr(x_coord) + sqr(y_coord)) <= 1.0)

            score++;

    }



    /* calculate pi */

    pi = 4.0 * (double)score/(double)darts;

    return(pi);

} 



int main(int argc, char ** argv) {

    int rank, size, darts;

    double pi, local_pi, start_time, end_time;

    int total_darts = 1000000; // Default value

    

    // Step 2: Initialize MPI

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    

    // Parse command line arguments if provided

    if (argc > 1) {

        if (rank == 0) {

            total_darts = atoi(argv[1]);

            printf("Using %d darts\n", total_darts);

        }

        MPI_Bcast(&total_darts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    }

    

    // Calculate how many darts each process should throw

    darts = total_darts / size;

    

    // Set different seeds for each process

    srand(time(NULL) + rank);

    

    // Start timing

    start_time = MPI_Wtime();

    

    // Step 3: Separate the Master (Rank 0) with the workers

    local_pi = dboard(darts);

    

    // Reduce all local pi calculations to get the average

    MPI_Reduce(&local_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    

    // End timing

    end_time = MPI_Wtime();

    

    // Master process averages the results and prints

    if (rank == 0) {

        pi = pi / size;

        printf("Approximated value of pi: %.16f\n", pi);

        printf("Error: %.16f\n", fabs(pi - M_PI));

        printf("Execution time: %.6f seconds\n", end_time - start_time);

        

        // Step 4: Benchmark - print parallel efficiency

        printf("\n--- Benchmark Results ---\n");

        printf("Number of processes: %d\n", size);

        printf("Total darts: %d\n", total_darts);

        printf("Darts per process: %d\n", darts);

        printf("Time taken: %.6f seconds\n", end_time - start_time);

    }

    

    // Step 2: Finalize MPI

    MPI_Finalize();

    return 0;

}
