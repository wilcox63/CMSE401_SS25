#include <mpi.h>

#include <stdio.h>

#include <stdlib.h>

#include <time.h>



int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);  // Initialize MPI



    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get process rank

    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes



    srand(time(NULL) + rank);  // Seed random number generator differently for each process

    int random_number = rand() % 100;  // Generate a random number (0-99)



    if (rank != size - 1) {  

        // All processes except the last one send their number

        MPI_Send(&random_number, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD);

        printf("Rank %d sent %d to Rank %d\n", rank, random_number, size - 1);

    } else {  

        // Last process (Rank size-1) collects numbers and computes mean

        int sum = 0, received;

        for (int i = 0; i < size - 1; i++) {

            MPI_Recv(&received, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            sum += received;

        }

        double mean = sum / (double)(size - 1);

        printf("Rank %d computed mean: %.2f\n", rank, mean);

    }



    MPI_Finalize();  // Clean up MPI environment

    return 0;

}
