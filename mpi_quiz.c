#include <stdio.h>

#include <mpi.h>



int main(int argc, char **argv) {

    int rank, size;

    int msg1;

    int msg2;

    int msg3;

    MPI_Status status;



    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);





    if (rank == 0) {

        msg1=rank;

        MPI_Send(&msg1, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
	
	MPI_Recv(&msg3, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, &status);

    } else if (rank == 1) {

        msg2=rank;

        MPI_Send(&msg2, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);

	MPI_Recv(&msg1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    } else if (rank == 2){
	
	msg3=rank;

	MPI_Send(&msg3, 1, MPI_INT, 0,0, MPI_COMM_WORLD);

	MPI_Recv(&msg2, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);

   } if (rank == 0){

        printf("I am rank %d and received: %d \n",rank, msg3);

    } else if (rank == 1){

        printf("I am rank %d and received: %d \n",rank, msg1);

    } else if (rank == 2){
	
	printf("I am rank %d and received: %d \n", rank, msg2);
   }

    MPI_Finalize();

    return 0;

}
