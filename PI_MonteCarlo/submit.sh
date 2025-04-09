#!/bin/bash

#SBATCH --job-name=monte_pi

#SBATCH --output=monte_pi_%j.out

#SBATCH --error=monte_pi_%j.err

#SBATCH --ntasks=4

#SBATCH --time=00:10:00





##SBATCH --login



# Load MPI module (adjust this based on your cluster)

module load OpenMPI



# Run the program with different numbers of processes

echo "Running with 1 process"

mpirun -np 1 ./monte_pi 10000000



echo "Running with 2 processes"

mpirun -np 2 ./monte_pi 10000000



echo "Running with 4 processes"

mpirun -np 4 ./monte_pi 10000000



