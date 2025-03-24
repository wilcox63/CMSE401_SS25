#!/bin/bash --login

#SBATCH --job-name Quiz3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=1GB
#SBATCH --time=0-00:05:00
#SBATCH --gpus-per-node=v100:1

module load CUDA

nvcc -o avg_CUDA moving_avg_CUDA.cu

echo srun time ./avg_CUDA
