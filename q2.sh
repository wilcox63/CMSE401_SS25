#!/bin/bash -login
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=1GB
#SBATCH --job-name=CMSE401

time ./quiz2
time ./q2_p


