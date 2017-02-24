#!/bin/bash

#SBATCH --job-name=matrixMul
#SBATCH --output=res_matrixMul.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./matrixMul.out
