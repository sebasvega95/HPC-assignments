#!/bin/bash
#
#SBATCH --job-name=matrixMul
#SBATCH --output=res_matrixMul.out
#SBATCH --ntasks=3
#SBATCH --nodes=3
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100
#SBATCH --gres=gpu:1

mpirun matrixMul.o

