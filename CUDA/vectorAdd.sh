#!/bin/bash
#
#SBATCH --job-name=vectorAdd
#SBATCH --output=res_vectorAdd.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100
#SBATCH --gres=gpu:1

echo $CUDA_VISIBLE_DEVICES
./vectorAdd.o

