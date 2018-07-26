#!/bin/bash
GPU_COUNT=$(grep -Po "^[^\#].+gpus = \K([0-9]+)" config.py)

#SBATCH -p ai
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --gres=gpu:$GPU_COUNT
#SBATCH --mem=128G
echo $(hostname) $CUDA_VISIBLE_DEVICES $GPU_COUNT
#--exclusive
#srun -p ai -N 1 -c 12 --mem=128G --gres=gpu:$GPU_COUNT singularity exec /public/DL_Data/cnic_ai.img python train.py
singularity exec /public/DL_Data/cnic_ai.img python train.py
