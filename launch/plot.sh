#!/usr/bin/env bash
#SBATCH --gres=gpu:0
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --output=tmp/PLOT-%j.log

srun python plotter.py --path_to_ckpt runs/2022-07-05_08-03-25_DEBUG_HalfCheetah-v2_Gaussian__hyper_123456/checkpoints/sac_checkpoint_0