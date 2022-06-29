#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c10
#SBATCH --output=tmp/APL-%j.log

srun python main.py --hyper --wandb --wandb-tag benchmark --seed 111

# srun python main.py --hyper --wandb --wandb-tag benchmark --seed 222

# srun python main.py --hyper --wandb --wandb-tag benchmark --seed 333

# srun python main.py --hyper --wandb --wandb-tag benchmark --seed 444

# srun python main.py --hyper --wandb --wandb-tag benchmark --seed 555