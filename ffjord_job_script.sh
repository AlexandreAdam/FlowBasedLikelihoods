#!/bin/bash
#SBATCH --account=def-lplevass
#SBATCH --mem=8G
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --time=0-00:05            # time (DD-HH:MM)

source activate ../../fbl/bin/activate
python train_ffjord.py --niters 5000
