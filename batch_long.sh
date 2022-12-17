#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=8G
#SBATCH --time=1:00:00

source ~/venv_py38/bin/activate
python ebm_dsm.py $@
