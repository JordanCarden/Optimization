#!/bin/bash

#SBATCH --job-name=experimental_fit_run
#SBATCH --output=logs/exp_fit_%j.txt
#SBATCH --error=logs/exp_fit_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=60
#SBATCH --time=24:00:00
#SBATCH -p workq
#SBATCH -A hpc_hpcsuvo02

VENV_PYTHON="/work/jcarde7/Optimization/venv/bin/python3"

OPTIMIZERS=("cmaes" "lshade" "pso" "dual_annealing" "basin_hopping")

for optimizer in "${OPTIMIZERS[@]}"
do
  for seed in {2..13}
  do
    srun --exclusive --ntasks=1 $VENV_PYTHON python/fit_experimental.py --optimizer $optimizer --seed $seed &
  done
done

wait