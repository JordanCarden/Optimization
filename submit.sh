#!/bin/bash

#SBATCH --job-name=dual_optimizer_run
#SBATCH --output=logs/run_%j.txt
#SBATCH --error=logs/run_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=60
#SBATCH --time=24:00:00
#SBATCH -p workq
#SBATCH -A hpc_hpc_tyw_01

echo "Job started on $(hostname) at $(date)"

VENV_PYTHON="/work/jcarde7/Optimize2/venv/bin/python3"

echo "--- Launching CMA-ES tasks ---"
for dataset in 1 2 3
do
  for seed in {1..10}
  do
    srun --exclusive --ntasks=1 $VENV_PYTHON python/main.py --optimizer cmaes --dataset $dataset --seed $seed &
  done
done

echo "--- Launching LSHADE tasks ---"
for dataset in 1 2 3
do
  for seed in {1..10}
  do
    srun --exclusive --ntasks=1 $VENV_PYTHON python/main.py --optimizer lshade --dataset $dataset --seed $seed &
  done
done

echo "--- Launching Dual Annealing tasks ---"
for dataset in 1 2 3
do
  for seed in {1..10}
  do
    srun --exclusive --ntasks=1 $VENV_PYTHON python/main.py --optimizer dual_annealing --dataset $dataset --seed $seed &
  done
done

echo "--- Launching Basin-Hopping tasks ---"
for dataset in 1 2 3
do
  for seed in {1..10}
  do
    srun --exclusive --ntasks=1 $VENV_PYTHON python/main.py --optimizer basin_hopping --dataset $dataset --seed $seed &
  done
done

wait

echo "Job finished at $(date)"