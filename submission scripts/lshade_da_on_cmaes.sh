#!/bin/bash

#SBATCH --job-name=lshade_da_on_cmaes
#SBATCH --output=logs/lshade_da_on_cmaes_%j_out.txt
#SBATCH --error=logs/lshade_da_on_cmaes_%j_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks=60
#SBATCH --time=24:00:00
#SBATCH -p workq
#SBATCH -A loni_pdrug

echo "Job started on $(hostname) at $(date)"

VENV_PYTHON="/work/jcarde7/Optimization/venv/bin/python3"
datasets=("cmaes_1" "cmaes_2" "cmaes_3")

optimizer_name_1="lshade"
for dataset_name in "${datasets[@]}"
do
  for seed in {1..10}
  do
    srun --exclusive --ntasks=1 $VENV_PYTHON python/fit_synthetic.py --optimizer "$optimizer_name_1" --dataset "$dataset_name" --seed "$seed" &
  done
done

optimizer_name_2="dual_annealing"
for dataset_name in "${datasets[@]}"
do
  for seed in {1..10}
  do
    srun --exclusive --ntasks=1 $VENV_PYTHON python/fit_synthetic.py --optimizer "$optimizer_name_2" --dataset "$dataset_name" --seed "$seed" &
  done
done

wait

echo "Job finished at $(date)"