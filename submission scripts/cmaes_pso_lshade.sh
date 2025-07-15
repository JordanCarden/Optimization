#!/bin/bash

#SBATCH --job-name=cmaes_pso_lshade
#SBATCH --output=logs/cmaes_pso_lshade_%j_out.txt
#SBATCH --error=logs/cmaes_pso_lshade_%j_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks=60
#SBATCH --time=24:00:00
#SBATCH -p workq
#SBATCH -A loni_pdrug

echo "Job started on $(hostname) at $(date)"

VENV_PYTHON="/work/jcarde7/Optimization/venv/bin/python3"

datasets=("pso_1" "pso_2" "pso_3" "lshade_1" "lshade_2" "lshade_3")

optimizer_name="cmaes"

for dataset_name in "${datasets[@]}"
do
  for seed in {1..10}
  do
    srun --exclusive --ntasks=1 $VENV_PYTHON python/fit_synthetic.py --optimizer "$optimizer_name" --dataset "$dataset_name" --seed "$seed" &
  done
done

wait

echo "Job finished at $(date)"