#!/bin/bash

#SBATCH --job-name=bh_on_cmaes
#SBATCH --output=logs/bh_on_cmaes_%j_out.txt
#SBATCH --error=logs/bh_on_cmaes_%j_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --time=24:00:00
#SBATCH -p workq
#SBATCH -A loni_pdrug

echo "Job started on $(hostname) at $(date)"

VENV_PYTHON="/work/jcarde7/Optimization/venv/bin/python3"
datasets=("cmaes_1" "cmaes_2" "cmaes_3")

optimizer_name="basin_hopping"
for dataset_name in "${datasets[@]}"
do
  for seed in {1..10}
  do
    srun --exclusive --ntasks=1 $VENV_PYTHON python/fit_synthetic.py --optimizer "$optimizer_name" --dataset "$dataset_name" --seed "$seed" &
  done
done

wait

echo "Job finished at $(date)"