#!/bin/bash

#SBATCH --job-name=dual_annealing_bh_da
#SBATCH --output=logs/dual_annealing_bh_da_%j_out.txt
#SBATCH --error=logs/dual_annealing_bh_da_%j_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks=60
#SBATCH --time=24:00:00
#SBATCH -p workq
#SBATCH -A loni_pdrug

echo "Job started on $(hostname) at $(date)"

VENV_PYTHON="/work/jcarde7/Optimization/venv/bin/python3"

datasets=("basin_hopping_1" "basin_hopping_2" "basin_hopping_3" "dual_annealing_1" "dual_annealing_2" "dual_annealing_3")

optimizer_name="dual_annealing"

for dataset_name in "${datasets[@]}"
do
  for seed in {1..10}
  do
    srun --exclusive --ntasks=1 $VENV_PYTHON python/fit_synthetic.py --optimizer "$optimizer_name" --dataset "$dataset_name" --seed "$seed" &
  done
done

wait

echo "Job finished at $(date)"