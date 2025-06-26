#!/bin/bash

#SBATCH --job-name=cmaes_single_node
#SBATCH --output=logs/run_%j.out      # Single output file for the entire job
#SBATCH --error=logs/run_%j.err       # Single error file
#SBATCH --nodes=1                     # Request exactly one node
#SBATCH --ntasks=30                   # Request 30 tasks (cores) on that node
#SBATCH --time=24:00:00               # Adjust time as needed
#SBATCH -p workq
#SBATCH -A hpc_hpc_tyw_01

# --- Environment Setup ---
echo "Job started on $(hostname) at $(date)"
module load anaconda3/2023.09-0

# --- Launch 30 Parallel Tasks ---
# Use a loop to launch all 30 python scripts in the background
for dataset in 1 2 3
do
  for seed in {1..10}
  do
    # The "srun" command launches one task on one of the cores we requested.
    # The "&" at the end runs it in the background so the loop can continue immediately.
    srun --exclusive --ntasks=1 python3 python/main.py --optimizer cmaes --dataset $dataset --seed $seed &
  done
done

# --- Wait for all background tasks to finish ---
# The "wait" command is crucial. It tells the script to wait here until all
# of the background jobs launched above have completed.
wait

echo "Job finished at $(date)"