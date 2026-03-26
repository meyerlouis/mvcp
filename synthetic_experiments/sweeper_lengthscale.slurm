#!/bin/bash
#SBATCH --job-name=kcp_sweep
#SBATCH --array=0-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/seed_%A_%a.out
#SBATCH --error=logs/seed_%A_%a.err
#SBATCH --partition=hmem,compute,stats


# ── Environment ──
module load PyTorch/2.0-Miniconda3-4.12.0-Python-3.11.3
source /storage/maths/strdnj/env/bin/activate

# ── Setup ──
cd /storage/maths/strdnj/kernel_cp/experiment_sandbox2
mkdir -p logs results_lengthscale

echo "=============================="
echo "Seed:  $SLURM_ARRAY_TASK_ID"
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "=============================="

# Each array task = one seed. That's it.
python run_one_seed_lengthscale.py $SLURM_ARRAY_TASK_ID

echo "Finished: $(date)"
