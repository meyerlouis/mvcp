#!/bin/bash
#SBATCH --job-name=blog3d
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/house_%A_%a.out
#SBATCH --error=logs/house_%A_%a.err
#SBATCH --partition=hmem,compute,stats

# ── Environment ──
module load PyTorch/2.0-Miniconda3-4.12.0-Python-3.11.3
source /storage/maths/strdnj/env/bin/activate

# ── Setup ──
cd /storage/maths/strdnj/mvcp/realdata_experiments

# ── Decode array index into (seed, model) ──
# 60 jobs = 20 seeds × 3 models
#MODELS=(Ridge GBR MLP)
MODELS=(Ridge MLP)
N_MODELS=${#MODELS[@]}

SEED=$((SLURM_ARRAY_TASK_ID / N_MODELS))
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID % N_MODELS))]}

echo "=============================="
echo "Dataset: house"
echo "Seed:    $SEED"
echo "Model:   $MODEL"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=============================="

python run_blog_3d.py $SEED $MODEL

echo "Finished: $(date)"
