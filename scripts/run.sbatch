#!/bin/bash
#SBATCH --job-name=prlb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/%x_%j.log
#
# Generic Slurm wrapper.  Submit from the repository root and pass the account and
# partition of your cluster (here held in $ACCOUNT and $PARTITION), then the command:
#   sbatch --account="$ACCOUNT" --partition="$PARTITION" scripts/run.sbatch \
#          python experiments/cifar10c/shift_pairs.py clip
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLBACKEND=Agg
export PYTHONHASHSEED=0
"$@"
