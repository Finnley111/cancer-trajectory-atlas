#!/bin/bash
# SLURM job script — Cancer Trajectory Atlas QC diagnostics (steps 1-4)
#
# Usage:
#   sbatch qc/submit_qc.sh atlas_full    reinhard
#   sbatch qc/submit_qc.sh atlas_macenko macenko
#
# Arguments:
#   $1 = run name (directory under $SCRATCH/results/)
#   $2 = stain method: reinhard | macenko | none  (default: reinhard)

#SBATCH --account=def-YOURACCOUNT        # <-- replace with your PI's account
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=atlas_qc
#SBATCH --output=%x_%j.log

set -euo pipefail

RUN_NAME=${1:-atlas_full}
STAIN_METHOD=${2:-reinhard}

SCRATCH=/scratch/$USER
PROJECT_DIR=$SCRATCH/cancer_trajectory_atlas
RUN_DIR=$SCRATCH/results/$RUN_NAME
SLIDES_DIR=$SCRATCH/data/MCF7_x5_cropped

echo "========================================"
echo "Atlas QC — $RUN_NAME  (stain: $STAIN_METHOD)"
echo "Run dir:    $RUN_DIR"
echo "Slides dir: $SLIDES_DIR"
echo "========================================"

# Activate the conda environment
module load python/3.10
conda activate atlas

# Run all four QC steps
python $PROJECT_DIR/qc/run_qc.py \
    --run-dir      "$RUN_DIR"     \
    --slides-dir   "$SLIDES_DIR"  \
    --stain-method "$STAIN_METHOD" \
    --steps        1234           \
    --patch-size   112            \
    --n-contact-patches 25

echo "Done. QC outputs in $RUN_DIR/qc/"
