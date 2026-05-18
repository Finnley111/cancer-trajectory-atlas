#!/bin/bash
# Per-task LOO runner. Called by submit_loo_array.sh via SLURM array.
# Do NOT submit this directly — use submit_loo_array.sh.
#
# Expects these environment variables set by the array job:
#   HELD_OUT   — stem of the held-out slide (e.g. 6027-4L-2M-1_x5)
#   FULL_RUN   — path to the full 16-slide reference run (for in-manifold PT)
#   CACHE_DIR  — path to feature cache directory

set -euo pipefail

if [[ -z "${HELD_OUT:-}" ]]; then
    echo "ERROR: HELD_OUT is not set"
    exit 1
fi

LOO_OUT="$SCRATCH/results/loo_${HELD_OUT}"
mkdir -p "$LOO_OUT"

echo "=== LOO run: held-out = $HELD_OUT ==="
echo "Output: $LOO_OUT"
echo "Reference run: ${FULL_RUN:-$SCRATCH/results/atlas_none_harmony}"
echo ""

# Build comma-separated list of all 15 training slides (exclude held-out)
SLIDE_LIST="$HOME/cancer_trajectory_atlas/data/loo_slides.txt"
if [[ ! -f "$SLIDE_LIST" ]]; then
    echo "ERROR: loo_slides.txt not found at $SLIDE_LIST"
    exit 1
fi

TRAINING_SLIDES=$(grep -v "^${HELD_OUT}$" "$SLIDE_LIST" | paste -sd,)
echo "Training slides (15): $TRAINING_SLIDES"
echo ""

cd ~

# Phase A — run pipeline on 15 training slides (features loaded from cache)
python -m cancer_trajectory_atlas.run_all \
    --run \
    --png-dir             $SCRATCH/data/MCF7_x5_cropped \
    --annotation-dir      $SCRATCH/data/annotations \
    --output-dir          "$LOO_OUT" \
    --stain-method        none \
    --model               phikon \
    --patch-size          112 \
    --stride              96 \
    --clustering-method   leiden \
    --leiden-resolution   0.5 \
    --harmony \
    --harmony-key         section_number \
    --n-permutations      200 \
    --slides              "$TRAINING_SLIDES" \
    --features-cache-dir  "${CACHE_DIR:-$SCRATCH/data/features_cache}"

echo ""
echo "=== Phase B: projecting held-out slide ==="

# Phase B — project held-out slide onto the LOO manifold
python -m cancer_trajectory_atlas.analysis.loo_project \
    --projector-dir  "$LOO_OUT/projector" \
    --held-out-slide "$HELD_OUT" \
    --cache-dir      "${CACHE_DIR:-$SCRATCH/data/features_cache}" \
    --full-run-dir   "${FULL_RUN:-$SCRATCH/results/atlas_none_harmony}" \
    --output-dir     "$LOO_OUT"

echo ""
echo "=== Done: $HELD_OUT ==="
echo "Results in: $LOO_OUT"
