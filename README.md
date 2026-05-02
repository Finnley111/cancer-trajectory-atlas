# Cancer Trajectory Atlas

Morphology analysis and diffusion pseudotime for whole-slide histopathology images.

This pipeline constructs a diffusion-based pseudotime from H&E morphology features across a cohort of whole-slide images. It identifies malignancy-associated morphological trajectories—progression patterns captured by nuclear density, size ratios, and texture—and projects new slides into the learned trajectory space for diagnosis support and biomarker discovery.

## Quick Start

**Step 1: Prepare NDPIs**

Place your NDPI files in `data/MCF7_x5/`, then convert them to PNGs (keeping only the left half) and extract dimensions:

```bash
python run_all.py --convert
```

This creates cropped PNGs in `data/MCF7_x5_cropped/` and writes `slide_dimensions.json` for ratio-based coordinate conversion.

**Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 3: Run the full pipeline**

From the repository root:

```bash
python run_all.py --run
```

The pipeline will:
1. Stain-normalize all slides
2. Extract patches and embed features (Phikon 768-dim)
3. Reduce to 254-dim PCA space and cluster (Leiden, 19 clusters by default)
4. Build a diffusion map and compute pseudotime (DPT)
5. Validate by testing morphological feature correlation with pseudotime

All outputs (embeddings, clustering, trajectory plots, validation) go to `results/atlas_full/`.

## Command-Line Options

### Full Pipeline (`run_all.py`)

All paths and parameters are configurable via CLI arguments. If `paths.json` exists, its defaults will be used as fallbacks; otherwise, local relative paths are used.

```bash
# Convert NDPI → PNG only
python run_all.py --convert

# Run pipeline on existing PNGs
python run_all.py --run

# Convert then run
python run_all.py --convert --run

# Specify custom paths
python run_all.py --run \
  --ndpi-dir ~/data/ndpi \
  --png-dir ~/data/png \
  --annotation-dir ~/data/annotations \
  --output-dir ~/results

# Customize feature extraction
python run_all.py --run \
  --model phikon \
  --patch-size 112 \
  --stride 96

# Customize clustering
python run_all.py --run \
  --clustering-method leiden \
  --leiden-resolution 1.0

# Customize stain normalization and validation
python run_all.py --run \
  --stain-method macenko \
  --n-permutations 1000
```

**Available flags:**
- `--convert` — Convert NDPI files to cropped PNGs
- `--run` — Run the full analysis pipeline
- `--ndpi-dir PATH` — Input NDPI directory (default: from `paths.json` or local `data/MCF7_x5`)
- `--png-dir PATH` — Cropped PNG directory (default: from `paths.json` or local `data/MCF7_x5_cropped`)
- `--annotation-dir PATH` — Annotation directory (default: from `paths.json` or local `data/annotations`)
- `--output-dir PATH` — Output directory (default: from `paths.json` or local `results/atlas_full`)
- `--ndpi-level INT` — NDPI pyramid level to read (0=full res, higher=lower res; default: 0)
- `--ndpi-scale FLOAT` — Additional downscale factor (default: 0.5)
- `--model STR` — Feature model (`phikon`, `resnet50`; default: `phikon`)
- `--patch-size INT` — Patch size in pixels (default: 112)
- `--stride INT` — Stride between patches (default: 96)
- `--clustering-method STR` — Clustering algorithm (`leiden`, `hdbscan`, `kmeans`; default: `leiden`)
- `--leiden-resolution FLOAT` — Leiden resolution (higher=more clusters; default: 0.5)
- `--stain-method STR` — Stain normalization (`reinhard`, `macenko`, `none`; default: `reinhard`)
- `--n-permutations INT` — Permutations for validation (default: 1000)
- `--use-stardist` — Enable StarDist nuclear segmentation (slower, more accurate)

### Per-Slide Analysis (`run_individual.py`)

Run pseudotime independently on each slide:

```bash
# All slides
python -m cancer_trajectory_atlas.run_individual

# Specific slide (substring match)
python -m cancer_trajectory_atlas.run_individual --slide 6027-4L-2M-1

# Custom paths
python -m cancer_trajectory_atlas.run_individual \
  --png-dir ~/scratch/data/png \
  --annotation-dir ~/scratch/data/annotations \
  --output-dir ~/results/per_slide

# Adjust clustering resolution
python -m cancer_trajectory_atlas.run_individual --leiden-resolution 1.0

# Change stain normalization
python -m cancer_trajectory_atlas.run_individual --stain-method macenko
```

**Available flags:**
- `--slide STR` — Substring filter (only run slides matching this name)
- `--png-dir PATH` — PNG directory (default: from `paths.json` or local `data/MCF7_x5_cropped`)
- `--annotation-dir PATH` — Annotation directory (default: from `paths.json` or local `data/annotations`)
- `--output-dir PATH` — Output directory (default: from `paths.json` or local `individual_pseudotime_runs`)
- `--model STR` — Feature model (`phikon`, `resnet50`; default: `phikon`)
- `--patch-size INT` — Patch size (default: 112)
- `--stride INT` — Stride (default: 96)
- `--leiden-resolution FLOAT` — Leiden resolution (default: 0.5)
- `--stain-method STR` — Stain normalization (`reinhard`, `macenko`, `none`; default: `reinhard`)

### Single-Slide Analysis (`run_pipeline.py`)

Analyze a single slide with custom parameters:

```bash
python -m cancer_trajectory_atlas.run_pipeline \
  --image path/to/slide.png \
  --output results/ \
  --model phikon \
  --patch-size 112 \
  --stride 96 \
  --leiden-resolution 0.5
```

## Configuration File (Optional)

For convenience, you can create a `paths.json` file in the repository root to avoid specifying paths every run:

```json
{
    "raw_ndpi": "~/scratch/data/ndpi",
    "cropped_png": "~/scratch/data/MCF7_x5_cropped",
    "annotations": "~/scratch/data/annotations",
    "results": "~/projects/my-project/cancer-trajectory-atlas/results",
    "stain_reference": "~/scratch/data/MCF7_x5_cropped/reference.png"
}
```

The `~` is automatically expanded to your home directory. If `paths.json` doesn't exist, the pipeline uses local relative paths suitable for development.

## Project Structure

```
cancer_trajectory_atlas/
├── config.py                 # Default settings
├── run_pipeline.py           # Single-slide entry point
│
├── data/
│   └── stain_normalization.py   # Stain normalization
│
├── features/
│   ├── patching.py              # Patch extraction
│   └── extractors.py            # Feature extraction
│
├── analysis/
│   ├── clustering.py            # PCA, UMAP, and clustering
│   ├── diffusion.py             # Diffusion map and DPT
│   └── projector.py             # Train/test projection
│
├── validation/
│   ├── morphological_features.py  # Morphology features
│   ├── correlations.py            # Correlation and permutation tests
│   └── annotations.py            # Annotation loading
│
├── utils/
│   ├── viz.py                    # Plotting helpers
│   └── io.py                     # Save/load helpers
│
└── converters/
    ├── ndpi_to_img.py            # NDPI conversion
    └── tiff_to_img.py            # TIFF conversion
```