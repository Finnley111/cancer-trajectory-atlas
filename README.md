# Cancer Trajectory Atlas

Morphology analysis and diffusion pseudotime for whole-slide histopathology images.

## Quick Workflow

1. Put the NDPI files in `data/MCF7_x5/`.
2. Crop them into PNGs with the conversion command below.
3. Run `run_all.py` on the cropped images.

The cropping step also writes `slide_dimensions.json`, which the annotation loading code uses for ratio-based coordinates.

### NDPI Conversion

Run this from the repository root after the NDPI files are in `data/MCF7_x5/`:

```bash
python run_all.py --convert
```

This creates left-half PNGs in `data/MCF7_x5_cropped/` and writes the matching `slide_dimensions.json` file.

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

## Quick Start

First make sure the raw NDPI slides are in `data/MCF7_x5/`, then crop them:

```bash
python run_all.py --convert
```

After that, run the full pipeline:

```bash
# Install dependencies
pip install torch transformers scanpy anndata umap-learn hdbscan \
    scikit-learn staintools stardist scikit-image scipy \
    matplotlib seaborn shapely staintools

# Run the full pipeline
python run_all.py --run
```