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