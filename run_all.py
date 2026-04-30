#!/usr/bin/env python3
"""Convert NDPI slides and run the full atlas pipeline on all slides.

Usage:
    python run_all.py --convert
    python run_all.py --run
    python run_all.py --convert --run
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# Configuration

# Load paths from paths.json if it exists, otherwise use local relative paths (for development)
_SCRIPT_DIR = Path(__file__).parent
_PATHS_FILE = _SCRIPT_DIR / "paths.json"

if _PATHS_FILE.exists():
    with open(_PATHS_FILE) as f:
        _paths_config = json.load(f)
    NDPI_DIR = Path(_paths_config["raw_ndpi"]).expanduser()
    PNG_DIR = Path(_paths_config["cropped_png"]).expanduser()
    ANNOTATION_DIR = Path(_paths_config["annotations"]).expanduser()
    OUTPUT_DIR = Path(_paths_config["results"]).expanduser() / "atlas_full"
else:
    # Fallback to local relative paths (for development)
    NDPI_DIR = _SCRIPT_DIR / "data" / "MCF7_x5"
    PNG_DIR = _SCRIPT_DIR / "data" / "MCF7_x5_cropped"
    ANNOTATION_DIR = _SCRIPT_DIR / "data" / "annotations"
    OUTPUT_DIR = _SCRIPT_DIR.parent / "results" / "atlas_full"

# NDPI conversion settings
NDPI_LEVEL = 0          # Pyramid level (0 = full res, higher = lower res)
NDPI_SCALE = 0.5        # Additional downscale factor (0.5 = half size)

# Pipeline settings
MODEL = "phikon"
PATCH_SIZE = 112
STRIDE = 96
CLUSTERING_METHOD = "leiden"
LEIDEN_RESOLUTION = 0.5
STAIN_NORMALIZATION = "reinhard"    # "reinhard" or "macenko" or "none"
N_PERMUTATIONS = 1000
USE_STARDIST = False               # True = better nuclear seg, but slower

# Known NDPI dimensions at level 0.
# Used as a fallback when slide_dimensions.json is missing.
KNOWN_NDPI_DIMENSIONS = {
    "6027-4L-2M-1": (96000, 42240),
    "6027-4L-2M-2": (94080, 45056),
    "6027-4R-2M-1": (86400, 38016),
    "6027-4R-2M-2": (94080, 45056),
    "6028-4L-2M-1": (96000, 49280),
    "6028-4L-2M-2": (86400, 40832),
    "6028-4R-2M-1": (80640, 35200),
    "6028-4R-2M-2": (86400, 38016),
    "6029-4L-2M-1": (78720, 30976),
    "6029-4L-2M-2": (74880, 32384),
    "6029-4R-2M-1": (71040, 35200),
    "6029-4R-2M-2": (76800, 32384),
    "6031-4L-2M-1": (82560, 46464),
    "6031-4L-2M-2": (94080, 46464),
    "6031-4R-2M-1": (94080, 38016),
    "6031-4R-2M-2": (78720, 35200),
}


# NDPI to PNG conversion

def convert_ndpi_to_left_half_png():
    """
    Convert all NDPI files to PNG, keeping only the left half.
    Your NDPIs contain two copies of the same slide side by side —
    annotations were done on the left, so we discard the right.
    """
    ndpi_dir = Path(NDPI_DIR)
    out_dir = Path(PNG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import openslide
    except ImportError:
        print("ERROR: openslide-python is required for NDPI conversion.")
        print("  pip install openslide-python")
        print("  (Also need system lib: apt install libopenslide0 on Ubuntu)")
        sys.exit(1)

    ndpi_files = sorted(ndpi_dir.glob("*.ndpi"))
    if not ndpi_files:
        print(f"No .ndpi files found in {ndpi_dir}")
        sys.exit(1)

    print(f"\nFound {len(ndpi_files)} NDPI files in {ndpi_dir}")
    print(f"Output: {out_dir}")
    print(f"Level: {NDPI_LEVEL}, Scale: {NDPI_SCALE}")
    print("=" * 60)

    dimensions_log = {}

    for ndpi_path in ndpi_files:
        out_name = f"{ndpi_path.stem}_x5.png"
        out_path = out_dir / out_name

        # Read dimensions even if the PNG already exists.
        slide = openslide.OpenSlide(str(ndpi_path))
        dims = slide.level_dimensions[NDPI_LEVEL]

        if out_path.exists():
            print(f"\n  SKIP (exists): {out_name}")
            slide.close()
        else:
            print(f"\n  Converting: {ndpi_path.name}")
            print(f"    Full size: {dims[0]} x {dims[1]}")

            img = slide.read_region((0, 0), NDPI_LEVEL, dims).convert("RGB")
            slide.close()

            if NDPI_SCALE != 1.0:
                new_size = (int(dims[0] * NDPI_SCALE), int(dims[1] * NDPI_SCALE))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"    Scaled to: {new_size[0]} x {new_size[1]}")

            full_w, full_h = img.size
            img_left = img.crop((0, 0, full_w // 2, full_h))
            print(f"    Left half: {img_left.size[0]} x {img_left.size[1]}")

            img_left.save(str(out_path), "PNG", compress_level=6)
            print(f"    Saved: {out_name}")

        # Log dimensions for later ratio-to-pixel conversion.
        full_w = int(dims[0] * NDPI_SCALE) if NDPI_SCALE != 1.0 else dims[0]
        full_h = int(dims[1] * NDPI_SCALE) if NDPI_SCALE != 1.0 else dims[1]
        dimensions_log[out_name] = {
            "original_full_width": full_w,
            "original_full_height": full_h,
            "cropped_width": full_w // 2,
            "cropped_height": full_h,
        }

    # Save dimensions sidecar
    dims_path = out_dir / "slide_dimensions.json"
    with open(dims_path, "w") as f:
        json.dump(dimensions_log, f, indent=2)
    print(f"\n  Dimensions log: {dims_path}")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"PNGs in: {out_dir}")


# Pipeline setup

def _get_known_dimensions(png_name):
    """Look up original NDPI dimensions from the fallback table."""
    stem = Path(png_name).stem              # "6027-4L-2M-1_x5"
    base_stem = stem.replace("_x5", "")     # "6027-4L-2M-1"

    dims = KNOWN_NDPI_DIMENSIONS.get(base_stem)
    if dims is not None:
        full_w, full_h = dims
        # Apply the configured downscale.
        if NDPI_SCALE != 1.0:
            full_w = int(full_w * NDPI_SCALE)
            full_h = int(full_h * NDPI_SCALE)
        return full_w, full_h
    return None, None


def discover_slides():
    """Discover slides, match annotations, and load original dimensions."""
    png_dir = Path(PNG_DIR)
    ann_dir = Path(ANNOTATION_DIR)

    if not png_dir.exists():
        print(f"ERROR: PNG directory not found: {png_dir}")
        print("  Run with --convert first, or create PNGs manually.")
        sys.exit(1)

    png_files = sorted(png_dir.glob("*.png"))
    if not png_files:
        print(f"ERROR: No PNG files found in {png_dir}")
        sys.exit(1)

    # Try loading the sidecar first.
    dims_path = png_dir / "slide_dimensions.json"
    dims_log = {}
    if dims_path.exists():
        with open(dims_path) as f:
            dims_log = json.load(f)
        print(f"  Loaded slide_dimensions.json ({len(dims_log)} entries)")
    else:
        print(f"  No slide_dimensions.json found — using hardcoded NDPI dimensions")

    slides = []
    for png_path in png_files:
        slide_entry = {"image": str(png_path)}

        # Match annotation file.
        stem = png_path.stem                    # e.g. "6027-4L-2M-1_x5"
        base_stem = stem.replace("_x5", "")     # e.g. "6027-4L-2M-1"

        ann_path = None
        for candidate in [
            ann_dir / f"{stem}.json",
            ann_dir / f"{base_stem}.json",
            ann_dir / f"{stem}.geojson",
            ann_dir / f"{base_stem}.geojson",
        ]:
            if candidate.exists():
                ann_path = str(candidate)
                break

        slide_entry["annotation"] = ann_path

        # Look up original dimensions: sidecar first, then the fallback table.
        sidecar_dims = dims_log.get(png_path.name, {})
        if sidecar_dims:
            slide_entry["original_full_width"] = sidecar_dims["original_full_width"]
            slide_entry["original_full_height"] = sidecar_dims["original_full_height"]
        else:
            fw, fh = _get_known_dimensions(png_path.name)
            slide_entry["original_full_width"] = fw
            slide_entry["original_full_height"] = fh

        slides.append(slide_entry)

    # Report discovered slides.
    n_ann = sum(1 for s in slides if s["annotation"] is not None)
    n_dims = sum(1 for s in slides if s["original_full_width"] is not None)
    print(f"\nDiscovered {len(slides)} slides "
          f"({n_ann} with annotations, {n_dims} with original dims)")
    for s in slides:
        ann_str = "✓ ann" if s["annotation"] else "✗ ann"
        dim_str = (f"full={s['original_full_width']}x{s['original_full_height']}"
                   if s["original_full_width"] else "NO DIMS")
        print(f"  {ann_str}  {dim_str}  {Path(s['image']).name}")

    # Warn if annotated slides are missing dimensions.
    missing = [s for s in slides
               if s["annotation"] is not None and s["original_full_width"] is None]
    if missing:
        print(f"\n  WARNING: {len(missing)} annotated slides have no original dimensions!")
        print(f"  Ratio annotation coordinates will be INCORRECT for these slides.")
        print(f"  Fix: run --convert, or add entries to KNOWN_NDPI_DIMENSIONS.")

    return slides


def run_pipeline():
    """Run the full pipeline on all discovered slides."""
    slides = discover_slides()

    if len(slides) == 0:
        return

    output_dir = Path(OUTPUT_DIR)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Imports (relative to this module)
    from .data.stain_normalization import build_normalizer, normalize_slide
    from .features.patching import get_patches_from_array, load_roi_polygons
    from .features.extractors import extract_features
    from .analysis.clustering import (
        fit_pca, run_umap, cluster, check_slide_independence, get_cluster_centroids,
    )
    from .analysis.diffusion import (
        build_adata, compute_diffusion_map, compute_dpt,
    )
    from .validation.morphological_features import compute_morphological_features
    from .validation.correlations import run_full_validation
    from . import utils
    from .utils import viz, io

    # Stain normalization
    print(f"\n{'='*60}")
    print("PHASE 1: Stain Normalization")
    print(f"{'='*60}")

    reference_image = slides[0]["image"]
    stain_normalizer = build_normalizer(STAIN_NORMALIZATION, reference_image)

    if stain_normalizer is not None:
        import shutil
        shutil.copy2(reference_image, output_dir / "stain_reference.png")

    # Patch extraction and features
    print(f"\n{'='*60}")
    print("PHASE 2: Patch Extraction & Feature Embedding")
    print(f"{'='*60}")

    all_patches_list = []
    all_coords_list = []
    all_slide_ids = []
    slide_names = []

    for i, slide_cfg in enumerate(slides):
        image_path = slide_cfg["image"]
        slide_name = Path(image_path).stem
        slide_names.append(slide_name)
 
        print(f"\n  [{i+1}/{len(slides)}] {slide_name}")
 
        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img)
        print(f"    Image size: {img_arr.shape[1]} x {img_arr.shape[0]}")
 
        # Stain normalize
        img_arr = normalize_slide(img_arr, stain_normalizer, slide_name)
 
        # Load ROI polygons if annotations exist.
        roi_polys = None
        ann_path = slide_cfg.get("annotation")
        if ann_path is not None:
            cropped_w, cropped_h = img_arr.shape[1], img_arr.shape[0]
            roi_polys = load_roi_polygons(
                ann_path,
                coordinate_space="ratio",
                original_full_width=slide_cfg.get("original_full_width"),
                original_full_height=slide_cfg.get("original_full_height"),
                cropped_w=cropped_w,
                cropped_h=cropped_h,
            )
            print(f"    Loaded {len(roi_polys)} ROI hotspot polygons")
        else:
            print(f"    No annotation — using full slide")
 
        # Extract patches, filtering to ROIs when available.
        patches, coords = get_patches_from_array(
            img_arr,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            image_name=slide_name,
            roi_polygons=roi_polys,
        )
 
        if len(patches) == 0:
            print(f"    WARNING: No patches found — skipping")
            continue
 
        all_patches_list.append(patches)
        all_coords_list.append(coords)
        all_slide_ids.extend([i] * len(patches))

    if not all_patches_list:
        print("ERROR: No patches extracted from any slide!")
        return

    all_patches = np.concatenate(all_patches_list)
    all_coords = np.concatenate(all_coords_list)
    slide_ids = np.array(all_slide_ids)

    print(f"\n  Total: {len(all_patches)} patches from {len(slide_names)} slides")

    # Feature extraction
    print(f"\n  Extracting {MODEL} features...")
    features = extract_features(all_patches, model_name=MODEL)
    print(f"  Feature shape: {features.shape}")

    # Clustering
    print(f"\n{'='*60}")
    print("PHASE 3: Morphological Clustering")
    print(f"{'='*60}")

    scaler, pca, X_pca = fit_pca(features, variance_target=0.95)
    umap_reducer, X_umap = run_umap(X_pca)

    cluster_labels = cluster(
        X_pca,
        method=CLUSTERING_METHOD,
        resolution=LEIDEN_RESOLUTION,
    )

    # Validation checks
    slide_check = check_slide_independence(cluster_labels, slide_ids)
    centroids = get_cluster_centroids(X_pca, cluster_labels)

    # Clustering figures
    if X_umap is not None:
        viz.plot_umap_clusters(X_umap, cluster_labels, fig_dir / "fig1_umap_clusters.png")
        viz.plot_umap_by_slide(X_umap, slide_ids, fig_dir / "qc_umap_by_slide.png")
    viz.plot_spatial_clusters(all_coords, cluster_labels, slide_ids, fig_dir)

    if len(centroids) > 0:
        viz.plot_cluster_patch_grid(
            all_patches, cluster_labels, centroids,
            fig_dir / "fig2_cluster_patches.png",
        )

    # Diffusion pseudotime
    print(f"\n{'='*60}")
    print("PHASE 4: Diffusion Pseudotime")
    print(f"{'='*60}")

    # Auto root: pick the first valid cluster and override it after inspection.
    valid_clusters = sorted([c for c in set(cluster_labels) if c != -1])
    root_cluster = str(valid_clusters[0])
    print(f"  Auto root cluster: {root_cluster}")
    print(f"  IMPORTANT: Inspect fig2_cluster_patches.png after this run.")
    print(f"  Set ROOT_CLUSTER to the most organized cluster, then re-run.\n")

    adata = build_adata(X_pca, cluster_labels, slide_ids, X_umap)
    compute_diffusion_map(adata, n_neighbors=30, n_comps=10)
    compute_dpt(adata, root_cluster=root_cluster)

    pseudotime = adata.obs["pseudotime"].values

    # Diffusion figures
    if X_umap is not None:
        viz.plot_umap_pseudotime(X_umap, pseudotime, fig_dir / "fig4_umap_pseudotime.png")
    viz.plot_pseudotime_violins(pseudotime, cluster_labels, fig_dir / "fig5_pt_violins.png")
    viz.plot_spatial_pseudotime(all_coords, pseudotime, slide_ids, fig_dir)

    if "X_diffmap" in adata.obsm and adata.obsm["X_diffmap"].shape[1] >= 3:
        viz.plot_3d_manifold(
            adata.obsm["X_diffmap"], pseudotime,
            fig_dir / "diffusion_3d.png",
        )

    # Validation
    print(f"\n{'='*60}")
    print("PHASE 5: Morphological Feature Validation")
    print(f"{'='*60}")

    morph_features = compute_morphological_features(
        all_patches, use_stardist=USE_STARDIST,
    )

    for name, values in morph_features.items():
        adata.obs[name] = values

    validation = run_full_validation(
        pseudotime, morph_features, cluster_labels, all_coords,
        n_permutations=N_PERMUTATIONS,
    )

    viz.plot_feature_vs_pseudotime(
        pseudotime, morph_features,
        validation["feature_correlations"],
        fig_dir / "fig6_features_vs_pt.png",
    )
    viz.plot_permutation_nulls(
        validation["permutation_tests"],
        fig_dir / "fig7_permutation_nulls.png",
    )

    # Save everything
    print(f"\n{'='*60}")
    print("Saving Results")
    print(f"{'='*60}")

    import pandas as pd

    # AnnData (the full atlas)
    adata.write(output_dir / "adata_full.h5ad")
    print(f"  AnnData: {output_dir / 'adata_full.h5ad'}")

    # CSV results
    df = pd.DataFrame({
        "x": all_coords[:, 0],
        "y": all_coords[:, 1],
        "slide_id": slide_ids,
        "slide_name": [slide_names[sid] for sid in slide_ids],
        "cluster": cluster_labels,
        "pseudotime": pseudotime,
    })
    for name, values in morph_features.items():
        df[name] = values
    df.to_csv(output_dir / "results.csv", index=False)
    print(f"  CSV: {output_dir / 'results.csv'}")

    # Validation JSON
    io.save_json(validation, output_dir / "validation.json")
    print(f"  Validation: {output_dir / 'validation.json'}")

    # Save fitted models for future projection
    io.save_pickle(scaler, output_dir / "scaler.pkl")
    io.save_pickle(pca, output_dir / "pca.pkl")
    io.save_pickle(umap_reducer, output_dir / "umap_reducer.pkl")

    # Slide independence report
    io.save_json(slide_check, output_dir / "slide_independence.json")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE! ({elapsed / 60:.1f} minutes)")
    print(f"{'='*60}")
    print(f"  Results:    {output_dir}")
    print(f"  Figures:    {fig_dir}")
    print(f"  Patches:    {len(all_patches)}")
    print(f"  Clusters:   {len(set(cluster_labels) - {-1})}")
    print(f"")
    print(f"  VERDICT: {validation['summary']['verdict']}")
    print(f"")
    print(f"  NEXT STEPS:")
    print(f"  1. Open {fig_dir / 'fig2_cluster_patches.png'}")
    print(f"     Identify the most organized/regular morphology cluster.")
    print(f"  2. Edit ROOT_CLUSTER at the top of this script.")
    print(f"  3. Re-run to get biologically anchored pseudotime.")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cancer Trajectory Atlas — Full Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --convert          # Convert NDPI → left-half PNG
  python run_all.py --run              # Run pipeline on existing PNGs
  python run_all.py --convert --run    # Convert then run
        """,
    )
    parser.add_argument("--convert", action="store_true",
                        help="Convert NDPI files to left-half PNGs")
    parser.add_argument("--run", action="store_true",
                        help="Run the analysis pipeline")

    args = parser.parse_args()

    if not args.convert and not args.run:
        parser.print_help()
        print("\n  Specify --convert, --run, or both.")
        sys.exit(0)

    if args.convert:
        convert_ndpi_to_left_half_png()

    if args.run:
        run_pipeline()