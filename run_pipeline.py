#!/usr/bin/env python3
"""Single-slide analysis entry point for quick checks and prototyping.

Usage:
    python -m cancer_trajectory_atlas.run_pipeline --image data/slide.png --output results/
"""

import argparse
import os
import numpy as np
import pandas as pd

from . import config as defaults
from .features.patching import get_patches
from .features.extractors import extract_features
from .analysis.clustering import fit_pca, run_umap, cluster, get_cluster_centroids
from .analysis.diffusion import build_adata, compute_diffusion_map, compute_dpt
from .utils import viz


def main():
    parser = argparse.ArgumentParser(
        description="Cancer Trajectory Atlas — Single Slide Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Path to slide image")
    parser.add_argument("--output", default=defaults.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=defaults.DEFAULT_MODEL, choices=defaults.AVAILABLE_MODELS)
    parser.add_argument("--patch_size", type=int, default=defaults.DEFAULT_PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=defaults.DEFAULT_STRIDE)
    parser.add_argument("--clustering", default=defaults.DEFAULT_CLUSTERING_METHOD,
                        choices=["leiden", "hdbscan", "kmeans"])
    parser.add_argument("--leiden_resolution", type=float, default=defaults.DEFAULT_LEIDEN_RESOLUTION,
                        help="Leiden resolution (higher = more clusters)")
    parser.add_argument("--root_cluster", default="0", help="Cluster to use as DPT root")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Patch extraction
    print("\n=== Patch Extraction ===")
    patches, coords = get_patches(args.image, args.patch_size, args.stride)
    if len(patches) == 0:
        print("No patches found. Exiting.")
        return

    # Feature extraction
    print(f"\n=== Feature Extraction ({args.model}) ===")
    features = extract_features(patches, model_name=args.model)

    # Clustering
    print("\n=== Morphological Clustering ===")
    scaler, pca, X_pca = fit_pca(features)
    _, X_umap = run_umap(X_pca)
    labels = cluster(X_pca, method=args.clustering, resolution=args.leiden_resolution)

    # Diffusion pseudotime
    print("\n=== Diffusion Pseudotime ===")
    slide_ids = np.zeros(len(features), dtype=int)
    adata = build_adata(X_pca, labels, slide_ids, X_umap)
    compute_diffusion_map(adata)
    compute_dpt(adata, root_cluster=args.root_cluster)
    pseudotime = adata.obs["pseudotime"].values

    # Save results
    df = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "cluster": labels, "pseudotime": pseudotime,
    })
    df.to_csv(os.path.join(args.output, "results.csv"), index=False)

    # Plots
    if X_umap is not None:
        viz.plot_umap_clusters(X_umap, labels, os.path.join(args.output, "umap_clusters.png"))
        viz.plot_umap_pseudotime(X_umap, pseudotime, os.path.join(args.output, "umap_pseudotime.png"))

    print(f"\nDone! Results in {args.output}")


if __name__ == "__main__":
    main()
