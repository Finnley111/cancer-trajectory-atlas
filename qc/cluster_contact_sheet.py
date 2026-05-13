"""
QC Step 3 — Cluster contact sheets.

For each cluster, samples N random patches from the original PNG slides and
arranges them in a 5×5 grid.  Patches are shown pre-normalization (the PNG
files are the originals).  This reveals whether a cluster is driven by real
morphology or by artifacts (background, fat tissue, slide edges, etc.).

Note: patches nearest the cluster centroid are already shown in the main
pipeline's fig2_cluster_patches.png.  This module shows a *random* sample
to expose the full spread of each cluster.

Usage (from the project root):
    python cancer_trajectory_atlas/qc/cluster_contact_sheet.py \\
        --results-csv /scratch/finnley1/results/atlas_full/results.csv \\
        --slides-dir  /scratch/finnley1/data/MCF7_x5_cropped \\
        --output-dir  /scratch/finnley1/results/atlas_full/qc
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def make_contact_sheets(
    results_csv: str,
    slides_dir: str,
    output_dir: str,
    n_patches: int = 25,
    patch_size: int = 112,
    seed: int = 42,
) -> None:
    """
    Save a contact-sheet figure per cluster showing random patches.

    Args:
        results_csv:  Path to results.csv produced by run_all.py.
        slides_dir:   Directory containing the original .png slides.
        output_dir:   Root QC directory; sheets go in output_dir/contact_sheets/.
        n_patches:    Number of random patches per cluster (default 25 → 5×5 grid).
        patch_size:   Patch side length in pixels (default 112).
        seed:         Random seed for reproducibility.
    """
    import pandas as pd
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    rng = np.random.RandomState(seed)

    df = pd.read_csv(results_csv)
    required = {"x", "y", "slide_name", "cluster"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: results.csv is missing columns: {missing}")
        return

    slides_dir = Path(slides_dir)
    out = Path(output_dir) / "contact_sheets"
    out.mkdir(parents=True, exist_ok=True)

    clusters = sorted(df["cluster"].unique())
    valid_clusters = [c for c in clusters if c != -1]
    print(f"Found {len(valid_clusters)} clusters, {len(df)} patches total.")

    for cluster_id in valid_clusters:
        cluster_df = df[df["cluster"] == cluster_id].reset_index(drop=True)
        n_avail = len(cluster_df)

        # Sample patches (or take all if fewer than n_patches)
        n_sample = min(n_patches, n_avail)
        idx = rng.choice(n_avail, size=n_sample, replace=False)
        sampled = cluster_df.iloc[idx]

        # Group by slide to load each PNG only once
        patches_by_slide = {}
        for _, row in sampled.iterrows():
            sname = row["slide_name"]
            if sname not in patches_by_slide:
                patches_by_slide[sname] = []
            patches_by_slide[sname].append((int(row["x"]), int(row["y"])))

        # Extract patches slide-by-slide
        all_patches = []
        for slide_name, coords in patches_by_slide.items():
            png_path = slides_dir / f"{slide_name}.png"
            if not png_path.exists():
                print(f"  WARNING: slide not found: {png_path}")
                for _ in coords:
                    all_patches.append(np.full((patch_size, patch_size, 3), 200, dtype=np.uint8))
                continue

            img = np.array(Image.open(png_path).convert("RGB"))
            h, w = img.shape[:2]
            for x, y in coords:
                x2, y2 = x + patch_size, y + patch_size
                if x2 > w or y2 > h:
                    all_patches.append(np.full((patch_size, patch_size, 3), 200, dtype=np.uint8))
                else:
                    all_patches.append(img[y:y2, x:x2])

        if not all_patches:
            print(f"  Cluster {cluster_id}: no patches could be loaded.")
            continue

        # Arrange in a grid (5 columns or fewer if less than 25 patches)
        cols = min(5, len(all_patches))
        rows = int(np.ceil(len(all_patches) / cols))
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 1.5, rows * 1.5 + 0.6))

        # Flatten axes to a 1-D list regardless of shape
        if rows == 1 and cols == 1:
            axes_flat = [axes]
        elif rows == 1 or cols == 1:
            axes_flat = list(np.array(axes).ravel())
        else:
            axes_flat = list(axes.ravel())

        for i, ax in enumerate(axes_flat):
            if i < len(all_patches):
                ax.imshow(all_patches[i])
            ax.axis("off")

        fig.suptitle(
            f"Cluster {cluster_id}  —  {n_sample} random patches"
            f" / {n_avail} total  (pre-normalization)",
            fontsize=10,
        )
        plt.tight_layout()
        save_path = out / f"cluster_{cluster_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Cluster {cluster_id}: {n_sample}/{n_avail} patches → {save_path.name}")

    print(f"\n  Contact sheets saved to: {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="QC Step 3: cluster contact sheets"
    )
    parser.add_argument("--results-csv", required=True, help="Path to results.csv")
    parser.add_argument("--slides-dir",  required=True, help="Directory of PNG slides")
    parser.add_argument("--output-dir",  required=True, help="Root QC output directory")
    parser.add_argument("--n-patches",   type=int, default=25,
                        help="Random patches per cluster (default 25)")
    parser.add_argument("--patch-size",  type=int, default=112,
                        help="Patch side length in pixels (default 112)")
    args = parser.parse_args()
    make_contact_sheets(
        args.results_csv, args.slides_dir, args.output_dir,
        args.n_patches, args.patch_size,
    )
