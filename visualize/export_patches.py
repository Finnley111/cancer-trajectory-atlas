"""Export representative patches organized by pseudotime range.

Crops patches directly from the slide PNGs (no re-extraction) using the
(x, y) coordinates stored in results.csv.  Patches are stratified into
low / mid / high pseudotime bins and a fixed number are sampled from each.

Usage:
    python -m cancer_trajectory_atlas.visualize.export_patches \\
        --results-csv  $SCRATCH/results/atlas_none_harmony/results.csv \\
        --png-dir      $SCRATCH/data/MCF7_x5_cropped \\
        --output-dir   $SCRATCH/results/patch_export \\
        --patch-size   112 \\
        --n-per-bin    50

Output structure:
    output_dir/
        low_pseudotime/      (PT 0.00 – 0.33)
            {slide}__pt{:.4f}__x{x}__y{y}.png
            ...
        mid_pseudotime/      (PT 0.33 – 0.67)
        high_pseudotime/     (PT 0.67 – 1.00)
        manifest.csv         (slide, bin, x, y, pseudotime, filename)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_DEFAULT_BINS = [
    ("low_pseudotime",  0.00, 0.33),
    ("mid_pseudotime",  0.33, 0.67),
    ("high_pseudotime", 0.67, 1.001),
]


def export_patches(
    df: pd.DataFrame,
    png_dir: Path,
    output_dir: Path,
    patch_size: int = 112,
    n_per_bin: int = 50,
    bins=None,
    rng: np.random.Generator = None,
):
    """Main export logic.  Returns a manifest DataFrame."""
    if bins is None:
        bins = _DEFAULT_BINS
    if rng is None:
        rng = np.random.default_rng(42)

    for bin_name, _, _ in bins:
        (output_dir / bin_name).mkdir(parents=True, exist_ok=True)

    slide_names = sorted(df["slide_name"].unique())
    manifest_rows = []

    for slide_name in slide_names:
        png_path = png_dir / f"{slide_name}.png"
        if not png_path.exists():
            print(f"  SKIP {slide_name}: PNG not found at {png_path}")
            continue

        slide_df = df[df["slide_name"] == slide_name].reset_index(drop=True)
        print(f"  {slide_name}: {len(slide_df)} patches")

        img = np.array(Image.open(png_path).convert("RGB"))
        img_h, img_w = img.shape[:2]

        for bin_name, lo, hi in bins:
            mask = (slide_df["pseudotime"] >= lo) & (slide_df["pseudotime"] < hi)
            pool = slide_df[mask].reset_index(drop=True)

            if len(pool) == 0:
                print(f"    {bin_name}: no patches in range [{lo:.2f}, {hi:.2f})")
                continue

            n_sample = min(n_per_bin, len(pool))
            chosen = pool.sample(n=n_sample, random_state=int(rng.integers(1 << 31)))

            for _, row in chosen.iterrows():
                x, y = int(row["x"]), int(row["y"])
                pt = float(row["pseudotime"])

                # Guard against out-of-bounds patches at image edges.
                x_end = min(x + patch_size, img_w)
                y_end = min(y + patch_size, img_h)
                patch = img[y:y_end, x:x_end]

                if patch.shape[0] < 4 or patch.shape[1] < 4:
                    continue

                fname = f"{slide_name}__pt{pt:.4f}__x{x}__y{y}.png"
                out_path = output_dir / bin_name / fname
                Image.fromarray(patch).save(str(out_path))

                manifest_rows.append({
                    "slide_name":  slide_name,
                    "bin":         bin_name,
                    "x":           x,
                    "y":           y,
                    "pseudotime":  pt,
                    "filename":    str(Path(bin_name) / fname),
                })

            print(f"    {bin_name}: saved {n_sample} patches")

    return pd.DataFrame(manifest_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Export pseudotime-stratified patch crops from slide PNGs."
    )
    parser.add_argument("--results-csv", type=Path, required=True,
                        help="results.csv from a pipeline run")
    parser.add_argument("--png-dir", type=Path, required=True,
                        help="Directory containing slide PNG files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination for patch folders and manifest.csv")
    parser.add_argument("--patch-size", type=int, default=112,
                        help="Patch size used during extraction (pixels)")
    parser.add_argument("--n-per-bin", type=int, default=50,
                        help="Number of patches to sample from each pseudotime bin")
    parser.add_argument("--slides", nargs="*",
                        help="Process only these slide stems (default: all)")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    required = {"x", "y", "slide_name", "pseudotime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results.csv is missing columns: {missing}")

    if args.slides:
        df = df[df["slide_name"].isin(args.slides)].reset_index(drop=True)

    print(f"Exporting patches: {args.n_per_bin} per bin across 3 bins")
    print(f"  Slides:  {sorted(df['slide_name'].unique())}")
    print(f"  Patches: {len(df)}")
    print()

    manifest = export_patches(
        df,
        png_dir=args.png_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        n_per_bin=args.n_per_bin,
    )

    manifest_path = args.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest saved: {manifest_path}  ({len(manifest)} patches total)")

    print("\nBin summary:")
    for bin_name in manifest["bin"].unique():
        n = (manifest["bin"] == bin_name).sum()
        print(f"  {bin_name}: {n} patches")


if __name__ == "__main__":
    main()
