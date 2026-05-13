"""
QC Step 4 — Per-slide and per-mouse pseudotime violin plots.

The main pipeline's fig5_pt_violins.png only stratifies by cluster.  This
module adds two more stratifications that directly reveal batch effects:

  pt_by_slide.png  — one violin per slide (16 total); if pseudotime is driven
                     by batch rather than biology, slides will occupy distinct
                     ranges instead of overlapping broadly.

  pt_by_mouse.png  — one violin per mouse (4 mice, ~4 slides each); a cleaner
                     signal if the per-slide plot looks noisy.

Mouse ID is parsed as the first hyphen-delimited token of the slide stem
(e.g. "6027-4L-2M-1_x5" → mouse "6027").

Usage (from the project root):
    python cancer_trajectory_atlas/qc/pseudotime_by_slide.py \\
        --adata       /scratch/finnley1/results/atlas_full/adata_full.h5ad \\
        --results-csv /scratch/finnley1/results/atlas_full/results.csv \\
        --output-dir  /scratch/finnley1/results/atlas_full/qc
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def _violin_figure(groups: list, labels: list, title: str, save_path: Path):
    """Draw a violin plot and save it."""
    # Drop any group that is all-NaN or empty
    valid = [(g, l) for g, l in zip(groups, labels) if len(g) > 0 and np.isfinite(g).any()]
    if not valid:
        print(f"  WARNING: no valid data for {title} — skipping.")
        return
    groups, labels = zip(*valid)

    global_median = float(np.median(np.concatenate(groups)))

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.9), 6))
    parts = ax.violinplot(groups, showmedians=True, showextrema=True)

    # Color violins sequentially so they're easy to distinguish
    cmap = plt.cm.get_cmap("tab20", len(groups))
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(cmap(i))
        pc.set_alpha(0.75)

    ax.axhline(global_median, color="black", linestyle="--", linewidth=1,
               alpha=0.5, label=f"global median ({global_median:.3f})")
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("Pseudotime [0, 1]")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pseudotime_by_slide(
    adata_path: str,
    results_csv: str,
    output_dir: str,
) -> None:
    """
    Plot pseudotime distributions stratified by slide and by mouse.

    Args:
        adata_path:   Path to adata_full.h5ad.
        results_csv:  Path to results.csv (provides slide_name strings).
        output_dir:   Where to save figures.
    """
    import anndata as ad
    import pandas as pd

    print(f"Loading: {adata_path}")
    adata = ad.read_h5ad(adata_path)

    if "pseudotime" not in adata.obs.columns:
        print("ERROR: 'pseudotime' not in adata.obs — run the full pipeline first.")
        return

    pseudotime = adata.obs["pseudotime"].values.astype(float)

    # Build slide_id (int) → slide_name (str) mapping from results.csv
    df = pd.read_csv(results_csv)
    if "slide_name" not in df.columns or "slide_id" not in df.columns:
        print("ERROR: results.csv must have 'slide_id' and 'slide_name' columns.")
        return

    id_to_name = (
        df[["slide_id", "slide_name"]]
        .drop_duplicates()
        .set_index("slide_id")["slide_name"]
        .to_dict()
    )

    # adata.obs["slide_id"] is stored as str (see diffusion.py:35)
    slide_ids_str = adata.obs["slide_id"].values
    slide_names = np.array([id_to_name.get(int(sid), f"slide_{sid}")
                            for sid in slide_ids_str])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Per-slide violins ────────────────────────────────────────────
    unique_slides = sorted(set(slide_names))
    groups_slide  = [pseudotime[slide_names == s] for s in unique_slides]
    # Shorten labels: strip trailing "_x5" for readability
    labels_slide  = [s.replace("_x5", "") for s in unique_slides]

    _violin_figure(
        groups_slide, labels_slide,
        title="Pseudotime distribution by slide\n"
              "(batch effect: slides should overlap broadly)",
        save_path=out / "pt_by_slide.png",
    )

    # ── Per-mouse violins ────────────────────────────────────────────
    # Mouse ID = first hyphen-separated token (e.g. "6027-4L-2M-1_x5" → "6027")
    mouse_ids = np.array([name.split("-")[0] for name in slide_names])
    unique_mice = sorted(set(mouse_ids))
    groups_mouse = [pseudotime[mouse_ids == m] for m in unique_mice]

    _violin_figure(
        groups_mouse, unique_mice,
        title="Pseudotime distribution by mouse\n"
              "(each mouse contributes ~4 slides)",
        save_path=out / "pt_by_mouse.png",
    )

    # Print summary stats
    print(f"\n  {'Slide':<35s} {'N':>7s} {'Median':>8s} {'IQR':>8s}")
    print("  " + "-" * 62)
    for name, grp in zip(unique_slides, groups_slide):
        if len(grp) == 0:
            continue
        med = np.median(grp)
        iqr = float(np.percentile(grp, 75) - np.percentile(grp, 25))
        print(f"  {name:<35s} {len(grp):>7d} {med:>8.3f} {iqr:>8.3f}")

    print(f"\n  {'Mouse':<10s} {'N':>7s} {'Median':>8s} {'IQR':>8s}")
    print("  " + "-" * 38)
    for mid, grp in zip(unique_mice, groups_mouse):
        if len(grp) == 0:
            continue
        med = np.median(grp)
        iqr = float(np.percentile(grp, 75) - np.percentile(grp, 25))
        print(f"  {mid:<10s} {len(grp):>7d} {med:>8.3f} {iqr:>8.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="QC Step 4: per-slide and per-mouse pseudotime violins"
    )
    parser.add_argument("--adata",       required=True, help="Path to adata_full.h5ad")
    parser.add_argument("--results-csv", required=True, help="Path to results.csv")
    parser.add_argument("--output-dir",  required=True, help="QC output directory")
    args = parser.parse_args()
    plot_pseudotime_by_slide(args.adata, args.results_csv, args.output_dir)
