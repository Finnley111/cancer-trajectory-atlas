"""Aggregate all LOO projection results into a summary CSV and stability bar chart.

Run after all 16 LOO jobs and their loo_project.py steps have completed.

Usage:
    python -m cancer_trajectory_atlas.analysis.loo_summary \\
        --loo-dirs $SCRATCH/results/loo_* \\
        --output-dir $SCRATCH/results/loo_summary
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate LOO projection results into summary table and stability figure."
    )
    parser.add_argument("--loo-dirs", nargs="+", type=Path, required=True,
                        help="LOO output directories, each containing loo_result_*.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination for loo_summary.csv and loo_stability_figure.png")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for loo_dir in sorted(args.loo_dirs):
        loo_dir = Path(loo_dir)
        for jf in sorted(loo_dir.glob("loo_result_*.json")):
            with open(jf) as f:
                rows.append(json.load(f))

    if not rows:
        print("No loo_result_*.json files found — check --loo-dirs.")
        return

    df = pd.DataFrame(rows).sort_values("wasserstein").reset_index(drop=True)

    csv_path = args.output_dir / "loo_summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Summary: {csv_path}")

    display_cols = ["slide_name", "n_inmanifold", "n_projected", "wasserstein", "ks_stat", "ks_pvalue"]
    print(df[display_cols].to_string(index=False))

    mean_w = df["wasserstein"].mean()
    print(f"\nMean Wasserstein: {mean_w:.4f}  (n={len(df)} slides)")

    outliers = df[df["wasserstein"] > 0.3]["slide_name"].tolist()
    if outliers:
        print(f"\nHigh-distance slides (Wasserstein > 0.3): {outliers}")
        print("  → Check whether these are predominantly from one section (2M-1 vs 2M-2).")
        print("  → If so, cross-section generalization is the limiting factor, not manifold instability.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#d62728" if w > 0.3 else "#1f77b4" for w in df["wasserstein"]]
        ax.barh(df["slide_name"], df["wasserstein"], color=colors)
        ax.axvline(0.3, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
                   label="Threshold (0.3)")
        ax.axvline(mean_w, color="black", linestyle=":", linewidth=1.0, alpha=0.7,
                   label=f"Mean ({mean_w:.3f})")
        ax.set_xlabel("Wasserstein distance  (in-manifold vs. projected pseudotime)")
        ax.set_title("LOO Projection Stability")
        ax.legend()
        fig.tight_layout()

        fig_path = args.output_dir / "loo_stability_figure.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {fig_path.name}")
    except Exception as exc:
        print(f"WARNING: Could not save figure: {exc}")


if __name__ == "__main__":
    main()
