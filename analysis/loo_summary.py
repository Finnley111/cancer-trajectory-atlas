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

    # Primary metric is Spearman rho (paired patch-level comparison).
    # Wasserstein is kept as secondary but is not the headline result.
    has_spearman = "spearman_rho" in df.columns
    display_cols = ["slide_name", "n_patches", "spearman_rho", "spearman_p", "wasserstein"] \
        if has_spearman else ["slide_name", "n_patches", "wasserstein", "ks_stat", "ks_pvalue"]
    print(df[display_cols].to_string(index=False))

    if has_spearman:
        mean_rho = df["spearman_rho"].mean()
        print(f"\nMean Spearman rho: {mean_rho:.4f}  (n={len(df)} slides)")
        low_rho = df[df["spearman_rho"] < 0.5]["slide_name"].tolist()
        if low_rho:
            print(f"\nLow-rho slides (rho < 0.5): {low_rho}")
            print("  → Check whether these are predominantly from one section (2M-1 vs 2M-2).")
            print("  → A strong negative rho means the pseudotime axis is flipped between runs.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))

        if has_spearman:
            df_plot = df.sort_values("spearman_rho")
            colors = ["#d62728" if r < 0.5 else "#1f77b4" for r in df_plot["spearman_rho"]]
            ax.barh(df_plot["slide_name"], df_plot["spearman_rho"], color=colors)
            ax.axvline(0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
                       label="Threshold (0.5)")
            mean_rho = df["spearman_rho"].mean()
            ax.axvline(mean_rho, color="black", linestyle=":", linewidth=1.0, alpha=0.7,
                       label=f"Mean ({mean_rho:.3f})")
            ax.set_xlabel("Spearman ρ  (paired patch pseudotime: in-manifold vs. projected)")
            ax.set_title("LOO Projection Stability  (primary metric: Spearman ρ)")
        else:
            df_plot = df.sort_values("wasserstein")
            mean_w = df["wasserstein"].mean()
            colors = ["#d62728" if w > 0.3 else "#1f77b4" for w in df_plot["wasserstein"]]
            ax.barh(df_plot["slide_name"], df_plot["wasserstein"], color=colors)
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
