"""Post-hoc analysis: how much of pseudotime is explained by gross cellularity?

Computes Spearman rho between pseudotime and a composite cellularity proxy
(PC1 of nuclear_density, mean_nuclear_area, nc_ratio), then checks whether
residual signal remains in texture_entropy, h_intensity, and packing_irregularity
after regressing out cellularity.

Usage:
    python -m cancer_trajectory_atlas.analysis.cellularity_confound \\
        --results-dirs $SCRATCH/results/atlas_full_macenko \\
                       $SCRATCH/results/atlas_full_reinhard \\
        --output-dir   $SCRATCH/results/confound_analysis
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA as SklearnPCA

CELLULARITY_FEATURES = ["nuclear_density", "mean_nuclear_area", "nc_ratio"]
OTHER_FEATURES = ["texture_entropy", "h_intensity", "packing_irregularity"]


def compute_cellularity_proxy(obs_df):
    """First PC of the three nuclear features (scaled). Returns (proxy, pc1_variance_explained)."""
    X = obs_df[CELLULARITY_FEATURES].values.astype(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X_scaled = (X - mu) / sd

    pca = SklearnPCA(n_components=1)
    proxy = pca.fit_transform(X_scaled)[:, 0]

    # Orient so that higher proxy = denser cellularity (nuclear_density loads positively)
    if pca.components_[0, 0] < 0:
        proxy = -proxy

    return proxy, float(pca.explained_variance_ratio_[0])


def compute_residual_pseudotime(pseudotime, cellularity_proxy):
    """OLS residuals of pseudotime ~ cellularity_proxy."""
    X = np.column_stack([np.ones(len(cellularity_proxy)), cellularity_proxy])
    coeffs, _, _, _ = np.linalg.lstsq(X, pseudotime, rcond=None)
    return pseudotime - X @ coeffs


def analyze_run(results_dir: Path, output_dir: Path):
    """Analyze one pipeline run directory. Returns a summary dict, or None on failure."""
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata is required: pip install anndata")

    h5ad = results_dir / "adata_full.h5ad"
    if not h5ad.exists():
        print(f"  SKIP: {h5ad} not found")
        return None

    print(f"\n  Loading {results_dir.name}...")
    adata = ad.read_h5ad(h5ad)
    obs = adata.obs.copy()

    missing = [c for c in CELLULARITY_FEATURES + ["pseudotime"] if c not in obs.columns]
    if missing:
        print(f"  SKIP: missing columns {missing}")
        return None

    pseudotime = obs["pseudotime"].values.astype(float)
    proxy, pc1_var = compute_cellularity_proxy(obs)
    residuals = compute_residual_pseudotime(pseudotime, proxy)

    rho_cell, p_cell = spearmanr(pseudotime, proxy)

    row = {
        "run": results_dir.name,
        "n_patches": len(obs),
        "rho_cellularity": rho_cell,
        "p_cellularity": p_cell,
        "pc1_var_explained": pc1_var,
    }

    for feat in OTHER_FEATURES:
        if feat in obs.columns:
            vals = obs[feat].values.astype(float)
            rho_full, p_full   = spearmanr(pseudotime, vals)
            rho_resid, p_resid = spearmanr(residuals,  vals)
            row[f"rho_{feat}_full"]     = rho_full
            row[f"rho_{feat}_residual"] = rho_resid
            row[f"p_{feat}_full"]       = p_full
            row[f"p_{feat}_residual"]   = p_resid

    # Scatter: cellularity proxy vs pseudotime
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hexbin(proxy, pseudotime, gridsize=40, cmap="Blues", mincnt=1)
        ax.set_xlabel("Cellularity proxy (PC1 of nuclear features)")
        ax.set_ylabel("Pseudotime")
        ax.set_title(
            f"{results_dir.name}\nSpearman ρ = {rho_cell:.3f}  p = {p_cell:.2e}"
        )
        fig.tight_layout()
        out_png = output_dir / f"cellularity_scatter_{results_dir.name}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_png.name}")
    except Exception as exc:
        print(f"  WARNING: Could not save scatter plot: {exc}")

    return row


def _decision_gate(rho):
    if rho < 0.3:
        return "SKIP Exp 3 — pseudotime barely tracks cellularity"
    if rho > 0.7:
        return "RECONSIDER — pseudotime is mainly a cellularity meter; reframe paper"
    return "RUN Exp 3 — partial confounding, residual signal worth exploring"


def main():
    parser = argparse.ArgumentParser(
        description="Cellularity confound test: rho(pseudotime, nuclear cellularity proxy)"
    )
    parser.add_argument(
        "--results-dirs", nargs="+", type=Path, required=True,
        help="Pipeline output directories containing adata_full.h5ad",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Destination for confound_summary.csv and scatter plots",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in args.results_dirs:
        row = analyze_run(Path(d), args.output_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No valid results found — nothing to summarise.")
        return

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "confound_summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSummary written: {csv_path}")

    display_cols = ["run", "n_patches", "rho_cellularity", "p_cellularity"]
    print(df[display_cols].to_string(index=False))

    print("\nDecision gates for Experiment 3:")
    for _, r in df.iterrows():
        print(f"  {r['run']}: ρ = {r['rho_cellularity']:.3f} → {_decision_gate(r['rho_cellularity'])}")


if __name__ == "__main__":
    main()
