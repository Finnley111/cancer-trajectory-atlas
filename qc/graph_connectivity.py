"""
QC Step 1 — k-NN graph connectivity check.

Checks whether the scanpy neighbor graph built during diffusion pseudotime
computation is fully connected.  A disconnected graph causes compute_dpt to
assign inf to cells in smaller components; those get clamped to the max
finite value, producing the binary (two-island) pseudotime problem.

Usage (from the project root):
    python cancer_trajectory_atlas/qc/graph_connectivity.py \\
        --adata /scratch/finnley1/results/atlas_full/adata_full.h5ad
"""

import numpy as np
from pathlib import Path


def check_graph_connectivity(adata_path: str, output_dir: str = None) -> dict:
    """
    Check whether the scanpy neighbor graph has more than one connected component.

    Args:
        adata_path:  Path to adata_full.h5ad produced by run_all.py.
        output_dir:  Where to write the report (default: adata directory / qc).

    Returns:
        Dict with n_components, component_sizes, is_connected.
    """
    import anndata as ad
    from scipy.sparse.csgraph import connected_components

    print(f"Loading: {adata_path}")
    adata = ad.read_h5ad(adata_path)

    if "connectivities" not in adata.obsp:
        print("\nERROR: 'connectivities' not found in adata.obsp.")
        print("  The neighbor graph has not been stored in this AnnData.")
        print("  Re-run the full pipeline (run_all.py --run) to populate it.")
        return {"error": "no_connectivities"}

    n_components, comp_labels = connected_components(
        adata.obsp["connectivities"], directed=False
    )
    _, comp_counts = np.unique(comp_labels, return_counts=True)
    component_sizes = sorted(comp_counts.tolist(), reverse=True)

    print(f"\n{'='*52}")
    print("GRAPH CONNECTIVITY CHECK")
    print(f"{'='*52}")
    print(f"  Total patches:        {len(adata)}")
    print(f"  Connected components: {n_components}")

    report_lines = [
        "GRAPH CONNECTIVITY CHECK",
        "=" * 52,
        f"adata: {adata_path}",
        f"n_patches: {len(adata)}",
        f"n_components: {n_components}",
        f"component_sizes: {component_sizes}",
        "",
    ]

    if n_components == 1:
        msg = ("STATUS: CONNECTED — single component.\n"
               "  The two-island UMAP is a visualization artifact, not a graph disconnect.\n"
               "  Investigate stain QC (step 2) as the next likely cause.")
        print(f"  {msg}")
        report_lines.append(msg)
    else:
        print(f"  STATUS: DISCONNECTED — {n_components} components  "
              f"<— root cause of binary pseudotime")
        report_lines.append(f"STATUS: DISCONNECTED — {n_components} components")
        report_lines.append("")

        for comp_id in range(n_components):
            mask = comp_labels == comp_id
            n_in = int(mask.sum())
            slides_in = adata.obs["slide_id"].values[mask]
            u_slides, c_slides = np.unique(slides_in, return_counts=True)

            top_prop = float(c_slides.max() / c_slides.sum())
            top_slide = u_slides[int(np.argmax(c_slides))]

            print(f"\n  Component {comp_id}: {n_in} patches")
            report_lines.append(f"Component {comp_id}: {n_in} patches")

            for s, c in sorted(zip(u_slides.tolist(), c_slides.tolist()),
                                key=lambda x: -x[1]):
                line = f"    slide {s}: {c} ({c/n_in:.1%})"
                print(f"  {line}")
                report_lines.append(line)

            if top_prop > 0.8:
                warn = (f"  *** WARNING: {top_prop:.0%} from slide {top_slide}"
                        f" — batch artifact ***")
                print(f"  {warn}")
                report_lines.append(warn)

    if output_dir is None:
        output_dir = Path(adata_path).parent / "qc"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_path = out / "graph_connectivity.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"\n  Report saved: {report_path}")

    return {
        "n_components": int(n_components),
        "component_sizes": component_sizes,
        "is_connected": n_components == 1,
        "n_patches": int(len(adata)),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="QC Step 1: k-NN graph connectivity check"
    )
    parser.add_argument("--adata", required=True, help="Path to adata_full.h5ad")
    parser.add_argument("--output-dir", default=None, help="QC output directory")
    args = parser.parse_args()
    check_graph_connectivity(args.adata, args.output_dir)
