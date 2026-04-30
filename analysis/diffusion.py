"""Diffusion pseudotime utilities built on scanpy."""

import numpy as np
from typing import Optional, Tuple

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


def _require_scanpy():
    if not SCANPY_AVAILABLE:
        raise ImportError(
            "scanpy and anndata are required for diffusion pseudotime.  "
            "pip install scanpy anndata"
        )


def build_adata(
    X_pca: np.ndarray,
    cluster_labels: np.ndarray,
    slide_ids: np.ndarray,
    X_umap: Optional[np.ndarray] = None,
) -> "ad.AnnData":
    """Create an AnnData object from PCA features and metadata."""
    _require_scanpy()

    adata = ad.AnnData(X=X_pca.astype(np.float32))
    adata.obs["cluster"] = cluster_labels.astype(str)
    adata.obs["slide_id"] = slide_ids.astype(str)

    if X_umap is not None:
        adata.obsm["X_umap"] = X_umap.astype(np.float32)

    return adata


def compute_diffusion_map(
    adata: "ad.AnnData",
    n_neighbors: int = 30,
    n_comps: int = 10,
) -> "ad.AnnData":
    """Build the neighbor graph and compute diffusion map components."""
    _require_scanpy()

    print(f"  Building neighbor graph (k={n_neighbors})...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")

    print(f"  Computing diffusion map ({n_comps} components)...")
    sc.tl.diffmap(adata, n_comps=n_comps)

    return adata


def choose_root_cell(
    adata: "ad.AnnData",
    root_cluster: str,
) -> int:
    """
    Select the root cell as the patch closest to the centroid of root_cluster.

    The root cluster should be the most well-organized, low-density,
    morphologically regular cluster — identified by visual inspection
    in Phase 3.

    Args:
        adata: AnnData with cluster labels in adata.obs['cluster'].
        root_cluster: String label of the root cluster.

    Returns:
        Global index of the root cell.
    """
    mask = adata.obs["cluster"].values == str(root_cluster)
    if mask.sum() == 0:
        raise ValueError(f"Root cluster '{root_cluster}' not found in data.")

    cluster_features = adata.X[mask]
    centroid = cluster_features.mean(axis=0)
    distances = np.linalg.norm(cluster_features - centroid, axis=1)
    nearest_local = int(np.argmin(distances))
    root_global = int(np.where(mask)[0][nearest_local])

    print(f"  Root cell: index {root_global} (cluster '{root_cluster}', "
          f"{mask.sum()} patches in cluster)")
    return root_global


def compute_dpt(
    adata: "ad.AnnData",
    root_cluster: Optional[str] = None,
    root_index: Optional[int] = None,
) -> "ad.AnnData":
    """
    Compute Diffusion Pseudotime from a biologically anchored root.

    Either specify root_cluster (recommended — will auto-select the centroid
    patch) or root_index (if you already know which patch to use).

    Results stored in adata.obs['dpt_pseudotime'] and adata.obs['pseudotime']
    (the latter normalized to [0, 1]).
    """
    _require_scanpy()

    if root_cluster is not None:
        root_idx = choose_root_cell(adata, root_cluster)
    elif root_index is not None:
        root_idx = root_index
    else:
        raise ValueError("Must specify either root_cluster or root_index.")

    adata.uns["iroot"] = root_idx

    print("  Computing diffusion pseudotime...")
    sc.tl.dpt(adata)

    # Normalize to [0, 1]
    pt = adata.obs["dpt_pseudotime"].values.copy()

    # Handle infinities (disconnected components)
    finite_mask = np.isfinite(pt)
    if not finite_mask.all():
        n_inf = (~finite_mask).sum()
        print(f"  WARNING: {n_inf} patches have infinite DPT (disconnected). "
              f"Clamping to max finite value.")
        pt[~finite_mask] = pt[finite_mask].max()

    pt_min, pt_max = pt.min(), pt.max()
    if pt_max - pt_min < 1e-10:
        print("  WARNING: DPT range is near-zero — no trajectory detected.")
        adata.obs["pseudotime"] = np.zeros(len(pt))
    else:
        adata.obs["pseudotime"] = (pt - pt_min) / (pt_max - pt_min)

    print(f"  Pseudotime range: [{pt_min:.4f}, {pt_max:.4f}] → normalized [0, 1]")
    return adata


# ── Convenience wrapper ──────────────────────────────────────────────

def run_diffusion_pseudotime(
    X_pca: np.ndarray,
    cluster_labels: np.ndarray,
    slide_ids: np.ndarray,
    root_cluster: str,
    X_umap: Optional[np.ndarray] = None,
    n_neighbors: int = 30,
    n_comps: int = 10,
) -> "ad.AnnData":
    """
    Full Phase 4 pipeline: build AnnData → diffusion map → DPT.

    Returns the fully populated AnnData object.
    """
    adata = build_adata(X_pca, cluster_labels, slide_ids, X_umap)
    compute_diffusion_map(adata, n_neighbors=n_neighbors, n_comps=n_comps)
    compute_dpt(adata, root_cluster=root_cluster)
    return adata
