"""
Harmony batch correction for phikon PCA features.

Uses scanpy.external.pp.harmony_integrate (wraps harmonypy) to remove
slide-level batch effects from the PCA embedding before clustering and DPT.

Supported batch keys:
  section_number — "2M-1" vs "2M-2" serial-section grouping (2 groups).
                   Use this when the section split dominates, as diagnosed
                   by QC pseudotime violin plots showing PT ≈ 0.8 for all
                   2M-1 slides and PT ≈ 0.05 for all 2M-2 slides.
  slide_id       — one batch per slide (16 groups); most aggressive correction.
  mouse_id       — one batch per mouse (4 groups: 6027/6028/6029/6031).

Expected slide name format: "6027-4L-2M-1_x5"
  mouse_id:       "6027"   (first hyphen-separated token, _x5 stripped)
  section_number: "2M-1"   (last two hyphen-separated tokens, _x5 stripped)
"""

import numpy as np


def _batch_labels(slide_names: list, slide_ids: np.ndarray, key: str) -> np.ndarray:
    """Build a per-patch batch label array from slide names and the batch key."""

    def _mouse(name):
        return name.replace("_x5", "").split("-")[0]

    def _section(name):
        parts = name.replace("_x5", "").split("-")   # ["6027", "4L", "2M", "1"]
        return f"{parts[-2]}-{parts[-1]}"             # "2M-1"

    if key == "slide_id":
        return slide_ids.astype(str)
    elif key == "mouse_id":
        return np.array([_mouse(slide_names[sid]) for sid in slide_ids])
    elif key == "section_number":
        return np.array([_section(slide_names[sid]) for sid in slide_ids])
    else:
        raise ValueError(
            f"Unknown harmony key: '{key}'. "
            f"Choose from: slide_id, section_number, mouse_id"
        )


def apply_harmony(
    X_pca: np.ndarray,
    slide_names: list,
    slide_ids: np.ndarray,
    key: str = "section_number",
    nclust: int = 10,
) -> np.ndarray:
    """
    Apply Harmony batch correction to PCA features.

    Removes batch effects defined by `key` so that clustering and DPT
    reflect morphological variation rather than slide-of-origin.  Returns
    a corrected embedding with the same shape as X_pca.

    Args:
        X_pca:       (N, k) raw PCA features from fit_pca().
        slide_names: List of slide name strings, one entry per slide,
                     indexed by the integer values in slide_ids.
        slide_ids:   (N,) int array mapping each patch to its slide index.
        key:         Batch grouping: "section_number", "slide_id", or "mouse_id".
        nclust:      Number of internal K-means clusters for Harmony. Default 10
                     is appropriate for 2–4 batches; harmonypy's default of 100
                     is over-parameterized here and triggers fast convergence.

    Returns:
        X_corrected: (N, k) Harmony-corrected embedding, same dtype as X_pca.
    """
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError as e:
        raise ImportError(
            "scanpy and anndata are required for Harmony correction. "
            "pip install scanpy anndata"
        ) from e

    try:
        import harmonypy  # noqa: F401 — raises a clear error if missing
    except ImportError:
        raise ImportError(
            "harmonypy is required for Harmony batch correction.\n"
            "  pip install harmonypy\n"
            "Then re-run with --harmony."
        )

    batch = _batch_labels(slide_names, slide_ids, key)
    unique_batches, batch_counts = np.unique(batch, return_counts=True)

    print(f"  Harmony key='{key}': {len(unique_batches)} batches")
    for b, c in sorted(zip(unique_batches.tolist(), batch_counts.tolist())):
        print(f"    {b}: {c} patches")

    # Build a minimal AnnData carrier.  harmony_integrate reads obsm["X_pca"]
    # and writes the corrected result to obsm["X_pca_harmony"].
    adata_tmp = ad.AnnData(X=X_pca.astype(np.float32))
    adata_tmp.obsm["X_pca"] = X_pca.astype(np.float32)
    adata_tmp.obs["batch"] = batch

    print(f"  Running harmony_integrate (nclust={nclust}, device=cpu)...")
    # device='cpu' forces the PyTorch CPU backend. Without it harmonypy 0.2.0
    # auto-detects CUDA and returns Z_corr with the wrong shape on fast
    # convergence (2 iterations), causing anndata to reject the obsm assignment.
    sc.external.pp.harmony_integrate(
        adata_tmp, key="batch", nclust=nclust, device='cpu',
    )

    X_corrected = adata_tmp.obsm["X_pca_harmony"]
    print(f"  Done. Output shape: {X_corrected.shape}")

    if X_corrected.shape != X_pca.shape:
        raise ValueError(
            f"Harmony output shape {X_corrected.shape} != input shape {X_pca.shape}. "
            f"Backend returned a squeezed array — check harmonypy version."
        )

    return X_corrected.astype(X_pca.dtype)
