"""Morphological clustering helpers for PCA, UMAP, and graph-based labels."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict, Optional

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False


# Dimensionality reduction

def fit_pca(
    features: np.ndarray,
    variance_target: float = 0.95,
    max_components: int = None,
) -> Tuple[StandardScaler, PCA, np.ndarray]:
    """Standardize features and fit PCA."""
    print(f"  Standardizing {features.shape[1]}-dim features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    print(f"  PCA: retaining {variance_target:.0%} variance...")
    pca = PCA(n_components=variance_target, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    if max_components and X_pca.shape[1] > max_components:
        print(f"  Capping PCA at {max_components} components (was {X_pca.shape[1]})")
        pca_capped = PCA(n_components=max_components, random_state=42)
        X_pca = pca_capped.fit_transform(X_scaled)
        pca = pca_capped

    var_explained = np.sum(pca.explained_variance_ratio_)
    print(f"  Retained {X_pca.shape[1]} components ({var_explained:.1%} variance)")
    return scaler, pca, X_pca


# UMAP visualization

def run_umap(
    X_pca: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute a UMAP embedding for visualization."""
    if not UMAP_AVAILABLE:
        print("  WARNING: umap-learn not installed, skipping UMAP.")
        return None, None

    print(f"  UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    X_umap = reducer.fit_transform(X_pca)
    return reducer, X_umap


def umap_sensitivity_check(
    X_pca: np.ndarray,
    neighbor_values=(15, 30, 50),
    min_dist_values=(0.05, 0.1, 0.3),
) -> Dict[str, np.ndarray]:
    """Run UMAP with several parameter combinations."""
    if not UMAP_AVAILABLE:
        return {}

    results = {}
    for nn in neighbor_values:
        for md in min_dist_values:
            key = f"nn{nn}_md{md}"
            print(f"  UMAP sensitivity: {key}...")
            reducer = umap.UMAP(
                n_neighbors=nn, min_dist=md, metric="cosine", random_state=42
            )
            results[key] = reducer.fit_transform(X_pca)
    return results


# Clustering in PCA space

def cluster_hdbscan(
    X_pca: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
) -> np.ndarray:
    """Run HDBSCAN in PCA space."""
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan package required. pip install hdbscan")

    print(f"  HDBSCAN (min_cluster={min_cluster_size}, min_samples={min_samples})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels):.1%})")
    return labels


def cluster_leiden(
    X_pca: np.ndarray,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    metric: str = "cosine",
) -> np.ndarray:
    """Run Leiden clustering on a KNN graph built in PCA space."""
    if not LEIDEN_AVAILABLE:
        raise ImportError(
            "leidenalg and igraph are required for Leiden clustering.  "
            "pip install leidenalg python-igraph"
        )

    import sklearn.neighbors

    print(f"  Leiden (n_neighbors={n_neighbors}, resolution={resolution})...")

    # Build the KNN graph in PCA space.
    adj = sklearn.neighbors.kneighbors_graph(
        X_pca, n_neighbors=n_neighbors, mode="distance", metric=metric,
    )

    # Convert the sparse graph to igraph.
    sources, targets = adj.nonzero()
    weights = np.array(adj[sources, targets]).flatten()

    # Convert distances to similarities.
    if weights.max() > 0:
        sigma = np.median(weights)
        similarities = np.exp(-(weights ** 2) / (2 * sigma ** 2))
    else:
        similarities = np.ones_like(weights)

    g = ig.Graph(n=X_pca.shape[0], edges=list(zip(sources.tolist(), targets.tolist())),
                 directed=False)
    g.es["weight"] = similarities.tolist()

    # Simplify the graph.
    g.simplify(combine_edges="max")

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights="weight",
    )
    labels = np.array(partition.membership)

    n_clusters = len(set(labels))
    print(f"  Found {n_clusters} clusters (modularity={partition.modularity:.3f})")
    return labels


def cluster_kmeans(
    X_pca: np.ndarray,
    k_range: Tuple[int, int] = (3, 10),
) -> Tuple[np.ndarray, int, Dict[int, float]]:
    """
    KMeans with silhouette analysis to choose k.

    Returns:
        labels: (N,) cluster assignments for best k.
        best_k: Optimal cluster count.
        scores: {k: silhouette_score} for all tested k.
    """
    print(f"  KMeans silhouette search k={k_range[0]}..{k_range[1]}...")
    scores = {}
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        lab = km.fit_predict(X_pca)
        s = silhouette_score(X_pca, lab)
        scores[k] = s
        print(f"    k={k}: silhouette={s:.3f}")

    best_k = max(scores, key=scores.get)
    print(f"  Best k={best_k} (silhouette={scores[best_k]:.3f})")

    km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km_final.fit_predict(X_pca)
    return labels, best_k, scores


def cluster(
    X_pca: np.ndarray,
    method: str = "leiden",
    **kwargs,
) -> np.ndarray:
    """
    Dispatch to the configured clustering method.

    Args:
        method: "leiden", "hdbscan", or "kmeans"

    Returns:
        labels: (N,) integer cluster assignments.
    """
    if method == "leiden":
        # Pull out only the kwargs that cluster_leiden accepts
        leiden_kwargs = {k: kwargs[k] for k in ("n_neighbors", "resolution", "metric")
                        if k in kwargs}
        return cluster_leiden(X_pca, **leiden_kwargs)
    elif method == "hdbscan":
        hdbscan_kwargs = {k: kwargs[k] for k in ("min_cluster_size", "min_samples")
                          if k in kwargs}
        return cluster_hdbscan(X_pca, **hdbscan_kwargs)
    elif method == "kmeans":
        kmeans_kwargs = {k: kwargs[k] for k in ("k_range",) if k in kwargs}
        labels, _, _ = cluster_kmeans(X_pca, **kmeans_kwargs)
        return labels
    else:
        raise ValueError(f"Unknown clustering method: {method}")


# ── Step 4: Cluster Validation ───────────────────────────────────────

def check_slide_independence(
    labels: np.ndarray,
    slide_ids: np.ndarray,
    dominance_threshold: float = 0.80,
) -> Dict:
    """
    Check whether any cluster is dominated (>threshold) by a single slide.

    Returns dict with per-cluster breakdown and warnings.
    """
    unique_clusters = sorted(set(labels))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    warnings = []
    cluster_breakdown = {}

    for c in unique_clusters:
        mask = labels == c
        slides_in_cluster = slide_ids[mask]
        unique, counts = np.unique(slides_in_cluster, return_counts=True)
        proportions = counts / counts.sum()

        cluster_breakdown[c] = dict(zip(unique.tolist(), proportions.tolist()))

        max_prop = proportions.max()
        if max_prop > dominance_threshold:
            dominant_slide = unique[np.argmax(proportions)]
            msg = (f"Cluster {c}: {max_prop:.0%} from slide {dominant_slide} "
                   f"— likely artifact, not morphology")
            warnings.append(msg)
            print(f"  WARNING: {msg}")

    if not warnings:
        print("  Slide independence check passed — no single-slide dominated clusters.")

    return {"cluster_breakdown": cluster_breakdown, "warnings": warnings}


def get_cluster_centroids(
    X_pca: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, Tuple[np.ndarray, int]]:
    """
    Compute centroid and nearest-to-centroid index for each cluster.

    Returns:
        {cluster_id: (centroid_vector, nearest_patch_index)}
    """
    centroids = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        mask = labels == c
        cluster_features = X_pca[mask]
        centroid = cluster_features.mean(axis=0)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        nearest_local = np.argmin(distances)
        nearest_global = np.where(mask)[0][nearest_local]
        centroids[c] = (centroid, nearest_global)
    return centroids
