"""
Default settings for the Cancer Trajectory Atlas pipeline.

Override these values with CLI args or a config JSON.
"""

# Model settings
DEFAULT_MODEL = "phikon"
AVAILABLE_MODELS = ["resnet18", "resnet50", "resnet101", "phikon", "phikon-v2"]

# Patching
DEFAULT_PATCH_SIZE = 112  # Matches Phikon's input size
DEFAULT_STRIDE = 96
DEFAULT_MIN_ROI_OVERLAP = 0.5  # Keep patches mostly inside the ROI
DEFAULT_WHITE_THRESH = 220  # RGB values above this count as background
DEFAULT_WHITE_FRAC = 0.70  # Discard patches with too much white space

# Legacy HSV tissue filter (still available as secondary filter)
DEFAULT_SAT_THRESH = 15
DEFAULT_VAL_THRESH = 230

# Dimensionality reduction
DEFAULT_PCA_VARIANCE = 0.95  # Keep 95% of the variance
DEFAULT_N_PCA_FALLBACK = 50  # Fallback if the variance rule fails

# UMAP (visualization only)
DEFAULT_UMAP_NEIGHBORS = 30
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_METRIC = "cosine"

# Clustering
DEFAULT_CLUSTERING_METHOD = "leiden"  # "leiden", "hdbscan", or "kmeans"
DEFAULT_LEIDEN_NEIGHBORS = 15
DEFAULT_LEIDEN_RESOLUTION = 1.0
DEFAULT_HDBSCAN_MIN_CLUSTER = 50
DEFAULT_HDBSCAN_MIN_SAMPLES = 10
DEFAULT_KMEANS_RANGE = (3, 10)  # Silhouette search range

# Diffusion pseudotime
DEFAULT_N_DIFFMAP_COMPS = 10
DEFAULT_DPT_NEIGHBORS = 30

# Validation
DEFAULT_N_PERMUTATIONS = 1000
DEFAULT_CORRELATION_THRESHOLD = 0.4  # |r| above this is meaningful
DEFAULT_CLUSTER_SLIDE_DOMINANCE = 0.80  # Flag clusters dominated by one slide

# Stain normalization
DEFAULT_STAIN_METHOD = "macenko"  # "macenko", "vahadane", or "none"

# Output
DEFAULT_OUTPUT_DIR = "results/atlas_output"
