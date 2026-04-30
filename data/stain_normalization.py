"""Stain normalization helpers for slide preprocessing."""

import numpy as np
from pathlib import Path


# Reinhard normalizer

class ReinhardNormalizer:
    """
    Reinhard color normalization in LAB space.

    Transfers the mean and standard deviation of each LAB channel from
    a reference image to the target image.  Simple, fast, and doesn't
    require any problematic native libraries.

    Reference:
        Reinhard et al., "Color Transfer between Images", IEEE CG&A 2001.
    """

    def __init__(self):
        self.ref_means = None
        self.ref_stds = None

    def fit(self, reference_rgb: np.ndarray):
        """Compute LAB channel statistics from a reference image."""
        import cv2
        lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)

        # Only compute stats from tissue pixels.
        tissue_mask = lab[:, :, 0] < 230
        if tissue_mask.sum() < 1000:
            # Fall back to all pixels if tissue detection fails.
            tissue_mask = np.ones(lab.shape[:2], dtype=bool)

        self.ref_means = np.array([lab[:, :, c][tissue_mask].mean() for c in range(3)])
        self.ref_stds = np.array([lab[:, :, c][tissue_mask].std() + 1e-6 for c in range(3)])

    def transform(self, image_rgb: np.ndarray) -> np.ndarray:
        """Normalize an image to match the reference color distribution."""
        import cv2

        if self.ref_means is None:
            raise RuntimeError("Call fit() with a reference image first.")

        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)

        # Compute stats from tissue pixels of the source image.
        tissue_mask = lab[:, :, 0] < 230
        if tissue_mask.sum() < 1000:
            tissue_mask = np.ones(lab.shape[:2], dtype=bool)

        src_means = np.array([lab[:, :, c][tissue_mask].mean() for c in range(3)])
        src_stds = np.array([lab[:, :, c][tissue_mask].std() + 1e-6 for c in range(3)])

        # Normalize the source, then shift to the reference distribution.
        for c in range(3):
            lab[:, :, c] = (lab[:, :, c] - src_means[c]) / src_stds[c]
            lab[:, :, c] = lab[:, :, c] * self.ref_stds[c] + self.ref_means[c]

        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# Public API

def build_normalizer(method: str, reference_image_path: str):
    """Build a stain normalizer fitted to a reference slide."""
    method = (method or "none").lower()
    if method == "none":
        return None

    print(f"  Building {method.title()} normalizer from: {reference_image_path}")

    if method == "reinhard":
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        ref_img = np.array(Image.open(reference_image_path).convert("RGB"))
        normalizer = ReinhardNormalizer()
        normalizer.fit(ref_img)
        print("  Normalizer ready.")
        return normalizer

    # Macenko and Vahadane use staintools.
    if method in ("macenko", "vahadane"):
        try:
            import staintools
        except ImportError as exc:
            raise ImportError(
                "staintools is required for Macenko/Vahadane normalization.  "
                "pip install staintools\n"
                "If spams won't install, use method='reinhard' instead."
            ) from exc

        target = staintools.read_image(str(reference_image_path))
        normalizer = staintools.StainNormalizer(method=method)
        normalizer.fit(target)
        print("  Normalizer ready.")
        return normalizer

    raise ValueError(f"Unsupported stain normalization method: {method}")


def normalize_slide(image_array: np.ndarray, normalizer, slide_name: str = "") -> np.ndarray:
    """Apply stain normalization to one slide image array."""
    if normalizer is None:
        return image_array

    try:
        normalized = np.asarray(normalizer.transform(image_array))
        if normalized.dtype != np.uint8:
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        print(f"    Applied stain normalization to {slide_name}")
        return normalized
    except Exception as exc:
        print(f"    WARNING: Stain normalization failed for {slide_name}: {exc}")
        return image_array