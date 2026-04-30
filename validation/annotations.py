"""Annotation loading for QuPath GeoJSON and legacy mask files."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_annotations(
    annotation_path: str,
    coords: np.ndarray,
    patch_size: int,
    label_order: Dict[str, int],
    fmt: str = "geojson",
    image_path: Optional[str] = None,
    stride: Optional[int] = None,
    coordinate_space: str = "pixel",
    min_annotated_fraction_warn: float = 0.01,
    original_full_width: Optional[int] = None,
    original_full_height: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load annotations for one slide and assign a label to each patch."""
    if annotation_path is None:
        n = len(coords)
        return np.full(n, np.nan), np.array([None] * n, dtype=object)

    if fmt == "geojson":
        return _load_qupath_geojson(
            annotation_path, coords, patch_size, label_order,
            image_path=image_path,
            coordinate_space=coordinate_space,
            min_annotated_fraction_warn=min_annotated_fraction_warn,
            original_full_width=original_full_width,
            original_full_height=original_full_height,
        )
    elif fmt == "mask":
        return _load_mask_annotations(
            annotation_path, image_path, coords, patch_size, stride, label_order
        )
    else:
        raise ValueError(f"Unknown annotation format: {fmt}")


def _load_qupath_geojson(
    geojson_path, coords, patch_size, label_order,
    image_path=None, coordinate_space="pixel",
    min_annotated_fraction_warn=0.01,
    original_full_width=None,
    original_full_height=None,
):
    """Parse QuPath GeoJSON and label patches by polygon containment."""
    from matplotlib.path import Path as MplPath

    # Determine the dimensions to use for ratio → pixel conversion
    img_w = img_h = None
    cropped_w = cropped_h = None

    if coordinate_space == "ratio":
        if image_path is None and original_full_width is None:
            raise ValueError("Ratio coordinate space requires image_path or original dimensions")

        # Read the actual image size.
        if image_path is not None:
            from PIL import Image
            with Image.open(image_path) as im:
                cropped_w, cropped_h = im.size

        # Use the original full dimensions if they are available.
        if original_full_width is not None:
            img_w = original_full_width
            img_h = original_full_height if original_full_height is not None else cropped_h
            print(f"    Ratio coords scaled against ORIGINAL size: {img_w}x{img_h}")
            print(f"    Cropped image size: {cropped_w}x{cropped_h}")
        elif cropped_w is not None:
            img_w, img_h = cropped_w, cropped_h
        else:
            raise ValueError("Cannot determine image dimensions for ratio scaling")

    with open(geojson_path) as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        features_list = data["features"]
    elif isinstance(data, list):
        features_list = data
    else:
        features_list = [data]

    def scale_ring(ring):
        arr = np.asarray(ring, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        if coordinate_space == "ratio":
            arr[:, 0] *= img_w
            arr[:, 1] *= img_h
        return arr[:, :2]

    polygons = []
    for feat in features_list:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        classification = props.get("classification", {})
        if isinstance(classification, dict):
            label_name = classification.get("name", "Unknown")
        elif isinstance(classification, str):
            label_name = classification
        else:
            label_name = "Unknown"

        if label_order and label_name not in label_order:
            continue

        geom_type = geom.get("type", "")
        if geom_type == "Polygon":
            ring = scale_ring(geom["coordinates"][0])
            if ring is not None and len(ring) >= 3:
                polygons.append((MplPath(ring), label_name))
        elif geom_type == "MultiPolygon":
            for poly_coords in geom["coordinates"]:
                ring = scale_ring(poly_coords[0])
                if ring is not None and len(ring) >= 3:
                    polygons.append((MplPath(ring), label_name))

    # Drop polygons that fall outside the cropped image bounds.
    if cropped_w is not None and original_full_width is not None:
        filtered = []
        n_discarded = 0
        for poly, label_name in polygons:
            verts = poly.vertices
            # Keep the polygon if its centroid is within the cropped bounds.
            cx, cy = verts[:, 0].mean(), verts[:, 1].mean()
            if cx <= cropped_w and cy <= (cropped_h or img_h):
                filtered.append((poly, label_name))
            else:
                n_discarded += 1
        if n_discarded > 0:
            print(f"    Discarded {n_discarded} polygons outside cropped region")
        polygons = filtered

    if not polygons:
        print(f"    WARNING: No valid polygons in {geojson_path}")
    else:
        print(f"    Loaded {len(polygons)} annotation polygons")

    n = len(coords)
    labels = np.full(n, np.nan)
    label_names = np.array([None] * n, dtype=object)

    half = patch_size / 2.0
    centers = coords.astype(float) + half

    for i in range(n):
        pt = (centers[i, 0], centers[i, 1])
        for poly, name in polygons:
            if poly.contains_point(pt):
                label_names[i] = name
                if label_order and name in label_order:
                    labels[i] = label_order[name]
                break

    annotated = np.array([ln is not None for ln in label_names])
    n_ann = int(annotated.sum())
    frac = n_ann / n if n > 0 else 0.0
    print(f"    {n_ann}/{n} patches annotated ({frac:.1%})")

    if frac < min_annotated_fraction_warn:
        print("    WARNING: Very low annotation coverage.")

    return labels, label_names


def _load_mask_annotations(mask_path, image_path, coords, patch_size, stride, label_order):
    """Legacy colored-mask label loading."""
    import cv2
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None

    with Image.open(image_path) as img:
        target_w, target_h = img.size

    img_mask = cv2.imread(str(mask_path))
    if img_mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")

    h_m, w_m = img_mask.shape[:2]
    if w_m != target_w or h_m != target_h:
        img_mask = cv2.resize(img_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    hsv = cv2.cvtColor(img_mask, cv2.COLOR_BGR2HSV)

    # Black ink (Early)
    mask_early = _get_filled_mask(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    # Red ink (Late)
    mask_late1 = _get_filled_mask(hsv, np.array([0, 100, 100]), np.array([15, 255, 255]))
    mask_late2 = _get_filled_mask(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    mask_late = cv2.bitwise_or(mask_late1, mask_late2)

    n = len(coords)
    labels = np.full(n, np.nan)
    label_names = np.array([None] * n, dtype=object)
    threshold = 0.3

    for i, (x, y) in enumerate(coords):
        x, y = int(x), int(y)
        roi_late = mask_late[y:y + patch_size, x:x + patch_size]
        if np.mean(roi_late) > 255 * threshold:
            labels[i] = 2
            label_names[i] = "Late"
        else:
            roi_early = mask_early[y:y + patch_size, x:x + patch_size]
            if np.mean(roi_early) > 255 * threshold:
                labels[i] = 1
                label_names[i] = "Early"

    annotated = np.array([ln is not None for ln in label_names])
    print(f"    {annotated.sum()}/{n} patches annotated")
    return labels, label_names


def _get_filled_mask(hsv, lower, upper):
    """Find colored lines in HSV and fill enclosed regions."""
    import cv2
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled
