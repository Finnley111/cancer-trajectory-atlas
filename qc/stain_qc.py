"""
QC Step 2 — Stain normalization quality check.

For each slide produces:
  - 6-panel figure: original thumbnail | normalized thumbnail | diff image,
    then H-channel histogram | E-channel histogram | LAB-L histogram (pre/post overlay)
  - For Macenko: stain vector angles relative to the reference slide
  - Summary CSV with per-slide LAB stats and angle deviations
  - For Macenko: bar chart of H/E stain angles across all slides

Silent normalization failures (where the output array == input array) are flagged
explicitly — the main pipeline swallows these exceptions silently at
data/stain_normalization.py:114-116.

Usage (from the project root):
    python cancer_trajectory_atlas/qc/stain_qc.py \\
        --slides-dir /scratch/finnley1/data/MCF7_x5_cropped \\
        --stain-method macenko \\
        --output-dir /scratch/finnley1/results/atlas_full/qc
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Max pixels on the longer dimension for QC thumbnails.
_THUMB_LONG = 1024


# ── Inline Reinhard normalizer ───────────────────────────────────────
# Copied from data/stain_normalization.py to keep this module self-contained.

class _ReinhardNorm:
    """Reinhard color normalization in LAB space (tissue-pixels only)."""

    def __init__(self):
        self.ref_means = None
        self.ref_stds = None

    def fit(self, rgb: np.ndarray):
        import cv2
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
        mask = lab[:, :, 0] < 230
        if mask.sum() < 1000:
            mask = np.ones(lab.shape[:2], dtype=bool)
        self.ref_means = [lab[:, :, c][mask].mean() for c in range(3)]
        self.ref_stds  = [lab[:, :, c][mask].std() + 1e-6 for c in range(3)]

    def transform(self, rgb: np.ndarray) -> np.ndarray:
        import cv2
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
        mask = lab[:, :, 0] < 230
        if mask.sum() < 1000:
            mask = np.ones(lab.shape[:2], dtype=bool)
        src_means = [lab[:, :, c][mask].mean() for c in range(3)]
        src_stds  = [lab[:, :, c][mask].std() + 1e-6 for c in range(3)]
        for c in range(3):
            lab[:, :, c] = (lab[:, :, c] - src_means[c]) / src_stds[c]
            lab[:, :, c] = lab[:, :, c] * self.ref_stds[c] + self.ref_means[c]
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        import cv2 as _cv2
        return _cv2.cvtColor(lab, _cv2.COLOR_LAB2RGB)


# ── Helpers ──────────────────────────────────────────────────────────

def _load_thumbnail(image_path: str) -> np.ndarray:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = _THUMB_LONG / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return np.array(img)


def _lab_stats(rgb: np.ndarray) -> dict:
    """Mean and std of each LAB channel for tissue pixels (L < 230)."""
    import cv2
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    mask = lab[:, :, 0] < 230
    if mask.sum() < 100:
        mask = np.ones(lab.shape[:2], dtype=bool)
    stats = {}
    for i, ch in enumerate(["L", "A", "B"]):
        vals = lab[:, :, i][mask]
        stats[f"{ch}_mean"] = float(vals.mean())
        stats[f"{ch}_std"]  = float(vals.std())
    return stats


def _he_channels(rgb: np.ndarray):
    """Return (H, E) channel arrays via color deconvolution."""
    from skimage.color import rgb2hed
    hed = rgb2hed(rgb)
    return hed[:, :, 0], hed[:, :, 1]


def _apply_norm(rgb: np.ndarray, normalizer) -> tuple:
    """Apply normalizer; return (result_array, changed_flag)."""
    if normalizer is None:
        return rgb, False
    try:
        out = np.asarray(normalizer.transform(rgb))
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        changed = not np.allclose(out.astype(float), rgb.astype(float), atol=1.0)
        return out, changed
    except Exception as e:
        print(f"    Normalization raised: {e}")
        return rgb, False


def _macenko_stain_matrix(rgb: np.ndarray):
    """Extract 2×3 Macenko stain matrix; returns None on failure."""
    try:
        import staintools
        mat = staintools.MacenkoStainExtractor().get_stain_matrix(rgb)
        return mat  # shape (2, 3): row 0 = H, row 1 = E
    except Exception as e:
        print(f"    Stain matrix extraction failed: {e}")
        return None


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ── Per-slide figure ─────────────────────────────────────────────────

def _make_slide_figure(slide_name, orig, norm, changed, stain_method, out_dir):
    import cv2
    orig_h, orig_e = _he_channels(orig)
    norm_h, norm_e = _he_channels(norm)
    orig_lab = cv2.cvtColor(orig, cv2.COLOR_RGB2LAB)
    norm_lab = cv2.cvtColor(norm, cv2.COLOR_RGB2LAB)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    warn_str = "" if changed else "  *** UNCHANGED — normalization may have failed ***"
    fig.suptitle(f"{slide_name}  [{stain_method}]{warn_str}", fontsize=12,
                 color="black" if changed else "red")

    # Row 0: thumbnails
    axes[0, 0].imshow(orig);  axes[0, 0].set_title("Original");   axes[0, 0].axis("off")
    axes[0, 1].imshow(norm);  axes[0, 1].set_title("Normalized"); axes[0, 1].axis("off")
    diff = np.clip(norm.astype(int) - orig.astype(int) + 128, 0, 255).astype(np.uint8)
    axes[0, 2].imshow(diff);  axes[0, 2].set_title("Diff (128 = no change)"); axes[0, 2].axis("off")

    # Row 1: histograms
    bins = 80
    for ax, pre, post, title, xlabel in [
        (axes[1, 0], orig_h.ravel(), norm_h.ravel(), "Hematoxylin channel", "HED H"),
        (axes[1, 1], orig_e.ravel(), norm_e.ravel(), "Eosin channel",       "HED E"),
        (axes[1, 2], orig_lab[:,:,0].ravel().astype(float),
                     norm_lab[:,:,0].ravel().astype(float), "LAB — L (brightness)", "L value"),
    ]:
        ax.hist(pre,  bins=bins, alpha=0.55, label="pre",  color="steelblue", density=True)
        ax.hist(post, bins=bins, alpha=0.55, label="post", color="darkorange", density=True)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = out_dir / f"{slide_name}_stain_qc.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


# ── Main entry point ─────────────────────────────────────────────────

def run_stain_qc(
    slides_dir: str,
    stain_method: str,
    output_dir: str,
    reference_slide: str = None,
) -> None:
    """
    Run stain normalization QC on all PNG slides in slides_dir.

    Args:
        slides_dir:      Directory containing .png slide files.
        stain_method:    "reinhard", "macenko", or "none".
        output_dir:      Root QC output directory; outputs land in output_dir/stain_qc/.
        reference_slide: Path to reference PNG (default: first slide alphabetically,
                         matching the main pipeline's behavior).
    """
    import pandas as pd

    slides_dir = Path(slides_dir)
    out = Path(output_dir) / "stain_qc"
    out.mkdir(parents=True, exist_ok=True)

    slide_paths = sorted(slides_dir.glob("*.png"))
    if not slide_paths:
        print(f"No PNG files found in {slides_dir}")
        return

    if reference_slide is None:
        reference_slide = str(slide_paths[0])
    print(f"Reference slide: {Path(reference_slide).name}")
    print(f"Stain method:    {stain_method}")
    print(f"Slides found:    {len(slide_paths)}")

    # Build normalizer from reference thumbnail
    normalizer = None
    ref_stain_mat = None
    ref_thumb = _load_thumbnail(reference_slide)

    if stain_method.lower() == "reinhard":
        normalizer = _ReinhardNorm()
        normalizer.fit(ref_thumb)
        print("  Reinhard normalizer fitted.")

    elif stain_method.lower() == "macenko":
        try:
            import staintools
            normalizer = staintools.StainNormalizer(method="macenko")
            normalizer.fit(ref_thumb)
            ref_stain_mat = _macenko_stain_matrix(ref_thumb)
            if ref_stain_mat is not None:
                print(f"  Reference stain matrix:\n"
                      f"    H: {np.round(ref_stain_mat[0], 4).tolist()}\n"
                      f"    E: {np.round(ref_stain_mat[1], 4).tolist()}")
        except ImportError:
            print("  WARNING: staintools not available — skipping normalization.")
    else:
        print("  Stain method = none — thumbnails shown without normalization.")

    rows = []
    vec_lines = [f"Reference: {reference_slide}"]
    if ref_stain_mat is not None:
        vec_lines += [
            f"  H: {ref_stain_mat[0].tolist()}",
            f"  E: {ref_stain_mat[1].tolist()}",
            "",
        ]

    print(f"\nProcessing {len(slide_paths)} slides...\n")

    for slide_path in slide_paths:
        name = slide_path.stem
        print(f"  {name}")

        orig = _load_thumbnail(str(slide_path))
        norm, changed = _apply_norm(orig, normalizer)

        if stain_method.lower() != "none" and not changed:
            print(f"    *** WARNING: normalization produced no change — silent failure! ***")

        pre_stats  = _lab_stats(orig)
        post_stats = _lab_stats(norm)

        row = {"slide": name, "norm_changed": changed}
        for k, v in pre_stats.items():
            row[f"pre_{k}"] = round(v, 3)
        for k, v in post_stats.items():
            row[f"post_{k}"] = round(v, 3)

        if stain_method.lower() == "macenko" and ref_stain_mat is not None:
            slide_mat = _macenko_stain_matrix(orig)
            if slide_mat is not None:
                h_ang = _angle_deg(slide_mat[0], ref_stain_mat[0])
                e_ang = _angle_deg(slide_mat[1], ref_stain_mat[1])
                row["H_angle_deg"] = round(h_ang, 2)
                row["E_angle_deg"] = round(e_ang, 2)
                row["stain_H"] = slide_mat[0].tolist()
                row["stain_E"] = slide_mat[1].tolist()
                flag = " ***" if (h_ang > 20 or e_ang > 20) else ""
                print(f"    Stain angles from ref — H: {h_ang:.1f}°  E: {e_ang:.1f}°{flag}")
                if flag:
                    print(f"    *** angle >20° — staining far from reference ***")
                vec_lines += [
                    f"Slide: {name}",
                    f"  H: {slide_mat[0].tolist()}",
                    f"  E: {slide_mat[1].tolist()}",
                    f"  H_angle: {h_ang:.2f}°",
                    f"  E_angle: {e_ang:.2f}°",
                    "",
                ]
            else:
                row["H_angle_deg"] = None
                row["E_angle_deg"] = None

        rows.append(row)
        _make_slide_figure(name, orig, norm, changed, stain_method, out)

    # Summary CSV (no list columns)
    csv_rows = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]
    df = pd.DataFrame(csv_rows)
    csv_path = out / "stain_qc_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Summary CSV: {csv_path}")

    # Macenko stain vector log
    if stain_method.lower() == "macenko":
        vec_path = out / "macenko_stain_vectors.txt"
        vec_path.write_text("\n".join(vec_lines) + "\n")
        print(f"  Stain vectors: {vec_path}")

        # Summary angle bar chart
        angle_rows = [r for r in rows if "H_angle_deg" in r and r["H_angle_deg"] is not None]
        if angle_rows:
            names_  = [r["slide"] for r in angle_rows]
            h_angs  = [r["H_angle_deg"] for r in angle_rows]
            e_angs  = [r["E_angle_deg"] for r in angle_rows]
            xs = np.arange(len(names_))
            w = 0.35

            fig, ax = plt.subplots(figsize=(max(10, len(names_) * 0.7), 5))
            h_cols = ["tomato" if a > 20 else "steelblue" for a in h_angs]
            e_cols = ["tomato" if a > 20 else "darkorange" for a in e_angs]
            ax.bar(xs - w/2, h_angs, w, color=h_cols, label="H angle")
            ax.bar(xs + w/2, e_angs, w, color=e_cols, label="E angle")
            ax.axhline(20, color="red", linestyle="--", linewidth=1, label="20° warning")
            ax.set_xticks(xs)
            ax.set_xticklabels(names_, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Angle from reference (degrees)")
            ax.set_title("Macenko stain vector angles — slides vs. reference")
            ax.legend()
            plt.tight_layout()
            angle_fig = out / "stain_angle_summary.png"
            plt.savefig(angle_fig, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Angle chart: {angle_fig}")

    # Print summary table
    print(f"\n  {'Slide':<35s} {'Changed':>8s} {'H°':>8s} {'E°':>8s}")
    print("  " + "-" * 62)
    for r in rows:
        h_str = f"{r['H_angle_deg']:>7.1f}" if r.get("H_angle_deg") is not None else "    N/A"
        e_str = f"{r['E_angle_deg']:>7.1f}" if r.get("E_angle_deg") is not None else "    N/A"
        chg = "YES" if r["norm_changed"] else "NO!"
        print(f"  {r['slide']:<35s} {chg:>8s} {h_str} {e_str}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QC Step 2: stain normalization check")
    parser.add_argument("--slides-dir", required=True, help="Directory of PNG slides")
    parser.add_argument("--stain-method", default="reinhard",
                        choices=["reinhard", "macenko", "none"])
    parser.add_argument("--output-dir", required=True, help="Root QC output directory")
    parser.add_argument("--reference-slide", default=None,
                        help="Reference slide path (default: first alphabetically)")
    args = parser.parse_args()
    run_stain_qc(args.slides_dir, args.stain_method, args.output_dir, args.reference_slide)
