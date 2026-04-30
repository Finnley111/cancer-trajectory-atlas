"""
NDPI → PNG/JPEG Converter

Converts NDPI (and other WSI formats) using OpenSlide.
Unchanged from original pipeline — this is a standalone utility.

Usage:
    python -m cancer_trajectory_atlas.converters.ndpi_to_img -i input.ndpi -o output.png
"""

import argparse
import os
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def convert_ndpi(input_path, output_path, format="png", level=0, scale=1.0):
    """Convert a single NDPI file to PNG or JPEG."""
    import openslide

    print(f"Opening: {Path(input_path).name}")
    slide = openslide.OpenSlide(input_path)
    dims = slide.level_dimensions[level]
    print(f"  Level {level}: {dims[0]} x {dims[1]}")

    img = slide.read_region((0, 0), level, dims).convert("RGB")

    if scale != 1.0:
        new_size = (int(dims[0] * scale), int(dims[1] * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    if format == "png":
        img.save(output_path, "PNG", compress_level=6)
    else:
        img.save(output_path, "JPEG", quality=95, subsampling=0)

    slide.close()
    print(f"  Saved: {output_path}")


def convert_folder(input_folder, output_folder, format="png", level=0, scale=1.0):
    os.makedirs(output_folder, exist_ok=True)
    for f in Path(input_folder).glob("*.ndpi"):
        out = Path(output_folder) / f"{f.stem}.{format}"
        convert_ndpi(str(f), str(out), format, level, scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NDPI to PNG/JPEG")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-f", "--format", default="png", choices=["png", "jpeg"])
    parser.add_argument("-l", "--level", type=int, default=0)
    parser.add_argument("-s", "--scale", type=float, default=1.0)
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        convert_folder(str(inp), args.output, args.format, args.level, args.scale)
    else:
        convert_ndpi(str(inp), args.output, args.format, args.level, args.scale)
