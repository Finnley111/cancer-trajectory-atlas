"""
TIFF → PNG/JPEG Converter

Unchanged from original — standalone utility.

Usage:
    python -m cancer_trajectory_atlas.converters.tiff_to_img -i input.tiff -o output.png
"""

import argparse
import os
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def convert_tiff(input_path, output_path, format="png"):
    try:
        img = Image.open(input_path)
        if format == "png":
            img.convert("RGBA").save(output_path, "PNG")
        else:
            img.convert("RGB").save(output_path, "JPEG", quality=100, subsampling=0)
        print(f"  {Path(input_path).name} → {Path(output_path).name}")
    except Exception as e:
        print(f"  Failed: {Path(input_path).name} — {e}")


def convert_folder(input_folder, output_folder, format="png"):
    os.makedirs(output_folder, exist_ok=True)
    for f in Path(input_folder).glob("*.tif*"):
        out = Path(output_folder) / f"{f.stem}.{format}"
        convert_tiff(str(f), str(out), format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TIFF to PNG/JPEG")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-f", "--format", default="png", choices=["png", "jpeg"])
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        convert_folder(str(inp), args.output, args.format)
    else:
        convert_tiff(str(inp), args.output, args.format)
