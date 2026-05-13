"""
Master QC runner — runs all four diagnostic steps for a completed atlas run.

Derives file paths from --run-dir (the same directory that run_all.py writes
its results to).  Steps can be filtered with --steps.

Usage (from the project root):
    python cancer_trajectory_atlas/qc/run_qc.py \\
        --run-dir    /scratch/finnley1/results/atlas_full \\
        --slides-dir /scratch/finnley1/data/MCF7_x5_cropped \\
        --stain-method macenko

    # Run only steps 1 and 2:
    python cancer_trajectory_atlas/qc/run_qc.py \\
        --run-dir    /scratch/finnley1/results/atlas_full \\
        --slides-dir /scratch/finnley1/data/MCF7_x5_cropped \\
        --stain-method macenko \\
        --steps 12

Expected files in --run-dir:
    adata_full.h5ad
    results.csv
"""

import sys
import argparse
import time
from pathlib import Path

# Allow running as a plain script from inside the project directory.
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent   # cancer_trajectory_atlas/
if str(_PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT.parent))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qc.graph_connectivity    import check_graph_connectivity
from qc.stain_qc               import run_stain_qc
from qc.cluster_contact_sheet  import make_contact_sheets
from qc.pseudotime_by_slide    import plot_pseudotime_by_slide


def main():
    parser = argparse.ArgumentParser(
        description="Cancer Trajectory Atlas — QC diagnostics (steps 1–4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  (c) k-NN graph connectivity check
  2  (a) Stain normalization QC viewer
  3  (e) Cluster contact sheets
  4  (b) Per-slide / per-mouse pseudotime violins

Examples:
  python cancer_trajectory_atlas/qc/run_qc.py \\
      --run-dir /scratch/finnley1/results/atlas_full \\
      --slides-dir /scratch/finnley1/data/MCF7_x5_cropped \\
      --stain-method macenko

  # Re-run only steps 1 and 3 (fast checks, no slides needed for step 1):
  python cancer_trajectory_atlas/qc/run_qc.py \\
      --run-dir /scratch/finnley1/results/atlas_full \\
      --slides-dir /scratch/finnley1/data/MCF7_x5_cropped \\
      --stain-method macenko \\
      --steps 13
        """,
    )

    parser.add_argument("--run-dir",    required=True, type=Path,
                        help="Directory containing adata_full.h5ad and results.csv")
    parser.add_argument("--slides-dir", required=True, type=Path,
                        help="Directory containing the original .png slides")
    parser.add_argument("--stain-method", default="reinhard",
                        choices=["reinhard", "macenko", "none"],
                        help="Stain normalization method used in the run (default: reinhard)")
    parser.add_argument("--steps", default="1234",
                        help="Which steps to run, e.g. '12' for steps 1 and 2 (default: 1234)")
    parser.add_argument("--patch-size", type=int, default=112,
                        help="Patch size used during extraction (default: 112)")
    parser.add_argument("--n-contact-patches", type=int, default=25,
                        help="Random patches per cluster for contact sheets (default: 25)")
    parser.add_argument("--reference-slide", default=None,
                        help="Override reference slide for stain QC "
                             "(default: first slide alphabetically)")

    args = parser.parse_args()

    run_dir   = args.run_dir
    adata_path = run_dir / "adata_full.h5ad"
    results_csv = run_dir / "results.csv"
    qc_dir    = run_dir / "qc"

    # Validate required files
    missing = []
    if not run_dir.exists():
        missing.append(f"--run-dir not found: {run_dir}")
    if not args.slides_dir.exists():
        missing.append(f"--slides-dir not found: {args.slides_dir}")
    if not adata_path.exists():
        missing.append(f"  adata_full.h5ad not found in {run_dir}")
    if not results_csv.exists():
        missing.append(f"  results.csv not found in {run_dir}")
    if missing:
        for m in missing:
            print(f"ERROR: {m}")
        sys.exit(1)

    steps = set(args.steps)
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Cancer Trajectory Atlas — QC Diagnostics")
    print(f"{'='*60}")
    print(f"  Run dir:      {run_dir}")
    print(f"  Slides dir:   {args.slides_dir}")
    print(f"  Stain method: {args.stain_method}")
    print(f"  Steps:        {', '.join(sorted(steps))}")
    print(f"  QC output:    {qc_dir}")
    print(f"{'='*60}\n")

    if "1" in steps:
        print(f"\n{'='*60}")
        print("STEP 1 — k-NN Graph Connectivity Check")
        print(f"{'='*60}")
        check_graph_connectivity(str(adata_path), str(qc_dir))

    if "2" in steps:
        print(f"\n{'='*60}")
        print("STEP 2 — Stain Normalization QC")
        print(f"{'='*60}")
        run_stain_qc(
            slides_dir=str(args.slides_dir),
            stain_method=args.stain_method,
            output_dir=str(qc_dir),
            reference_slide=args.reference_slide,
        )

    if "3" in steps:
        print(f"\n{'='*60}")
        print("STEP 3 — Cluster Contact Sheets")
        print(f"{'='*60}")
        make_contact_sheets(
            results_csv=str(results_csv),
            slides_dir=str(args.slides_dir),
            output_dir=str(qc_dir),
            n_patches=args.n_contact_patches,
            patch_size=args.patch_size,
        )

    if "4" in steps:
        print(f"\n{'='*60}")
        print("STEP 4 — Per-slide / Per-mouse Pseudotime Violins")
        print(f"{'='*60}")
        plot_pseudotime_by_slide(
            adata_path=str(adata_path),
            results_csv=str(results_csv),
            output_dir=str(qc_dir),
        )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"QC complete in {elapsed:.1f}s")
    print(f"  Outputs: {qc_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
