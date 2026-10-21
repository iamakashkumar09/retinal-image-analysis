"""
main.py — Command-line / batch pipeline entry point.

Usage examples:

    # Analyse a single image
    python main.py --image data/DRIVE/training/images/001_training.tif

    # Analyse with ground-truth vessel mask (computes metrics)
    python main.py \
        --image data/DRIVE/training/images/001_training.tif \
        --fov   data/DRIVE/training/fov_masks/001_training_fov.png \
        --vessel-gt data/DRIVE/training/vessel_masks/001_training_vessel.png

    # Batch process all DRIVE training images
    python main.py --batch-drive

    # Batch process all IDRiD images
    python main.py --batch-idrid

    # Run the full experiment sweep (E1–E5)
    python main.py --experiment
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import analyze_retina, validate_image
from src.evaluation import calculate_metrics
from src.visualization import save_pipeline_figure, pipeline_grid
from src.report import generate_report


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as RGB uint8. Raises FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise ValueError(f"Could not decode image: {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_gray(path: str | Path | None) -> np.ndarray | None:
    """Load a grayscale mask. Returns None if path is None or missing."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img


def save_result(result: dict, stem: str, outputs_dir: Path) -> None:
    """Save vessel mask, exudate mask, and overlay images."""
    def _save(arr, subdir, fname):
        if arr is None:
            return
        out = outputs_dir / subdir
        out.mkdir(parents=True, exist_ok=True)
        if arr.ndim == 2:
            cv2.imwrite(str(out / fname), arr)
        else:
            cv2.imwrite(str(out / fname),
                        cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    _save(result.get("vessel_mask"),  "vessels",  f"{stem}_vessel.png")
    _save(result.get("exudate_mask"), "exudates", f"{stem}_exudate.png")
    _save(result.get("overlay"),      "overlays",  f"{stem}_overlay.png")

    # Save pipeline grid figure
    grid_path = outputs_dir / "preprocessed" / f"{stem}_pipeline.png"
    (outputs_dir / "preprocessed").mkdir(parents=True, exist_ok=True)
    try:
        pipeline_grid(result, grid_path)
    except Exception as e:
        print(f"  [warn] Could not save pipeline figure: {e}")


# ─────────────────────────────────────────────
# Single-image analysis
# ─────────────────────────────────────────────

def analyse_one(image_path: str,
                fov_path: str | None = None,
                vessel_gt_path: str | None = None,
                exudate_gt_path: str | None = None,
                disc_path: str | None = None,
                outputs_dir: Path = Path("outputs"),
                verbose: bool = True,
                **pipeline_kwargs) -> dict:
    """
    Analyse a single retinal image and optionally evaluate against GT.

    Returns:
        dict with 'result' (pipeline output) and 'metrics' (if GT provided).
    """
    image    = load_rgb(image_path)
    fov      = load_gray(fov_path)
    disc     = load_gray(disc_path)
    vessel_gt  = load_gray(vessel_gt_path)
    exudate_gt = load_gray(exudate_gt_path)

    ok, reason = validate_image(image)
    if not ok:
        raise ValueError(f"Invalid image: {reason}")

    if verbose:
        h, w = image.shape[:2]
        print(f"  Image: {Path(image_path).name}  [{w}×{h}]")

    result = analyze_retina(image, fov_mask=fov,
                              optic_disc_mask=disc, **pipeline_kwargs)

    stem = Path(image_path).stem
    save_result(result, stem, outputs_dir)

    metrics = {}
    if vessel_gt is not None:
        if vessel_gt.shape != result["vessel_mask"].shape:
            vessel_gt = cv2.resize(
                vessel_gt,
                (result["vessel_mask"].shape[1], result["vessel_mask"].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        metrics["vessel"] = calculate_metrics(
            result["vessel_mask"], vessel_gt, result.get("fov_mask")
        )
        if verbose:
            m = metrics["vessel"]
            print(f"    Vessel  → Dice:{m['dice']:.4f}  IoU:{m['iou']:.4f}"
                  f"  Sens:{m['sensitivity']:.4f}  Spec:{m['specificity']:.4f}")

    if exudate_gt is not None:
        if exudate_gt.shape != result["exudate_mask"].shape:
            exudate_gt = cv2.resize(
                exudate_gt,
                (result["exudate_mask"].shape[1], result["exudate_mask"].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        metrics["exudate"] = calculate_metrics(result["exudate_mask"], exudate_gt)
        if verbose:
            m = metrics["exudate"]
            print(f"    Exudate → Dice:{m['dice']:.4f}  IoU:{m['iou']:.4f}"
                  f"  Sens:{m['sensitivity']:.4f}  Spec:{m['specificity']:.4f}")

    if verbose:
        print(f"    Vessel density: {result['vessel_density']:.2f}%  "
              f"Candidates: {result['region_stats']['count']}")

    return {"result": result, "metrics": metrics, "stem": stem}


# ─────────────────────────────────────────────
# Batch DRIVE
# ─────────────────────────────────────────────

def batch_drive(outputs_dir: Path = Path("outputs"),
                experiments_dir: Path = Path("experiments"),
                **pipeline_kwargs) -> None:
    """Process all available DRIVE training images."""
    img_dir    = Path("data/DRIVE/training/images")
    fov_dir    = Path("data/DRIVE/training/fov_masks")
    gt_dir     = Path("data/DRIVE/training/vessel_masks")

    images = sorted(img_dir.glob("*.tif")) + sorted(img_dir.glob("*.png"))
    if not images:
        print("[DRIVE] No images found. Run: python generate_synthetic_data.py")
        return

    rows = []
    print(f"\n=== DRIVE Batch ({len(images)} images) ===")
    for img_path in images:
        stem    = img_path.stem
        fov_p   = next(fov_dir.glob(f"{stem}_fov*"), None)
        gt_p    = next(gt_dir.glob(f"{stem}_vessel*"), None)

        try:
            out = analyse_one(
                str(img_path), fov_path=str(fov_p) if fov_p else None,
                vessel_gt_path=str(gt_p) if gt_p else None,
                outputs_dir=outputs_dir, **pipeline_kwargs
            )
            row = {"image": stem,
                   "vessel_density": out["result"]["vessel_density"],
                   "num_candidates": out["result"]["region_stats"]["count"]}
            if "vessel" in out["metrics"]:
                row.update({f"vessel_{k}": v
                             for k, v in out["metrics"]["vessel"].items()})
            rows.append(row)
        except Exception as e:
            print(f"  [ERROR] {stem}: {e}")

    experiments_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(str(experiments_dir / "vessel_experiments.csv"), index=False)
    print(f"\n  Saved → experiments/vessel_experiments.csv")
    if "vessel_dice" in df.columns:
        print(f"  Mean vessel Dice: {df['vessel_dice'].mean():.4f}")


# ─────────────────────────────────────────────
# Batch IDRiD
# ─────────────────────────────────────────────

def batch_idrid(outputs_dir: Path = Path("outputs"),
                experiments_dir: Path = Path("experiments"),
                **pipeline_kwargs) -> None:
    """Process all available IDRiD images."""
    img_dir  = Path("data/IDRiD/images")
    ex_dir   = Path("data/IDRiD/exudate_masks")
    od_dir   = Path("data/IDRiD/optic_disc_masks")

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not images:
        print("[IDRiD] No images found. Run: python generate_synthetic_data.py")
        return

    rows = []
    print(f"\n=== IDRiD Batch ({len(images)} images) ===")
    for img_path in images:
        stem   = img_path.stem
        ex_p   = next(ex_dir.glob(f"{stem}_EX*"), None)
        od_p   = next(od_dir.glob(f"{stem}_OD*"), None)

        try:
            out = analyse_one(
                str(img_path),
                disc_path=str(od_p) if od_p else None,
                exudate_gt_path=str(ex_p) if ex_p else None,
                outputs_dir=outputs_dir, **pipeline_kwargs
            )
            row = {"image": stem,
                   "vessel_density": out["result"]["vessel_density"],
                   "num_candidates": out["result"]["region_stats"]["count"]}
            if "exudate" in out["metrics"]:
                row.update({f"exudate_{k}": v
                             for k, v in out["metrics"]["exudate"].items()})
            rows.append(row)
        except Exception as e:
            print(f"  [ERROR] {stem}: {e}")

    experiments_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(str(experiments_dir / "exudate_experiments.csv"), index=False)
    print(f"\n  Saved → experiments/exudate_experiments.csv")
    if "exudate_dice" in df.columns:
        print(f"  Mean exudate Dice: {df['exudate_dice'].mean():.4f}")


# ─────────────────────────────────────────────
# Experiment sweep (E1–E5)
# ─────────────────────────────────────────────

EXPERIMENTS = [
    {"id": "E1", "clahe": False, "gabor": False, "morphology": False, "fov": False,
     "desc": "Simple intensity baseline"},
    {"id": "E2", "clahe": True,  "gabor": False, "morphology": False, "fov": False,
     "desc": "+ Local contrast (CLAHE)"},
    {"id": "E3", "clahe": True,  "gabor": True,  "morphology": False, "fov": False,
     "desc": "+ Oriented Gabor filtering"},
    {"id": "E4", "clahe": True,  "gabor": True,  "morphology": True,  "fov": False,
     "desc": "+ Morphological cleanup"},
    {"id": "E5", "clahe": True,  "gabor": True,  "morphology": True,  "fov": True,
     "desc": "Full pipeline (+ FOV mask)"},
]


def run_experiment_sweep(outputs_dir: Path = Path("outputs"),
                          experiments_dir: Path = Path("experiments")) -> None:
    """
    Run vessel segmentation experiments E1–E5 on all available DRIVE images.
    Each experiment adds one DIP stage to demonstrate its measurable effect.
    """
    img_dir = Path("data/DRIVE/training/images")
    fov_dir = Path("data/DRIVE/training/fov_masks")
    gt_dir  = Path("data/DRIVE/training/vessel_masks")

    images = sorted(img_dir.glob("*.tif")) + sorted(img_dir.glob("*.png"))
    if not images:
        print("[Experiment] No DRIVE images. Run generate_synthetic_data.py first.")
        return

    all_rows = []
    print("\n=== Experiment Sweep (E1–E5) ===")
    for exp in EXPERIMENTS:
        print(f"\n  [{exp['id']}] {exp['desc']}")
        for img_path in images:
            stem = img_path.stem
            fov_p = next(fov_dir.glob(f"{stem}_fov*"), None)
            gt_p  = next(gt_dir.glob(f"{stem}_vessel*"), None)

            # Build kwargs based on experiment flags
            kwargs = {}
            if not exp["clahe"]:
                kwargs["clahe_clip"] = 0.01   # effectively no CLAHE
            if not exp["gabor"]:
                kwargs["n_orientations"] = 1
                kwargs["gabor_sigma"]    = 1
                kwargs["gabor_lambda"]   = 1
            if not exp["morphology"]:
                kwargs["vessel_morph_size"] = 1
            fov_to_use = str(fov_p) if (exp["fov"] and fov_p) else None

            try:
                out = analyse_one(
                    str(img_path),
                    fov_path=fov_to_use,
                    vessel_gt_path=str(gt_p) if gt_p else None,
                    outputs_dir=outputs_dir,
                    verbose=False, **kwargs
                )
                row = {
                    "experiment_id": exp["id"],
                    "description":   exp["desc"],
                    "image":         stem,
                    "vessel_density": out["result"]["vessel_density"],
                }
                if "vessel" in out["metrics"]:
                    row.update({f"vessel_{k}": v
                                 for k, v in out["metrics"]["vessel"].items()})
                all_rows.append(row)
                if "vessel" in out["metrics"]:
                    m = out["metrics"]["vessel"]
                    print(f"    {stem}: Dice={m['dice']:.4f}")
            except Exception as e:
                print(f"    [ERROR] {stem}: {e}")

    experiments_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(str(experiments_dir / "results.csv"), index=False)
    print(f"\n  Experiment results saved → experiments/results.csv")

    # Print summary
    if "vessel_dice" in df.columns:
        summary = df.groupby("experiment_id")["vessel_dice"].mean()
        print("\n  Dice summary per experiment:")
        print(summary.to_string())


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Retinal DIP — Command-line pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--image", help="Path to a single retinal image")
    p.add_argument("--fov",   help="Path to FOV mask")
    p.add_argument("--vessel-gt",  help="Path to vessel ground-truth mask")
    p.add_argument("--exudate-gt", help="Path to exudate ground-truth mask")
    p.add_argument("--disc",  help="Path to optic-disc mask")
    p.add_argument("--batch-drive",  action="store_true",
                   help="Batch-process all DRIVE training images")
    p.add_argument("--batch-idrid", action="store_true",
                   help="Batch-process all IDRiD images")
    p.add_argument("--experiment", action="store_true",
                   help="Run experiment sweep E1–E5")
    p.add_argument("--report", action="store_true",
                   help="Generate PDF report for single image")
    p.add_argument("--outputs", default="outputs",
                   help="Output directory (default: outputs/)")
    p.add_argument("--experiments-dir", default="experiments",
                   help="Experiments directory (default: experiments/)")
    # Pipeline parameters
    p.add_argument("--gabor-sigma",  type=float, default=4.0)
    p.add_argument("--gabor-lambda", type=float, default=10.0)
    p.add_argument("--gabor-gamma",  type=float, default=0.5)
    p.add_argument("--orientations", type=int,   default=12)
    p.add_argument("--tophat-size",  type=int,   default=15)
    p.add_argument("--min-area",     type=int,   default=20)
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    outputs_dir = Path(args.outputs)
    exp_dir     = Path(args.experiments_dir)

    pipeline_kwargs = dict(
        gabor_sigma=args.gabor_sigma,
        gabor_lambda=args.gabor_lambda,
        gabor_gamma=args.gabor_gamma,
        n_orientations=args.orientations,
        tophat_kernel_size=args.tophat_size,
        min_area=args.min_area,
    )

    if args.image:
        print(f"\nAnalysing: {args.image}")
        out = analyse_one(
            args.image,
            fov_path=args.fov,
            vessel_gt_path=args.vessel_gt,
            exudate_gt_path=args.exudate_gt,
            disc_path=args.disc,
            outputs_dir=outputs_dir,
            **pipeline_kwargs,
        )
        if args.report:
            report_path = outputs_dir / "reports" / f"{out['stem']}_report.pdf"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            metrics = next(iter(out["metrics"].values()), None) if out["metrics"] else None
            generate_report(out["result"], report_path,
                            image_id=out["stem"], metrics=metrics)
            print(f"  PDF report saved → {report_path}")

    elif args.batch_drive:
        batch_drive(outputs_dir, exp_dir, **pipeline_kwargs)

    elif args.batch_idrid:
        batch_idrid(outputs_dir, exp_dir, **pipeline_kwargs)

    elif args.experiment:
        run_experiment_sweep(outputs_dir, exp_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
