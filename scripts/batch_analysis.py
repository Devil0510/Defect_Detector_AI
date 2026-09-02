#!/usr/bin/env python3
"""
AI-Based Steel Surface Defect Severity Estimation System
Batch Analysis Orchestrator

Provides interactive and command-line interfaces for multi-image batch inference.
Performs defect detection, ROI segmentation, quantitative damage feature extraction,
dual geometric/neural severity estimation, epistemic uncertainty quantification,
conformal prediction intervals, Grad-CAM visualization, per-specimen reports,
and master batch-level summaries.
"""

import os
import sys
import json
import time
import shutil
import argparse
import datetime
import platform
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

# Add root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from scripts.run_pipeline import (
    load_pipeline_models,
    process_single_image,
    select_device
)

logger = get_logger("batch_analysis")

SUPPORTED_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

def discover_images(
    input_dir: Path,
    recursive: bool = False
) -> List[Path]:
    """
    Discovers all supported image files in input_dir.
    Filters out unsupported files, directories, and hidden files.
    Returns alphabetically sorted list of image Paths for deterministic processing.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        return []

    images = []
    if recursive:
        for p in input_dir.rglob("*"):
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(p)
    else:
        for p in input_dir.iterdir():
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(p)

    return sorted(images, key=lambda x: str(x).lower())


def get_unique_specimen_dir(
    output_dir: Path,
    image_path: Path,
    allocated_ids: Set[str]
) -> Tuple[str, Path]:
    """
    Generates a deterministic collision-safe specimen directory name.
    Example: sample.jpg -> sample/, sample.png -> sample_2/
    """
    base_stem = image_path.stem
    # Sanitize base stem
    clean_stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_stem)
    if not clean_stem:
        clean_stem = "specimen"

    specimen_id = clean_stem
    counter = 2
    while specimen_id in allocated_ids or (output_dir / specimen_id).exists() and not (output_dir / specimen_id / "specimen_summary.json").exists():
        # Check if the existing folder belongs to another run/image
        if specimen_id not in allocated_ids:
            # First time seeing this in current run
            allocated_ids.add(specimen_id)
            return specimen_id, output_dir / specimen_id
        specimen_id = f"{clean_stem}_{counter}"
        counter += 1

    allocated_ids.add(specimen_id)
    specimen_dir = output_dir / specimen_id
    return specimen_id, specimen_dir


def check_write_permission(directory: Path) -> bool:
    """Tests write permission in the target directory."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        test_file = directory / f".write_test_{int(time.time()*1000)}"
        test_file.touch()
        test_file.unlink()
        return True
    except Exception as e:
        logger.error(f"Directory write check failed on {directory}: {e}")
        return False


def run_interactive_mode() -> Tuple[Path, Path, bool]:
    """
    Executes Interactive Mode (Mode 1), prompting the user with ASCII header,
    path validation, image discovery summary, and confirmation prompt.
    """
    print("=" * 60)
    print("AI SURFACE DEFECT SEVERITY ESTIMATION SYSTEM")
    print("BATCH ANALYSIS MODE")
    print("=" * 60)
    print()

    # 1. Prompt and validate input directory
    while True:
        raw_in = input("Enter path to input image folder:\n> ").strip()
        if not raw_in:
            print("❌ Input path cannot be empty. Please enter a valid directory path.\n")
            continue
        in_path = Path(raw_in).resolve()
        if not in_path.exists():
            print(f"❌ Path does not exist: {in_path}. Please try again.\n")
            continue
        if not in_path.is_dir():
            print(f"❌ Path is not a directory: {in_path}. Please try again.\n")
            continue

        images = discover_images(in_path)
        if not images:
            print(f"❌ No supported images found in: {in_path}")
            print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}\n")
            continue
        break

    # 2. Prompt and validate output directory
    while True:
        raw_out = input("\nEnter path where results should be saved:\n> ").strip()
        if not raw_out:
            print("❌ Output path cannot be empty. Please enter a valid directory path.\n")
            continue
        out_path = Path(raw_out).resolve()
        if not check_write_permission(out_path):
            print(f"❌ Permission denied or cannot write to: {out_path}. Please try again.\n")
            continue
        break

    # 3. Summary & Confirmation
    print(f"\nFound {len(images)} supported images in '{in_path}'.")
    proceed = input("Proceed with batch analysis? [Y/N]: ").strip().lower()
    if proceed not in ("y", "yes"):
        print("\nBatch analysis cancelled by user.")
        sys.exit(0)

    return in_path, out_path, False


def process_batch(
    input_dir: Path,
    output_dir: Path,
    detector_weights: Optional[Path] = None,
    neural_severity_weights: Optional[Path] = None,
    conformal_calibrator_path: Optional[Path] = None,
    temperature_scaler_path: Optional[Path] = None,
    use_clahe: bool = False,
    device_name: Optional[str] = None,
    conf_threshold: float = 0.20,
    recursive: bool = False
) -> Dict[str, Any]:
    """
    Core Batch Processing Pipeline.
    Loads models ONCE, iterates over all discovered images, saves individual specimen
    results and builds comprehensive master summary tables.
    """
    start_time = datetime.datetime.now()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover images
    images = discover_images(input_dir, recursive=recursive)
    total_images = len(images)
    if total_images == 0:
        logger.warning(f"No supported images found in {input_dir}")
        print(f"⚠️ No supported images found in {input_dir}")
        return {
            "total_images": 0,
            "successful_images": 0,
            "failed_images": 0,
            "total_defects": 0
        }

    # 2. Resolve default model paths if not supplied
    if detector_weights is None:
        if (PROJECT_ROOT / "models/detector_yolo11m.pt").exists():
            detector_weights = PROJECT_ROOT / "models/detector_yolo11m.pt"
        else:
            detector_weights = PROJECT_ROOT / "models/detector_yolo.pt"

    if neural_severity_weights is None:
        neural_severity_weights = PROJECT_ROOT / "models/severity_efficientnet_v2.pt"

    if conformal_calibrator_path is None:
        conformal_calibrator_path = PROJECT_ROOT / "models/conformal_calibrator.json"

    if temperature_scaler_path is None:
        temperature_scaler_path = PROJECT_ROOT / "models/temperature_scaler.json"

    # 3. Load Models ONCE
    print("\n" + "=" * 60)
    print("INITIALIZING INFERENCE PIPELINE MODELS...")
    print("=" * 60)
    models = load_pipeline_models(
        detector_weights=detector_weights,
        neural_severity_weights=neural_severity_weights,
        conformal_calibrator_path=conformal_calibrator_path,
        temperature_scaler_path=temperature_scaler_path,
        use_clahe=use_clahe,
        device_name=device_name,
        conf_threshold=conf_threshold
    )

    print(f"\nStarting batch analysis of {total_images} images...")
    print(f"Output directory: {output_dir}\n")
    print("-" * 60)

    # Containers for reporting
    allocated_ids: Set[str] = set()
    all_master_defects: List[Dict[str, Any]] = []
    all_specimen_summaries: List[Dict[str, Any]] = []
    processing_errors: List[Dict[str, Any]] = []

    successful_count = 0
    failed_count = 0
    total_defects_count = 0
    human_triage_specimens_count = 0

    # 4. Batch Processing Loop
    for idx, img_path in enumerate(images, 1):
        specimen_id, specimen_dir = get_unique_specimen_dir(output_dir, img_path, allocated_ids)
        diagnostics_dir = specimen_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{total_images}] Processing {img_path.name}")

        try:
            # Process single image with pre-loaded models
            defect_records = process_single_image(
                image_path=img_path,
                pipeline_models=models,
                output_dir=diagnostics_dir,
                specimen_id=specimen_id,
                quiet=True
            )

            # Analyze defect records
            is_nominal = (len(defect_records) == 1 and defect_records[0].get("defect_class") == "Clean_Surface")
            defects_detected = 0 if is_nominal else len(defect_records)
            total_defects_count += defects_detected

            specimen_triage = any(d.get("requires_human_triage", False) for d in defect_records)
            if specimen_triage:
                human_triage_specimens_count += 1

            # Format defects for JSON export
            formatted_defects = []
            if not is_nominal:
                for d in defect_records:
                    d_dict = {
                        "defect_id": d["defect_index"],
                        "predicted_class": d["defect_class"],
                        "detector_confidence": d["detector_confidence"],
                        "bounding_box": [d["bbox_xmin"], d["bbox_ymin"], d["bbox_xmax"], d["bbox_ymax"]],
                        "defect_area_px": d["defect_area_px"],
                        "defect_area_percentage": d["area_ratio_pct"],
                        "aspect_ratio": d["aspect_ratio"],
                        "circularity": d["circularity"],
                        "local_contrast": d["local_contrast"],
                        "glcm_contrast": d["glcm_contrast"],
                        "geometric_severity": d["severity_score"],
                        "geometric_severity_grade": d["severity_grade"],
                        "neural_severity_mean": d["neural_severity_mean"],
                        "neural_severity_std": d["neural_uncertainty_std"],
                        "conformal_interval_lower": d["ci_95_lower"],
                        "conformal_interval_upper": d["ci_95_upper"],
                        "conformal_interval_width": d["conformal_interval_width"],
                        "qc_status": d["qc_status"],
                        "human_triage_required": d["requires_human_triage"],
                        "diagnostic_report_path": d["diagnostic_report_path"]
                    }
                    formatted_defects.append(d_dict)
                    all_master_defects.append(d)

            # Build specimen summary JSON
            specimen_json = {
                "specimen_id": specimen_id,
                "original_filename": img_path.name,
                "source_image": str(img_path.resolve()),
                "processing_timestamp": datetime.datetime.now().isoformat(),
                "processing_status": "SUCCESS",
                "total_defects_detected": defects_detected,
                "requires_human_inspection": specimen_triage,
                "defects": formatted_defects
            }

            json_path = specimen_dir / "specimen_summary.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(specimen_json, f, indent=4)

            # Build specimen summary CSV
            csv_path = specimen_dir / "specimen_summary.csv"
            specimen_df = pd.DataFrame(defect_records)
            specimen_df.to_csv(csv_path, index=False)

            # Compute aggregate metrics for SPECIMEN_SUMMARY.csv
            if defects_detected > 0:
                avg_geom = float(np.mean([d["severity_score"] for d in defect_records]))
                max_geom = float(np.max([d["severity_score"] for d in defect_records]))
                avg_neural = float(np.mean([d["neural_severity_mean"] for d in defect_records]))
                max_neural = float(np.max([d["neural_severity_mean"] for d in defect_records]))
                classes = [d["defect_class"] for d in defect_records]
                dom_class = max(set(classes), key=classes.count)
                qc_summary = "HUMAN_REVIEW_REQUIRED" if specimen_triage else (
                    "MODERATE_CONFIDENCE" if any(d.get("qc_status") == "MODERATE_CONFIDENCE" for d in defect_records) else "HIGH_CONFIDENCE"
                )
            else:
                avg_geom, max_geom, avg_neural, max_neural = 0.0, 0.0, 0.0, 0.0
                dom_class = "None (Clean Surface)"
                qc_summary = "HIGH_CONFIDENCE"

            specimen_row = {
                "specimen_id": specimen_id,
                "original_filename": img_path.name,
                "source_path": str(img_path.resolve()),
                "total_defects_detected": defects_detected,
                "processing_status": "SUCCESS",
                "dominant_defect_class": dom_class,
                "average_geometric_severity": round(avg_geom, 2),
                "maximum_geometric_severity": round(max_geom, 2),
                "average_neural_severity": round(avg_neural, 2),
                "maximum_neural_severity": round(max_neural, 2),
                "overall_qc_status": qc_summary,
                "requires_human_inspection": specimen_triage,
                "specimen_dir": str(specimen_dir.resolve())
            }
            all_specimen_summaries.append(specimen_row)

            successful_count += 1

            status_icon = "⚠️" if specimen_triage else "✅"
            print(f"Status: SUCCESS | Defects detected: {defects_detected} | QC: {status_icon} {qc_summary}")
            print("-" * 60)

        except Exception as e:
            failed_count += 1
            err_msg = str(e)
            logger.error(f"Error processing {img_path.name}: {err_msg}", exc_info=True)
            print(f"Status: ❌ FAILED ({type(e).__name__}: {err_msg})")
            print("-" * 60)

            # Record error
            processing_errors.append({
                "filename": img_path.name,
                "image_path": str(img_path.resolve()),
                "error_type": type(e).__name__,
                "error_message": err_msg,
                "timestamp": datetime.datetime.now().isoformat()
            })

            # Record failed entry in specimen summary
            all_specimen_summaries.append({
                "specimen_id": specimen_id,
                "original_filename": img_path.name,
                "source_path": str(img_path.resolve()),
                "total_defects_detected": 0,
                "processing_status": f"FAILED: {type(e).__name__}",
                "dominant_defect_class": "ERROR",
                "average_geometric_severity": 0.0,
                "maximum_geometric_severity": 0.0,
                "average_neural_severity": 0.0,
                "maximum_neural_severity": 0.0,
                "overall_qc_status": "ERROR",
                "requires_human_inspection": True,
                "specimen_dir": str(specimen_dir.resolve())
            })

    # 5. Write Master CSV Files
    # 5A. MASTER_SUMMARY.csv (defect-level)
    master_csv_path = output_dir / "MASTER_SUMMARY.csv"
    if all_master_defects:
        master_df = pd.DataFrame(all_master_defects)
        # Standardize column naming for master summary
        col_rename = {
            "image_name": "original_filename",
            "defect_index": "defect_id",
            "defect_class": "predicted_class",
            "severity_score": "geometric_severity",
            "severity_grade": "geometric_grade",
            "neural_uncertainty_std": "neural_severity_std",
            "ci_95_lower": "conformal_lower",
            "ci_95_upper": "conformal_upper",
            "requires_human_triage": "human_triage_required"
        }
        master_df = master_df.rename(columns=col_rename)
        desired_cols = [
            "specimen_id", "original_filename", "defect_id", "predicted_class",
            "detector_confidence", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
            "defect_area_px", "area_ratio_pct", "aspect_ratio", "circularity",
            "local_contrast", "glcm_contrast", "geometric_severity", "geometric_grade",
            "neural_severity_mean", "neural_severity_std", "conformal_lower", "conformal_upper",
            "conformal_interval_width", "qc_status", "human_triage_required", "diagnostic_report_path"
        ]
        available_cols = [c for c in desired_cols if c in master_df.columns]
        master_df = master_df[available_cols]
        master_df.to_csv(master_csv_path, index=False)
    else:
        # Write empty template with headers
        headers = [
            "specimen_id", "original_filename", "defect_id", "predicted_class",
            "detector_confidence", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
            "defect_area_px", "area_ratio_pct", "aspect_ratio", "circularity",
            "local_contrast", "glcm_contrast", "geometric_severity", "geometric_grade",
            "neural_severity_mean", "neural_severity_std", "conformal_lower", "conformal_upper",
            "conformal_interval_width", "qc_status", "human_triage_required", "diagnostic_report_path"
        ]
        pd.DataFrame(columns=headers).to_csv(master_csv_path, index=False)

    # 5B. SPECIMEN_SUMMARY.csv (specimen-level)
    specimen_summary_csv = output_dir / "SPECIMEN_SUMMARY.csv"
    specimen_master_df = pd.DataFrame(all_specimen_summaries)
    specimen_master_df.to_csv(specimen_summary_csv, index=False)

    # 5C. PROCESSING_ERRORS.csv (if any errors occurred)
    if processing_errors:
        errors_csv = output_dir / "PROCESSING_ERRORS.csv"
        pd.DataFrame(processing_errors).to_csv(errors_csv, index=False)

    # 6. BATCH_METADATA.json (Reproducibility)
    end_time = datetime.datetime.now()
    batch_metadata = {
        "processing_timestamp": start_time.isoformat(),
        "completion_timestamp": end_time.isoformat(),
        "elapsed_seconds": round((end_time - start_time).total_seconds(), 2),
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "total_images_discovered": total_images,
        "successful_images": successful_count,
        "failed_images": failed_count,
        "total_defects_detected": total_defects_count,
        "specimens_requiring_triage": human_triage_specimens_count,
        "device_used": str(models["device"]),
        "detector_weights": str(detector_weights) if detector_weights else "N/A",
        "neural_severity_model": str(neural_severity_weights) if neural_severity_weights else "N/A",
        "conformal_calibrator": str(conformal_calibrator_path) if conformal_calibrator_path else "N/A",
        "temperature_scaler": str(temperature_scaler_path) if temperature_scaler_path else "N/A",
        "use_clahe": use_clahe,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "platform": platform.platform()
    }

    metadata_path = output_dir / "BATCH_METADATA.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(batch_metadata, f, indent=4)

    # 7. Print Final Batch Summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Images discovered        : {total_images}")
    print(f"Successfully processed   : {successful_count}")
    print(f"Failed                   : {failed_count}")
    print(f"Total defects detected   : {total_defects_count}")
    print(f"Human triage recommended : {human_triage_specimens_count} specimen(s)")
    print(f"Master Defect CSV        : {master_csv_path.resolve()}")
    print(f"Specimen Summary CSV     : {specimen_summary_csv.resolve()}")
    print(f"Batch Metadata JSON      : {metadata_path.resolve()}")
    if processing_errors:
        print(f"Processing Errors CSV    : {(output_dir / 'PROCESSING_ERRORS.csv').resolve()}")
    print(f"Results saved to         : {output_dir.resolve()}")
    print("=" * 60 + "\n")

    return batch_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference & severity estimation for steel surface defect inspection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input_dir", type=str, default=None, help="Path to input directory containing defect images")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Path to directory where batch results will be saved")
    parser.add_argument("--detector_weights", type=str, default=None, help="Path to trained YOLO detector weights (.pt)")
    parser.add_argument("--neural_severity", type=str, default=None, help="Path to trained EfficientNet-V2 severity weights (.pt)")
    parser.add_argument("--conformal_calibrator", type=str, default=None, help="Path to conformal calibration JSON")
    parser.add_argument("--temperature_scaler", type=str, default=None, help="Path to temperature scaler JSON")
    parser.add_argument("--use_clahe", action="store_true", help="Apply CLAHE contrast amplification")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', or auto)")
    parser.add_argument("--conf_threshold", type=float, default=0.20, help="YOLO detection confidence threshold")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively search subdirectories for images")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt in CLI mode")

    args = parser.parse_args()

    # Determine execution mode
    if args.input_dir is None or args.output_dir is None:
        # Mode 1: Interactive Mode
        in_path, out_path, recursive = run_interactive_mode()
    else:
        # Mode 2: Command Line Mode
        in_path = Path(args.input_dir).resolve()
        out_path = Path(args.output_dir).resolve()
        recursive = args.recursive

        if not in_path.exists() or not in_path.is_dir():
            print(f"❌ Error: Input path '{in_path}' does not exist or is not a directory.")
            sys.exit(1)

        images = discover_images(in_path, recursive=recursive)
        if not images:
            print(f"❌ Error: No supported images found in '{in_path}'.")
            sys.exit(1)

        if not check_write_permission(out_path):
            print(f"❌ Error: Permission denied to write in output path '{out_path}'.")
            sys.exit(1)

        if not args.yes:
            print(f"Found {len(images)} supported images in '{in_path}'.")
            proceed = input("Proceed with batch analysis? [Y/N]: ").strip().lower()
            if proceed not in ("y", "yes"):
                print("Batch analysis cancelled by user.")
                sys.exit(0)

    # Execute Batch Processing
    process_batch(
        input_dir=in_path,
        output_dir=out_path,
        detector_weights=Path(args.detector_weights) if args.detector_weights else None,
        neural_severity_weights=Path(args.neural_severity) if args.neural_severity else None,
        conformal_calibrator_path=Path(args.conformal_calibrator) if args.conformal_calibrator else None,
        temperature_scaler_path=Path(args.temperature_scaler) if args.temperature_scaler else None,
        use_clahe=args.use_clahe,
        device_name=args.device,
        conf_threshold=args.conf_threshold,
        recursive=recursive
    )


if __name__ == "__main__":
    main()
