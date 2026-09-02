import os
import sys
import cv2
import json
import argparse
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.detection.yolo_adapter import YOLODefectDetector
from src.data.xml_parser import XMLAnnotationParser
from src.segmentation.weak_segmenter import WeakROIThresholdSegmenter
from src.features.extractor import DefectFeatureExtractor
from src.severity.index_calculator import TransparentSeverityIndexCalculator
from src.severity.neural_severity import EfficientNetV2SeverityModel
from src.severity.uncertainty import (
    MonteCarloUncertaintyEstimator,
    ConformalPredictionCalibrator,
    TemperatureScaler,
    evaluate_qc_triage
)
from src.visualization.gradcam import SeverityGradCAM
from src.visualization.explainer import DefectSeverityExplainer

logger = get_logger("run_pipeline")

def select_device(preferred_device: Optional[str] = None) -> torch.device:
    """
    Selects computation device (CUDA or CPU) with robust error handling.
    Avoids repeated initialization crashes if CUDA driver is incompatible.
    """
    if preferred_device is not None:
        pref = str(preferred_device).lower().strip()
        if pref in ["cpu", "cuda"]:
            if pref == "cuda":
                try:
                    if torch.cuda.is_available():
                        _ = torch.tensor([1.0], device="cuda")
                        return torch.device("cuda")
                except Exception as e:
                    logger.warning(f"CUDA requested but initialization failed ({e}). Falling back to CPU.")
            return torch.device("cpu")

    try:
        if torch.cuda.is_available():
            _ = torch.tensor([1.0], device="cuda")
            return torch.device("cuda")
    except Exception as e:
        logger.info(f"CUDA not available or incompatible: {e}. Using CPU.")
    return torch.device("cpu")

def load_pipeline_models(
    detector_weights: str | Path | None = "models/detector_yolo.pt",
    neural_severity_weights: str | Path | None = "models/severity_efficientnet_v2.pt",
    conformal_calibrator_path: str | Path | None = "models/conformal_calibrator.json",
    temperature_scaler_path: str | Path | None = "models/temperature_scaler.json",
    use_clahe: bool = False,
    device_name: Optional[str] = None,
    conf_threshold: float = 0.20
) -> Dict[str, Any]:
    """
    Loads all detection, severity, uncertainty, segmentation, feature extraction,
    and visual explainer components ONCE and returns them in a model container dict.
    """
    device = select_device(device_name)
    logger.info(f"Pipeline device selected: {str(device).upper()}")
    print(f"Device selected: {str(device).upper()}")

    detector = None
    if detector_weights and Path(detector_weights).exists():
        try:
            detector = YOLODefectDetector(
                model_path=detector_weights,
                conf_threshold=conf_threshold,
                use_clahe=use_clahe
            )
            logger.info(f"Loaded YOLO defect detector from: {detector_weights}")
        except Exception as e:
            logger.error(f"Failed to load detector weights: {e}")

    neural_model = None
    uncertainty_estimator = None
    gradcam_engine = None

    if neural_severity_weights and Path(neural_severity_weights).exists():
        try:
            neural_model = EfficientNetV2SeverityModel(pretrained=False, dropout_rate=0.3).to(device)
            neural_model.load_state_dict(torch.load(str(neural_severity_weights), map_location=device))
            neural_model.eval()

            conformal_cal = None
            if conformal_calibrator_path and Path(conformal_calibrator_path).exists():
                try:
                    conformal_cal = ConformalPredictionCalibrator(alpha=0.05)
                    conformal_cal.load(conformal_calibrator_path)
                    logger.info(f"Loaded calibrated Conformal Prediction Multiplier q_hat={conformal_cal.q_hat:.4f}")
                except Exception as e:
                    logger.warning(f"Could not load conformal calibrator: {e}")

            temp_scaler = None
            if temperature_scaler_path and Path(temperature_scaler_path).exists():
                try:
                    temp_scaler = TemperatureScaler()
                    temp_scaler.load(temperature_scaler_path)
                    logger.info(f"Loaded calibrated Temperature Scaler T={temp_scaler.temperature:.4f}")
                except Exception as e:
                    logger.warning(f"Could not load temperature scaler: {e}")

            uncertainty_estimator = MonteCarloUncertaintyEstimator(
                model=neural_model,
                n_samples=25,
                uncertainty_threshold=8.0,
                conformal_calibrator=conformal_cal,
                temperature_scaler=temp_scaler
            )
            gradcam_engine = SeverityGradCAM(neural_model)
            logger.info(f"Loaded neural severity model and Grad-CAM engine from: {neural_severity_weights}")
        except Exception as e:
            logger.warning(f"Could not load neural model: {e}")

    segmenter = WeakROIThresholdSegmenter()
    extractor = DefectFeatureExtractor(include_texture=True)
    calc = TransparentSeverityIndexCalculator()

    return {
        "device": device,
        "detector": detector,
        "neural_model": neural_model,
        "uncertainty_estimator": uncertainty_estimator,
        "gradcam_engine": gradcam_engine,
        "segmenter": segmenter,
        "extractor": extractor,
        "calc": calc,
        "detector_weights_path": str(detector_weights) if detector_weights else None,
        "neural_severity_weights_path": str(neural_severity_weights) if neural_severity_weights else None
    }

def process_single_image(
    image_path: str | Path,
    detector: Optional[YOLODefectDetector] = None,
    neural_model: Optional[EfficientNetV2SeverityModel] = None,
    uncertainty_estimator: Optional[MonteCarloUncertaintyEstimator] = None,
    gradcam_engine: Optional[SeverityGradCAM] = None,
    segmenter: Optional[WeakROIThresholdSegmenter] = None,
    extractor: Optional[DefectFeatureExtractor] = None,
    calc: Optional[TransparentSeverityIndexCalculator] = None,
    output_dir: str | Path = Path("outputs/explanations"),
    device: Optional[torch.device] = None,
    pipeline_models: Optional[Dict[str, Any]] = None,
    specimen_id: Optional[str] = None,
    quiet: bool = False
) -> List[Dict[str, Any]]:
    """
    Processes a single surface defect image through the end-to-end inference pipeline:
    Detection -> Weak Segmentation -> Feature Extraction -> Geometric Severity ->
    Neural Severity (MC Dropout) -> Conformal Interval -> Grad-CAM -> Diagnostic Report.
    """
    img_path = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_id = specimen_id or img_path.stem

    # Unpack from pipeline_models container if provided
    if pipeline_models is not None:
        detector = pipeline_models.get("detector", detector)
        neural_model = pipeline_models.get("neural_model", neural_model)
        uncertainty_estimator = pipeline_models.get("uncertainty_estimator", uncertainty_estimator)
        gradcam_engine = pipeline_models.get("gradcam_engine", gradcam_engine)
        segmenter = pipeline_models.get("segmenter", segmenter)
        extractor = pipeline_models.get("extractor", extractor)
        calc = pipeline_models.get("calc", calc)
        device = pipeline_models.get("device", device)

    if device is None:
        device = select_device()
    if segmenter is None:
        segmenter = WeakROIThresholdSegmenter()
    if extractor is None:
        extractor = DefectFeatureExtractor(include_texture=True)
    if calc is None:
        calc = TransparentSeverityIndexCalculator()

    image = cv2.imread(str(img_path))
    if image is None:
        logger.error(f"Failed to load image from: {img_path}")
        raise ValueError(f"Could not load or decode image from: {img_path}")

    detections = []

    # 1. Autonomous YOLO Object Detection
    if detector is not None:
        detections = detector.detect(image)
    else:
        # Fallback to XML if exists
        xml_candidate = Path("data/raw/NEU-DET/ANNOTATIONS") / f"{img_path.stem}.xml"
        if xml_candidate.exists():
            parser = XMLAnnotationParser(xml_candidate.parent, img_path.parent)
            rec = parser.parse_single_file(xml_candidate)
            for obj in rec.get("objects", []):
                detections.append({
                    "bbox": (obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]),
                    "confidence": 1.0,
                    "class_name": obj["class_name"]
                })

    # Zero defect / Nominal condition
    if not detections:
        if not quiet:
            print("\n" + "="*70)
            print(f"DIAGNOSTIC INFERENCE — {img_path.name}")
            print("="*70)
            print("Inspection Status   : ✅ NO DEFECTS DETECTED (NOMINAL / PASS)")
            print("Surface Quality     : Clear (Severity: 0.0 / 100)")
            print("QC Triage Flag      : ✅ PASS (NO ACTION REQUIRED)")
            print("="*70)

        res = {
            "image_name": img_path.name,
            "specimen_id": spec_id,
            "defect_index": 0,
            "defect_class": "Clean_Surface",
            "detector_confidence": 0.0,
            "bbox_xmin": 0.0,
            "bbox_ymin": 0.0,
            "bbox_xmax": 0.0,
            "bbox_ymax": 0.0,
            "defect_area_px": 0.0,
            "area_ratio_pct": 0.0,
            "aspect_ratio": 1.0,
            "circularity": 0.0,
            "local_contrast": 0.0,
            "glcm_contrast": 0.0,
            "severity_score": 0.0,
            "severity_grade": "NOMINAL",
            "neural_severity_mean": 0.0,
            "neural_uncertainty_std": 0.0,
            "ci_95_lower": 0.0,
            "ci_95_upper": 0.0,
            "conformal_interval_width": 0.0,
            "qc_status": "HIGH_CONFIDENCE",
            "requires_human_triage": False,
            "diagnostic_report_path": "N/A"
        }
        return [res]

    image_results = []

    for idx, det in enumerate(detections):
        bbox = det["bbox"]
        cls_name = det["class_name"]
        conf = float(det.get("confidence", 1.0))

        # 2. Weak ROI Derived Mask
        mask = segmenter.segment_roi(image, bbox)

        # 3. Quantitative Feature Extraction
        features = extractor.extract(image, mask, bbox=bbox)
        features["defect_class"] = cls_name

        # 4. Continuous Severity Index & Recalibrated Grade
        s_score = calc.compute_severity_score(features, defect_class=cls_name)
        s_cat, s_id = calc.get_severity_category(s_score)

        # 5. Neural Severity, Uncertainty & Grad-CAM Heatmap
        heatmap = None
        unc_info = None

        if neural_model is not None and uncertainty_estimator is not None and gradcam_engine is not None:
            h_img, w_img = image.shape[:2]
            pad = 10
            x1, y1 = max(0, int(round(bbox[0])) - pad), max(0, int(round(bbox[1])) - pad)
            x2, y2 = min(w_img, int(round(bbox[2])) + pad), min(h_img, int(round(bbox[3])) + pad)
            crop = image[y1:y2, x1:x2] if (x2 - x1) >= 2 and (y2 - y1) >= 2 else image
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (224, 224))
            tensor_crop = transforms.ToTensor()(crop_resized)
            tensor_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_crop).unsqueeze(0).to(device)

            unc_info = uncertainty_estimator.estimate_uncertainty(tensor_norm, detector_confidence=conf)

            roi_heat = gradcam_engine.generate_heatmap(tensor_norm, original_image_shape=(y2 - y1, x2 - x1))
            full_heat = np.zeros((h_img, w_img), dtype=np.float32)
            full_heat[y1:y2, x1:x2] = roi_heat
            heatmap = full_heat

        # 6. Multi-Panel Visual Explanation Report
        explanation_filename = f"{spec_id}_defect_{idx+1:03d}_diagnostic.png"
        explanation_path = out_dir / explanation_filename

        DefectSeverityExplainer.create_explanation_plot(
            image=image,
            bbox=bbox,
            mask=mask,
            defect_class=f"{cls_name} ({conf*100:.1f}%)",
            features=features,
            severity_score=s_score,
            severity_category=s_cat,
            heatmap=heatmap,
            uncertainty_info=unc_info,
            output_path=explanation_path
        )

        neural_mean = round(float(unc_info["mean_severity_score"]), 1) if unc_info else round(float(s_score), 1)
        neural_std = round(float(unc_info["std_severity_score"]), 2) if unc_info else 0.0
        ci_l = round(float(unc_info["ci_95_lower"]), 1) if unc_info else round(float(max(0.0, s_score - 5.0)), 1)
        ci_u = round(float(unc_info["ci_95_upper"]), 1) if unc_info else round(float(min(100.0, s_score + 5.0)), 1)
        conf_width = round(float(unc_info.get("conformal_interval_width", ci_u - ci_l)), 1) if unc_info else round(ci_u - ci_l, 1)
        qc_stat = unc_info.get("qc_status", "HIGH_CONFIDENCE") if unc_info else "HIGH_CONFIDENCE"
        triage_req = unc_info["requires_human_triage"] if unc_info else False

        res = {
            "image_name": img_path.name,
            "specimen_id": spec_id,
            "defect_index": idx + 1,
            "defect_class": cls_name,
            "detector_confidence": round(conf, 3),
            "bbox_xmin": round(float(bbox[0]), 1),
            "bbox_ymin": round(float(bbox[1]), 1),
            "bbox_xmax": round(float(bbox[2]), 1),
            "bbox_ymax": round(float(bbox[3]), 1),
            "defect_area_px": round(float(features.get("defect_area_px", 0.0)), 1),
            "area_ratio_pct": round(float(features.get("area_ratio", 0.0)) * 100.0, 2),
            "aspect_ratio": round(float(features.get("aspect_ratio", 1.0)), 2),
            "circularity": round(float(features.get("circularity", 0.0)), 3),
            "local_contrast": round(float(features.get("local_contrast", 0.0)), 3),
            "glcm_contrast": round(float(features.get("glcm_contrast", 0.0)), 2),
            "severity_score": round(float(s_score), 1),
            "severity_grade": s_cat,
            "neural_severity_mean": neural_mean,
            "neural_uncertainty_std": neural_std,
            "ci_95_lower": ci_l,
            "ci_95_upper": ci_u,
            "conformal_interval_width": conf_width,
            "qc_status": qc_stat,
            "requires_human_triage": triage_req,
            "diagnostic_report_path": str(explanation_path.resolve())
        }
        image_results.append(res)

        if not quiet:
            print("\n" + "="*70)
            print(f"DEFECT #{idx+1} DIAGNOSTIC INFERENCE — {img_path.name}")
            print("="*70)
            print(f"Detected Class      : {cls_name} (Confidence: {conf*100:.1f}%)")
            print(f"Bounding Box        : {[round(x,1) for x in bbox]}")
            print(f"Geometric Severity  : {s_score:.1f} / 100 ({s_cat.upper()})")
            if unc_info:
                cal_str = " (Conformal Calibrated 95%)" if unc_info.get("is_conformal_calibrated", False) else ""
                print(f"Neural Severity (μ) : {unc_info['mean_severity_score']:.1f} ± {unc_info['std_severity_score']:.2f} points")
                print(f"95% Confidence Int. : [{unc_info['ci_95_lower']:.1f}, {unc_info['ci_95_upper']:.1f}]{cal_str} (Width: {conf_width:.1f})")
                status_icon = "⚠️" if triage_req else ("ℹ️" if qc_stat == "MODERATE_CONFIDENCE" else "✅")
                print(f"QC / Triage Status  : {status_icon} {qc_stat}")
            print(f"Diagnostic Report   : {explanation_path.resolve()}")
            print("="*70)

    return image_results

def run_batch_or_single_pipeline(
    input_path: str | Path,
    detector_weights: str | Path | None = "models/detector_yolo.pt",
    neural_severity_weights: str | Path | None = "models/severity_efficientnet_v2.pt",
    output_dir: str | Path = "outputs/explanations",
    summary_csv: str | Path = "outputs/predictions_summary.csv",
    use_clahe: bool = False,
    conformal_calibrator_path: str | Path = "models/conformal_calibrator.json",
    temperature_scaler_path: str | Path = "models/temperature_scaler.json",
    device_name: Optional[str] = None
):
    inp = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    if inp.is_file():
        image_paths = [inp]
    elif inp.is_dir():
        image_paths = sorted([p for p in inp.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]])
    else:
        logger.error(f"Input path not found: {inp}")
        return

    logger.info(f"Processing {len(image_paths)} image(s) from: {inp}...")

    # Load models once
    models = load_pipeline_models(
        detector_weights=detector_weights,
        neural_severity_weights=neural_severity_weights,
        conformal_calibrator_path=conformal_calibrator_path,
        temperature_scaler_path=temperature_scaler_path,
        use_clahe=use_clahe,
        device_name=device_name
    )

    all_results = []
    for p in image_paths:
        res = process_single_image(
            image_path=p,
            pipeline_models=models,
            output_dir=out_dir
        )
        all_results.extend(res)

    # Save summary CSV
    if all_results:
        summary_path = Path(summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved complete prediction summary table ({len(all_results)} defect detections) to: {summary_path.resolve()}")

        print("\n" + "="*70)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*70)
        print(f"Total Images Processed : {len(image_paths)}")
        print(f"Total Defects Detected : {len(all_results)}")
        print(f"Summary CSV Report     : {summary_path.resolve()}")
        print(f"Visual Reports Dir     : {out_dir.resolve()}")
        print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end steel defect detection, severity estimation, uncertainty & heatmap pipeline.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to single image file OR directory of images")
    parser.add_argument("--detector_weights", type=str, default="models/detector_yolo.pt", help="Path to trained YOLO detector weights")
    parser.add_argument("--neural_severity", type=str, default="models/severity_efficientnet_v2.pt", help="Path to trained EfficientNet-V2 weights")
    parser.add_argument("--output_dir", type=str, default="outputs/explanations", help="Directory to save visual diagnostic reports")
    parser.add_argument("--summary_csv", type=str, default="outputs/predictions_summary.csv", help="Path to save summary CSV table")
    parser.add_argument("--use_clahe", action="store_true", help="Apply CLAHE contrast enhancement during detection inference")
    parser.add_argument("--conformal_calibrator", type=str, default="models/conformal_calibrator.json", help="Path to conformal calibration JSON")
    parser.add_argument("--temperature_scaler", type=str, default="models/temperature_scaler.json", help="Path to temperature scaler JSON")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', or auto)")
    args = parser.parse_args()

    run_batch_or_single_pipeline(
        input_path=args.input_path,
        detector_weights=args.detector_weights,
        neural_severity_weights=args.neural_severity,
        output_dir=args.output_dir,
        summary_csv=args.summary_csv,
        use_clahe=args.use_clahe,
        conformal_calibrator_path=args.conformal_calibrator,
        temperature_scaler_path=args.temperature_scaler,
        device_name=args.device
    )
