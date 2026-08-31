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

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.detection.yolo_adapter import YOLODefectDetector
from src.data.xml_parser import XMLAnnotationParser
from src.segmentation.weak_segmenter import WeakROIThresholdSegmenter
from src.features.extractor import DefectFeatureExtractor
from src.severity.index_calculator import TransparentSeverityIndexCalculator
from src.severity.neural_severity import EfficientNetV2SeverityModel
from src.severity.uncertainty import MonteCarloUncertaintyEstimator
from src.visualization.gradcam import SeverityGradCAM
from src.visualization.explainer import DefectSeverityExplainer

logger = get_logger("run_pipeline")

def process_single_image(
    image_path: Path,
    detector: YOLODefectDetector | None,
    neural_model: EfficientNetV2SeverityModel | None,
    uncertainty_estimator: MonteCarloUncertaintyEstimator | None,
    gradcam_engine: SeverityGradCAM | None,
    segmenter: WeakROIThresholdSegmenter,
    extractor: DefectFeatureExtractor,
    calc: TransparentSeverityIndexCalculator,
    output_dir: Path,
    device: torch.device
):
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image from: {image_path}")
        return []

    detections = []

    # 1. Autonomous YOLO Object Detection
    if detector is not None:
        detections = detector.detect(image)
    else:
        # Fallback to XML if exists
        xml_candidate = Path("data/raw/NEU-DET/ANNOTATIONS") / f"{image_path.stem}.xml"
        if xml_candidate.exists():
            parser = XMLAnnotationParser(xml_candidate.parent, image_path.parent)
            rec = parser.parse_single_file(xml_candidate)
            for obj in rec.get("objects", []):
                detections.append({
                    "bbox": (obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]),
                    "confidence": 1.0,
                    "class_name": obj["class_name"]
                })

    if not detections:
        h, w = image.shape[:2]
        detections.append({
            "bbox": (w * 0.15, h * 0.15, w * 0.85, h * 0.85),
            "confidence": 0.50,
            "class_name": "Unclassified_Defect"
        })

    image_results = []

    for idx, det in enumerate(detections):
        bbox = det["bbox"]
        cls_name = det["class_name"]
        conf = det.get("confidence", 1.0)

        # 2. Weak ROI Derived Mask
        mask = segmenter.segment_roi(image, bbox)

        # 3. Quantitative Feature Extraction
        features = extractor.extract(image, mask, bbox=bbox)

        # 4. Continuous Severity Index & Recalibrated Grade
        s_score = calc.compute_severity_score(features)
        s_cat, s_id = calc.get_severity_category(s_score)

        # 5. Neural Severity, Uncertainty & Grad-CAM Heatmap
        heatmap = None
        unc_info = None

        if neural_model is not None:
            h_img, w_img = image.shape[:2]
            x1, y1 = max(0, int(round(bbox[0]))), max(0, int(round(bbox[1])))
            x2, y2 = min(w_img, int(round(bbox[2]))), min(h_img, int(round(bbox[3])))
            crop = image[y1:y2, x1:x2] if (x2 - x1) >= 2 and (y2 - y1) >= 2 else image
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (224, 224))
            tensor_crop = transforms.ToTensor()(crop_resized)
            tensor_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_crop).unsqueeze(0).to(device)

            unc_info = uncertainty_estimator.estimate_uncertainty(tensor_norm)

            roi_heat = gradcam_engine.generate_heatmap(tensor_norm, original_image_shape=(y2 - y1, x2 - x1))
            full_heat = np.zeros((h_img, w_img), dtype=np.float32)
            full_heat[y1:y2, x1:x2] = roi_heat
            heatmap = full_heat

        # 6. Multi-Panel Visual Explanation Report
        explanation_filename = f"{image_path.stem}_defect_{idx+1}_diagnostic.png"
        explanation_path = output_dir / explanation_filename

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

        res = {
            "image_name": image_path.name,
            "defect_index": idx + 1,
            "defect_class": cls_name,
            "detector_confidence": round(conf, 3),
            "bbox_xmin": round(bbox[0], 1),
            "bbox_ymin": round(bbox[1], 1),
            "bbox_xmax": round(bbox[2], 1),
            "bbox_ymax": round(bbox[3], 1),
            "defect_area_px": round(features.get("defect_area_px", 0.0), 1),
            "area_ratio_pct": round(features.get("area_ratio", 0.0) * 100.0, 2),
            "aspect_ratio": round(features.get("aspect_ratio", 1.0), 2),
            "circularity": round(features.get("circularity", 0.0), 3),
            "local_contrast": round(features.get("local_contrast", 0.0), 3),
            "severity_score": round(s_score, 1),
            "severity_grade": s_cat,
            "neural_severity_mean": round(unc_info["mean_severity_score"], 1) if unc_info else round(s_score, 1),
            "neural_uncertainty_std": round(unc_info["std_severity_score"], 2) if unc_info else 0.0,
            "requires_human_triage": unc_info["requires_human_triage"] if unc_info else False,
            "diagnostic_report_path": str(explanation_path.resolve())
        }
        image_results.append(res)

        print("\n" + "="*70)
        print(f"DEFECT #{idx+1} DIAGNOSTIC INFERENCE — {image_path.name}")
        print("="*70)
        print(f"Detected Class      : {cls_name} (Confidence: {conf*100:.1f}%)")
        print(f"Bounding Box        : {[round(x,1) for x in bbox]}")
        print(f"Geometric Severity  : {s_score:.1f} / 100 ({s_cat.upper()})")
        if unc_info:
            print(f"Neural Severity (μ) : {unc_info['mean_severity_score']:.1f} ± {unc_info['std_severity_score']:.2f} points")
            print(f"95% Credible Int.   : [{unc_info['ci_95_lower']:.1f}, {unc_info['ci_95_upper']:.1f}]")
            print(f"Human Triage Flag   : {'⚠️ YES (UNCERTAIN)' if unc_info['requires_human_triage'] else '✅ NO (HIGH CERTAINTY)'}")
        print(f"Diagnostic Heatmap  : {explanation_path.resolve()}")
        print("="*70)

    return image_results

def run_batch_or_single_pipeline(
    input_path: str | Path,
    detector_weights: str | Path | None = "models/detector_yolo.pt",
    neural_severity_weights: str | Path | None = "models/severity_efficientnet_v2.pt",
    output_dir: str | Path = "outputs/explanations",
    summary_csv: str | Path = "outputs/predictions_summary.csv"
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

    # Initialize models
    detector = None
    if detector_weights and Path(detector_weights).exists():
        detector = YOLODefectDetector(model_path=detector_weights, conf_threshold=0.20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural_model = None
    uncertainty_estimator = None
    gradcam_engine = None

    if neural_severity_weights and Path(neural_severity_weights).exists():
        try:
            neural_model = EfficientNetV2SeverityModel(pretrained=False, dropout_rate=0.3).to(device)
            neural_model.load_state_dict(torch.load(str(neural_severity_weights), map_location=device))
            uncertainty_estimator = MonteCarloUncertaintyEstimator(neural_model, n_samples=25, uncertainty_threshold=8.0)
            gradcam_engine = SeverityGradCAM(neural_model)
        except Exception as e:
            logger.warning(f"Could not load neural model: {e}")

    segmenter = WeakROIThresholdSegmenter()
    extractor = DefectFeatureExtractor(include_texture=True)
    calc = TransparentSeverityIndexCalculator()

    all_results = []
    for p in image_paths:
        res = process_single_image(
            image_path=p,
            detector=detector,
            neural_model=neural_model,
            uncertainty_estimator=uncertainty_estimator,
            gradcam_engine=gradcam_engine,
            segmenter=segmenter,
            extractor=extractor,
            calc=calc,
            output_dir=out_dir,
            device=device
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
    args = parser.parse_args()

    run_batch_or_single_pipeline(
        input_path=args.input_path,
        detector_weights=args.detector_weights,
        neural_severity_weights=args.neural_severity,
        output_dir=args.output_dir,
        summary_csv=args.summary_csv
    )
