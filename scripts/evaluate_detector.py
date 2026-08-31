import os
import sys
import json
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger

logger = get_logger("evaluate_detector")

def evaluate_yolo(model_path: str | Path, config_file: str | Path, output_report: str | Path):
    logger.info(f"Loading trained YOLO model from {model_path}...")
    model = YOLO(str(model_path))

    logger.info(f"Evaluating detector on held-out test split defined in {config_file}...")
    val_results = model.val(
        data=str(config_file),
        split="test",
        project="outputs/yolo_runs",
        name="val_exp",
        exist_ok=True,
        verbose=True
    )

    metrics_dict = {
        "mAP_0.5": float(val_results.box.map50),
        "mAP_0.5_0.95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
        "class_names": val_results.names,
        "per_class_map50": {
            val_results.names[idx]: float(val_results.box.maps[idx])
            for idx in range(len(val_results.names))
        } if hasattr(val_results.box, "maps") and len(val_results.box.maps) == len(val_results.names) else {}
    }

    # Measure average latency on test set
    test_files = [f.strip() for f in open("data/splits/test_yolo.txt") if f.strip()]
    latencies = []
    for test_img in test_files[:50]:
        t0 = time.perf_counter()
        _ = model.predict(test_img, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0) # in ms

    metrics_dict["mean_inference_latency_ms"] = float(sum(latencies) / len(latencies)) if latencies else 0.0

    out_p = Path(output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)

    print("\n" + "="*60)
    print("YOLO TEST EVALUATION METRICS")
    print("="*60)
    print(f"mAP@0.5                : {metrics_dict['mAP_0.5']:.4f}")
    print(f"mAP@0.5:0.95           : {metrics_dict['mAP_0.5_0.95']:.4f}")
    print(f"Mean Precision         : {metrics_dict['precision']:.4f}")
    print(f"Mean Recall            : {metrics_dict['recall']:.4f}")
    print(f"Mean Latency           : {metrics_dict['mean_inference_latency_ms']:.2f} ms / frame")
    print(f"Saved Report           : {out_p.resolve()}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector on test split.")
    parser.add_argument("--model_path", type=str, default="models/detector_yolo.pt", help="Path to trained YOLO weights")
    parser.add_argument("--config_file", type=str, default="configs/yolo_dataset.yaml", help="Path to YOLO dataset YAML")
    parser.add_argument("--output_report", type=str, default="reports/detection_metrics.json", help="Path to save evaluation report JSON")
    args = parser.parse_args()

    evaluate_yolo(args.model_path, args.config_file, args.output_report)
