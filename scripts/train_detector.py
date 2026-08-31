import os
import sys
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger

logger = get_logger("train_detector")

def train_yolo(config_file: str | Path, model_variant: str, epochs: int, batch_size: int, imgsz: int, output_model: str | Path):
    logger.info(f"Initializing YOLO model variant '{model_variant}' for training...")
    model = YOLO(model_variant)

    logger.info(f"Starting training on {config_file} for {epochs} epochs (imgsz={imgsz}, batch={batch_size})...")
    results = model.train(
        data=str(config_file),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project="outputs/yolo_runs",
        name="train_exp",
        exist_ok=True,
        seed=42,
        device="cpu", # Fallback to CPU or GPU automatically handled
        verbose=True
    )

    best_weight = Path("outputs/yolo_runs/train_exp/weights/best.pt")
    out_path = Path(output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if best_weight.exists():
        shutil.copy(best_weight, out_path)
        logger.info(f"Best detector weights copied to {out_path}")
    else:
        last_weight = Path("outputs/yolo_runs/train_exp/weights/last.pt")
        if last_weight.exists():
            shutil.copy(last_weight, out_path)
            logger.info(f"Last detector weights copied to {out_path}")

    print("\n" + "="*60)
    print("YOLO DETECTOR TRAINING COMPLETE")
    print("="*60)
    print(f"Saved weights: {out_path.resolve()}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO steel defect detector.")
    parser.add_argument("--config_file", type=str, default="configs/yolo_dataset.yaml", help="Path to YOLO dataset YAML")
    parser.add_argument("--model_variant", type=str, default="yolov8n.pt", help="Pretrained YOLO model variant")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=200, help="Image resolution")
    parser.add_argument("--output_model", type=str, default="models/detector_yolo.pt", help="Path to save trained detector weights")
    args = parser.parse_args()

    train_yolo(args.config_file, args.model_variant, args.epochs, args.batch_size, args.imgsz, args.output_model)
