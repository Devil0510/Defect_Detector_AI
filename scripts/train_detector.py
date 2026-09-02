import os
import sys
import cv2
import yaml
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.data.clahe import apply_clahe

logger = get_logger("train_detector")

def generate_clahe_yolo_dataset(base_yolo_dir: Path, out_clahe_dir: Path, clip_limit: float = 2.5) -> Path:
    """
    Creates a CLAHE contrast-enhanced copy of the YOLO dataset for training.
    """
    logger.info(f"Generating CLAHE-enhanced dataset at: {out_clahe_dir}")
    out_clahe_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        img_src_dir = base_yolo_dir / "images" / split
        img_dst_dir = out_clahe_dir / "images" / split
        lbl_src_dir = base_yolo_dir / "labels" / split
        lbl_dst_dir = out_clahe_dir / "labels" / split

        img_dst_dir.mkdir(parents=True, exist_ok=True)
        lbl_dst_dir.mkdir(parents=True, exist_ok=True)

        if img_src_dir.exists():
            for img_file in tqdm(list(img_src_dir.glob("*.*")), desc=f"CLAHE {split} images"):
                im = cv2.imread(str(img_file))
                if im is not None:
                    im_clahe = apply_clahe(im, clip_limit=clip_limit)
                    cv2.imwrite(str(img_dst_dir / img_file.name), im_clahe)

        if lbl_src_dir.exists():
            for lbl_file in lbl_src_dir.glob("*.txt"):
                shutil.copy(lbl_file, lbl_dst_dir / lbl_file.name)

    # Create dataset yaml for CLAHE dataset
    clahe_yaml_path = Path("configs/yolo_dataset_clahe.yaml")
    clahe_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_dict = {
        "path": str(out_clahe_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 6,
        "names": {
            0: "crazing",
            1: "inclusion",
            2: "patches",
            3: "pitted_surface",
            4: "rolled-in_scale",
            5: "scratches"
        }
    }
    with open(clahe_yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)

    logger.info(f"Saved CLAHE YOLO dataset config to: {clahe_yaml_path}")
    return clahe_yaml_path

def train_yolo(
    config_file: str | Path,
    model_variant: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    output_model: str | Path,
    use_clahe: bool = False,
    clahe_clip_limit: float = 2.5,
    mosaic: float = 1.0,
    mixup: float = 0.15,
    degrees: float = 10.0,
    fliplr: float = 0.5,
    flipud: float = 0.5,
    scale: float = 0.5
):
    cfg_path = Path(config_file)
    if use_clahe:
        base_yolo_dir = Path("data/interim/yolo")
        out_clahe_dir = Path("data/interim/yolo_clahe")
        cfg_path = generate_clahe_yolo_dataset(base_yolo_dir, out_clahe_dir, clip_limit=clahe_clip_limit)

    logger.info(f"Initializing YOLO model variant '{model_variant}' for training...")
    model = YOLO(model_variant)

    logger.info(
        f"Starting training on {cfg_path} for {epochs} epochs (imgsz={imgsz}, batch={batch_size}, "
        f"mosaic={mosaic}, mixup={mixup}, degrees={degrees}, fliplr={fliplr}, flipud={flipud})..."
    )
    results = model.train(
        data=str(cfg_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project="outputs/yolo_runs",
        name="train_exp",
        exist_ok=True,
        seed=42,
        mosaic=mosaic,
        mixup=mixup,
        degrees=degrees,
        fliplr=fliplr,
        flipud=flipud,
        scale=scale,
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
    parser = argparse.ArgumentParser(description="Train YOLO steel defect detector with augmentations and CLAHE.")
    parser.add_argument("--config_file", type=str, default="configs/yolo_dataset.yaml", help="Path to YOLO dataset YAML")
    parser.add_argument("--model_variant", type=str, default="yolov8n.pt", help="Pretrained YOLO model variant")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=200, help="Image resolution")
    parser.add_argument("--output_model", type=str, default="models/detector_yolo.pt", help="Path to save trained detector weights")
    parser.add_argument("--use_clahe", action="store_true", help="Apply CLAHE contrast enhancement before training")
    parser.add_argument("--clahe_clip_limit", type=float, default=2.5, help="CLAHE clip limit")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.15, help="Mixup augmentation probability")
    parser.add_argument("--degrees", type=float, default=10.0, help="Rotation degrees augmentation")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--flipud", type=float, default=0.5, help="Vertical flip probability")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale jitter")
    args = parser.parse_args()

    train_yolo(
        config_file=args.config_file,
        model_variant=args.model_variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        output_model=args.output_model,
        use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        mosaic=args.mosaic,
        mixup=args.mixup,
        degrees=args.degrees,
        fliplr=args.fliplr,
        flipud=args.flipud,
        scale=args.scale
    )

