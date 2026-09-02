import os
import sys
import argparse
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.severity.neural_severity import EfficientNetV2SeverityModel
from src.severity.index_calculator import TransparentSeverityIndexCalculator

logger = get_logger("train_neural_severity")

class DefectROIImageDataset(Dataset):
    """
    Dynamic Dataset for Defect ROI Cropping with Data Augmentation for End-to-End Training.
    """
    def __init__(self, df: pd.DataFrame, images_dir: Path, is_train: bool = True, pad: int = 10):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.is_train = is_train
        self.pad = pad
        
        # Pre-cache images in memory for high-throughput training
        self.img_cache = {}
        unique_images = self.df["image_filename"].unique()
        logger.info(f"Pre-caching {len(unique_images)} images into RAM for dataset...")
        for img_name in unique_images:
            p = self.images_dir / img_name
            im = cv2.imread(str(p))
            self.img_cache[img_name] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else np.zeros((200, 200, 3), dtype=np.uint8)

        # Transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row["image_filename"]
        image = self.img_cache[img_name]
        h, w = image.shape[:2]

        x1 = max(0, int(round(row["bbox_xmin"])) - self.pad)
        y1 = max(0, int(round(row["bbox_ymin"])) - self.pad)
        x2 = min(w, int(round(row["bbox_xmax"])) + self.pad)
        y2 = min(h, int(round(row["bbox_ymax"])) + self.pad)

        crop = image[y1:y2, x1:x2] if (x2 - x1) >= 2 and (y2 - y1) >= 2 else image
        crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)

        tensor_crop = self.transform(crop_resized)
        score = float(row["severity_score"])
        cat_id = int(row["severity_id"])

        return tensor_crop, torch.tensor(score, dtype=torch.float32), torch.tensor(cat_id, dtype=torch.long)

def train_model(
    features_csv: str | Path,
    splits_dir: str | Path,
    images_dir: str | Path,
    epochs: int,
    batch_size: int,
    lr_backbone: float,
    lr_heads: float,
    freeze_backbone: bool,
    output_model: str | Path
):
    df = pd.read_csv(features_csv)
    splits_path = Path(splits_dir)
    img_path = Path(images_dir)

    train_files = set(Path(f).name for f in open(splits_path / "train.txt").read().splitlines() if f.strip())
    val_files = set(Path(f).name for f in open(splits_path / "val.txt").read().splitlines() if f.strip())

    calc = TransparentSeverityIndexCalculator()
    scores, cat_ids = [], []
    for _, row in df.iterrows():
        s = calc.compute_severity_score(row)
        _, cid = calc.get_severity_category(s)
        scores.append(s)
        cat_ids.append(cid)

    df["severity_score"] = scores
    df["severity_id"] = cat_ids

    train_df = df[df["image_filename"].isin(train_files)].reset_index(drop=True)
    val_df = df[df["image_filename"].isin(val_files)].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing EfficientNet-V2 on device: {device}")
    model = EfficientNetV2SeverityModel(pretrained=True, dropout_rate=0.3).to(device)

    train_dataset = DefectROIImageDataset(train_df, img_path, is_train=True, pad=10)
    val_dataset = DefectROIImageDataset(val_df, img_path, is_train=False, pad=10)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    if freeze_backbone:
        logger.info("Training with FROZEN backbone (training heads only)...")
        for p in model.features.parameters():
            p.requires_grad = False
        trainable_params = list(model.shared_fc.parameters()) + list(model.regression_head.parameters()) + list(model.classification_head.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=lr_heads, weight_decay=1e-4)
    else:
        logger.info(f"Training with END-TO-END FINE-TUNING (Backbone LR: {lr_backbone}, Heads LR: {lr_heads})...")
        for p in model.features.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW([
            {"params": model.features.parameters(), "lr": lr_backbone, "weight_decay": 1e-4},
            {"params": model.shared_fc.parameters(), "lr": lr_heads, "weight_decay": 1e-4},
            {"params": model.regression_head.parameters(), "lr": lr_heads, "weight_decay": 1e-4},
            {"params": model.classification_head.parameters(), "lr": lr_heads, "weight_decay": 1e-4},
        ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_reg = nn.SmoothL1Loss()
    criterion_cls = nn.CrossEntropyLoss()

    best_val_mae = float("inf")
    out_p = Path(output_model)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for crops, score_t, cat_t in train_loader:
            crops, score_t, cat_t = crops.to(device), score_t.to(device), cat_t.to(device)

            optimizer.zero_grad()
            pred_score, pred_logits = model(crops, enable_mc_dropout=False)

            loss_r = criterion_reg(pred_score, score_t)
            loss_c = criterion_cls(pred_logits, cat_t)
            loss = loss_r + 0.5 * loss_c

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(crops)

        train_loss /= len(train_dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_mae = 0.0
        val_correct = 0

        with torch.no_grad():
            for crops, score_t, cat_t in val_loader:
                crops, score_t, cat_t = crops.to(device), score_t.to(device), cat_t.to(device)
                pred_score, pred_logits = model(crops, enable_mc_dropout=False)

                val_mae += torch.sum(torch.abs(pred_score - score_t)).item()
                preds = torch.argmax(pred_logits, dim=-1)
                val_correct += torch.sum(preds == cat_t).item()

        val_mae /= len(val_dataset)
        val_acc = (val_correct / len(val_dataset)) * 100.0

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Score MAE: {val_mae:.2f} pts | Val Acc: {val_acc:.2f}%")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), str(out_p))
            logger.info(f"--> Saved best EfficientNet-V2 model (Val MAE: {val_mae:.2f}) to {out_p}")

    print("\n" + "="*65)
    print("EFFICIENTNET-V2 SEVERITY MODEL TRAINING COMPLETE")
    print("="*65)
    print(f"Best Validation Score MAE : {best_val_mae:.2f} points (out of 100)")
    print(f"Saved Model Weights       : {out_p.resolve()}")
    print("="*65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-V2 neural severity model with end-to-end fine-tuning.")
    parser.add_argument("--features_csv", type=str, default="data/processed/defect_features.csv")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--images_dir", type=str, default="data/raw/NEU-DET/IMAGES")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Backbone learning rate for end-to-end fine-tuning")
    parser.add_argument("--lr_heads", type=float, default=1e-3, help="Heads learning rate")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone features and only train heads")
    parser.add_argument("--output_model", type=str, default="models/severity_efficientnet_v2.pt")
    args = parser.parse_args()

    train_model(
        features_csv=args.features_csv,
        splits_dir=args.splits_dir,
        images_dir=args.images_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_heads=args.lr_heads,
        freeze_backbone=args.freeze_backbone,
        output_model=args.output_model
    )

