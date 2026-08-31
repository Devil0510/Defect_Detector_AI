import os
import sys
import argparse
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.severity.neural_severity import EfficientNetV2SeverityModel
from src.severity.index_calculator import TransparentSeverityIndexCalculator

logger = get_logger("train_neural_severity")

def extract_features_and_targets_batched(df: pd.DataFrame, images_dir: Path, feature_extractor: nn.Module, device: torch.device, batch_size: int = 64):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_cache = {}
    
    crop_tensors = []
    scores = []
    cat_ids = []

    for _, row in df.iterrows():
        img_name = row["image_filename"]
        if img_name not in img_cache:
            p = images_dir / img_name
            im = cv2.imread(str(p))
            img_cache[img_name] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else np.zeros((200, 200, 3), dtype=np.uint8)

        image = img_cache[img_name]
        h, w = image.shape[:2]
        x1 = max(0, int(round(row["bbox_xmin"])))
        y1 = max(0, int(round(row["bbox_ymin"])))
        x2 = min(w, int(round(row["bbox_xmax"])))
        y2 = min(h, int(round(row["bbox_ymax"])))

        crop = image[y1:y2, x1:x2] if (x2 - x1) >= 2 and (y2 - y1) >= 2 else image
        crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        tensor_crop = normalize(transforms.ToTensor()(crop_resized))

        crop_tensors.append(tensor_crop)
        scores.append(float(row["severity_score"]))
        cat_ids.append(int(row["severity_id"]))

    feature_extractor.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(crop_tensors), batch_size):
            batch_tensors = torch.stack(crop_tensors[i:i+batch_size]).to(device)
            feat_map = feature_extractor.features(batch_tensors)
            pooled = feature_extractor.avgpool(feat_map)
            flat = torch.flatten(pooled, 1).cpu()
            embeddings.append(flat)

    X_emb = torch.cat(embeddings, dim=0)
    y_score = torch.tensor(scores, dtype=torch.float32)
    y_cat = torch.tensor(cat_ids, dtype=torch.long)

    return X_emb, y_score, y_cat

def train_model(features_csv: str | Path, splits_dir: str | Path, images_dir: str | Path, epochs: int, batch_size: int, lr: float, output_model: str | Path):
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

    logger.info("Pre-extracting convolutional feature embeddings in vectorized batches...")
    X_train, y_train_score, y_train_cat = extract_features_and_targets_batched(train_df, img_path, model, device, batch_size=64)
    X_val, y_val_score, y_val_cat = extract_features_and_targets_batched(val_df, img_path, model, device, batch_size=64)
    logger.info(f"Extracted embeddings: Train={X_train.shape}, Val={X_val.shape}")

    train_dataset = TensorDataset(X_train, y_train_score, y_train_cat)
    val_dataset = TensorDataset(X_val, y_val_score, y_val_cat)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    trainable_params = list(model.shared_fc.parameters()) + list(model.regression_head.parameters()) + list(model.classification_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_reg = nn.SmoothL1Loss()
    criterion_cls = nn.CrossEntropyLoss()

    best_val_mae = float("inf")
    out_p = Path(output_model)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for emb, score_t, cat_t in train_loader:
            emb, score_t, cat_t = emb.to(device), score_t.to(device), cat_t.to(device)

            optimizer.zero_grad()
            emb_drop = nn.functional.dropout(emb, p=model.dropout_rate, training=True)
            proj = model.shared_fc(emb_drop)
            pred_score = model.regression_head(proj).squeeze(-1) * 100.0
            pred_logits = model.classification_head(proj)

            loss_r = criterion_reg(pred_score, score_t)
            loss_c = criterion_cls(pred_logits, cat_t)
            loss = loss_r + 0.5 * loss_c

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(emb)

        train_loss /= len(train_dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_mae = 0.0
        val_correct = 0

        with torch.no_grad():
            for emb, score_t, cat_t in val_loader:
                emb, score_t, cat_t = emb.to(device), score_t.to(device), cat_t.to(device)
                proj = model.shared_fc(emb)
                pred_score = model.regression_head(proj).squeeze(-1) * 100.0
                pred_logits = model.classification_head(proj)

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
    parser = argparse.ArgumentParser(description="Train EfficientNet-V2 neural severity model.")
    parser.add_argument("--features_csv", type=str, default="data/processed/defect_features.csv")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--images_dir", type=str, default="data/raw/NEU-DET/IMAGES")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--output_model", type=str, default="models/severity_efficientnet_v2.pt")
    args = parser.parse_args()

    train_model(args.features_csv, args.splits_dir, args.images_dir, args.epochs, args.batch_size, args.lr, args.output_model)
