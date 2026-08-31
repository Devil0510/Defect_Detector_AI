import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger("steel_defect.splitter")

class StratifiedDatasetSplitter:
    """
    Splits dataset into deterministic train, validation, and test sets using stratified class sampling.
    Enforces zero data leakage across splits.
    """
    
    def __init__(self, records: List[Dict[str, Any]], seed: int = 42):
        self.records = records
        self.seed = seed

    def create_splits(self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, List[str]]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        
        # Group image filenames by primary class label
        class_to_images: Dict[str, List[str]] = {}
        for rec in self.records:
            if not rec.get("valid", True) or not rec.get("valid_objects"):
                continue
            img_name = rec["image_filename"]
            # Assign image to primary (first) defect class tag
            primary_class = rec["valid_objects"][0]["class_name"]
            if primary_class not in class_to_images:
                class_to_images[primary_class] = []
            if img_name not in class_to_images[primary_class]:
                class_to_images[primary_class].append(img_name)

        train_imgs, val_imgs, test_imgs = [], [], []

        rng = random.Random(self.seed)

        for cls_name, img_list in sorted(class_to_images.items()):
            # Sort before shuffle for exact reproducibility
            img_list_sorted = sorted(list(set(img_list)))
            rng.shuffle(img_list_sorted)

            n_total = len(img_list_sorted)
            n_train = int(round(n_total * train_ratio))
            n_val = int(round(n_total * val_ratio))
            # Put remaining in test set
            n_test = n_total - n_train - n_val

            c_train = img_list_sorted[:n_train]
            c_val = img_list_sorted[n_train:n_train + n_val]
            c_test = img_list_sorted[n_train + n_val:]

            train_imgs.extend(c_train)
            val_imgs.extend(c_val)
            test_imgs.extend(c_test)

            logger.info(f"Class '{cls_name}' ({n_total} imgs) -> Train: {len(c_train)}, Val: {len(c_val)}, Test: {len(c_test)}")

        # Verify zero leakage
        s_train = set(train_imgs)
        s_val = set(val_imgs)
        s_test = set(test_imgs)

        overlap_tv = s_train.intersection(s_val)
        overlap_tt = s_train.intersection(s_test)
        overlap_vt = s_val.intersection(s_test)

        if overlap_tv or overlap_tt or overlap_vt:
            raise ValueError(f"Data leakage detected! Overlaps: T-V: {len(overlap_tv)}, T-T: {len(overlap_tt)}, V-T: {len(overlap_vt)}")

        logger.info(f"Splits generated successfully with zero leakage. Total Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

        return {
            "train": sorted(train_imgs),
            "val": sorted(val_imgs),
            "test": sorted(test_imgs)
        }

    def save_splits(self, splits: Dict[str, List[str]], output_dir: str | Path):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for split_name, img_list in splits.items():
            file_path = out_path / f"{split_name}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                for img in img_list:
                    f.write(f"{img}\n")
            logger.info(f"Saved {split_name} split list ({len(img_list)} entries) to {file_path}")
