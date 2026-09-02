import pytest
import cv2
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.batch_analysis import (
    discover_images,
    get_unique_specimen_dir,
    check_write_permission,
    process_batch,
    SUPPORTED_EXTENSIONS
)
from src.severity.uncertainty import evaluate_qc_triage

def test_image_discovery():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create valid image files with mixed case
        (tmp_path / "img_b.JPG").touch()
        (tmp_path / "img_a.png").touch()
        (tmp_path / "img_c.JPEG").touch()
        (tmp_path / "img_d.bmp").touch()
        (tmp_path / "img_e.TIF").touch()
        (tmp_path / "img_f.tiff").touch()

        # Create unsupported files
        (tmp_path / "notes.txt").touch()
        (tmp_path / "data.csv").touch()
        (tmp_path / "config.json").touch()
        (tmp_path / ".hidden.png").touch()

        # Create nested directory
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "nested.png").touch()

        # Non-recursive discovery
        discovered = discover_images(tmp_path, recursive=False)
        names = [p.name for p in discovered]

        assert len(discovered) == 6
        assert names == ["img_a.png", "img_b.JPG", "img_c.JPEG", "img_d.bmp", "img_e.TIF", "img_f.tiff"]
        assert "notes.txt" not in names
        assert "data.csv" not in names
        assert ".hidden.png" not in names

        # Recursive discovery
        rec_discovered = discover_images(tmp_path, recursive=True)
        assert len(rec_discovered) == 7
        assert any(p.name == "nested.png" for p in rec_discovered)

def test_specimen_collision_resolution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        allocated = set()

        img1 = Path("sample.jpg")
        img2 = Path("sample.png")
        img3 = Path("sample.bmp")
        img4 = Path("other_sample.jpg")

        spec_id1, path1 = get_unique_specimen_dir(out_dir, img1, allocated)
        spec_id2, path2 = get_unique_specimen_dir(out_dir, img2, allocated)
        spec_id3, path3 = get_unique_specimen_dir(out_dir, img3, allocated)
        spec_id4, path4 = get_unique_specimen_dir(out_dir, img4, allocated)

        assert spec_id1 == "sample"
        assert spec_id2 == "sample_2"
        assert spec_id3 == "sample_3"
        assert spec_id4 == "other_sample"

        assert path1 == out_dir / "sample"
        assert path2 == out_dir / "sample_2"
        assert path3 == out_dir / "sample_3"
        assert path4 == out_dir / "other_sample"

def test_multi_factor_qc_triage():
    # 1. High Confidence scenario
    status, triage, reasons = evaluate_qc_triage(
        detector_confidence=0.85,
        mc_dropout_std=3.0,
        conformal_interval_width=15.0
    )
    assert status == "HIGH_CONFIDENCE"
    assert triage is False
    assert len(reasons) == 0

    # 2. Moderate Confidence scenario (e.g. conf=0.60, std=4.0, width=18.0)
    status, triage, reasons = evaluate_qc_triage(
        detector_confidence=0.60,
        mc_dropout_std=4.0,
        conformal_interval_width=18.0
    )
    assert status == "MODERATE_CONFIDENCE"
    assert triage is False
    assert len(reasons) == 0

    # 3. Low detector confidence triggering triage (conf = 0.40)
    status, triage, reasons = evaluate_qc_triage(
        detector_confidence=0.40,
        mc_dropout_std=3.0,
        conformal_interval_width=15.0
    )
    assert status == "HUMAN_REVIEW_REQUIRED"
    assert triage is True
    assert any("Low detector confidence" in r for r in reasons)

    # 4. High epistemic variance triggering triage (mc_std = 9.5)
    status, triage, reasons = evaluate_qc_triage(
        detector_confidence=0.90,
        mc_dropout_std=9.5,
        conformal_interval_width=18.0
    )
    assert status == "HUMAN_REVIEW_REQUIRED"
    assert triage is True
    assert any("High epistemic uncertainty" in r for r in reasons)

    # 5. Wide conformal interval triggering triage (width = 38.0)
    status, triage, reasons = evaluate_qc_triage(
        detector_confidence=0.80,
        mc_dropout_std=4.0,
        conformal_interval_width=38.0
    )
    assert status == "HUMAN_REVIEW_REQUIRED"
    assert triage is True
    assert any("Wide conformal interval" in r for r in reasons)

    # 6. Low detector confidence (40%) and wide conformal interval [0, 61.6] (width = 61.6)
    # The previous scientific bug test case
    status, triage, reasons = evaluate_qc_triage(
        detector_confidence=0.40,
        mc_dropout_std=5.0,
        conformal_interval_width=61.6
    )
    assert status == "HUMAN_REVIEW_REQUIRED"
    assert triage is True
    assert len(reasons) >= 2

def test_end_to_end_batch_orchestrator():
    with tempfile.TemporaryDirectory() as tmp_in_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
        in_path = Path(tmp_in_dir)
        out_path = Path(tmp_out_dir)

        # Create dummy synthetic steel surface images
        # Image 1: Normal uniform image (likely 0 defects)
        img1 = np.full((200, 200, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(in_path / "steel_sample_01.jpg"), img1)

        # Image 2: Image with an intense dark bar (defect)
        img2 = np.full((200, 200, 3), 150, dtype=np.uint8)
        img2[40:160, 40:60] = 20
        cv2.imwrite(str(in_path / "steel_sample_02.png"), img2)

        # Image 3: Unsupported file
        (in_path / "notes.txt").write_text("Test notes")

        # Run batch processing
        summary = process_batch(
            input_dir=in_path,
            output_dir=out_path,
            device_name="cpu"
        )

        assert summary["total_images_discovered"] == 2
        assert summary["successful_images"] == 2
        assert summary["failed_images"] == 0

        # Verify output files
        assert (out_path / "MASTER_SUMMARY.csv").exists()
        assert (out_path / "SPECIMEN_SUMMARY.csv").exists()
        assert (out_path / "BATCH_METADATA.json").exists()

        # Check specimen directories
        spec1_dir = out_path / "steel_sample_01"
        spec2_dir = out_path / "steel_sample_02"
        assert spec1_dir.exists()
        assert spec2_dir.exists()

        assert (spec1_dir / "specimen_summary.json").exists()
        assert (spec1_dir / "specimen_summary.csv").exists()
        assert (spec2_dir / "specimen_summary.json").exists()
        assert (spec2_dir / "specimen_summary.csv").exists()

        # Verify specimen json contents
        with open(spec1_dir / "specimen_summary.json") as f:
            j1 = json.load(f)
            assert j1["specimen_id"] == "steel_sample_01"
            assert j1["processing_status"] == "SUCCESS"
            assert "total_defects_detected" in j1

        # Verify SPECIMEN_SUMMARY.csv
        df_spec = pd.read_csv(out_path / "SPECIMEN_SUMMARY.csv")
        assert len(df_spec) == 2
        assert set(df_spec["specimen_id"]) == {"steel_sample_01", "steel_sample_02"}
        assert "average_geometric_severity" in df_spec.columns
        assert "requires_human_inspection" in df_spec.columns
