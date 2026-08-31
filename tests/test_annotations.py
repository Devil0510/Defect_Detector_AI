import pytest
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.xml_parser import XMLAnnotationParser
from src.data.validator import DatasetValidator
from src.data.splitter import StratifiedDatasetSplitter


SAMPLE_XML_VALID = """<annotation>
	<folder>IMAGES</folder>
	<filename>crazing_1.jpg</filename>
	<size>
		<width>200</width>
		<height>200</height>
		<depth>1</depth>
	</size>
	<object>
		<name>crazing</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>18</xmin>
			<ymin>10</ymin>
			<xmax>150</xmax>
			<ymax>180</ymax>
		</bndbox>
	</object>
</annotation>"""

SAMPLE_XML_INVALID_BOX = """<annotation>
	<folder>IMAGES</folder>
	<filename>crazing_2.jpg</filename>
	<size>
		<width>200</width>
		<height>200</height>
		<depth>1</depth>
	</size>
	<object>
		<name>crazing</name>
		<bndbox>
			<xmin>150</xmin>
			<ymin>180</ymin>
			<xmax>18</xmax>
			<ymax>10</ymax>
		</bndbox>
	</object>
</annotation>"""

def test_xml_parser_valid():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        xml_dir = tmp_path / "ANNOTATIONS"
        img_dir = tmp_path / "IMAGES"
        xml_dir.mkdir()
        img_dir.mkdir()

        xml_file = xml_dir / "crazing_1.xml"
        xml_file.write_text(SAMPLE_XML_VALID)

        parser = XMLAnnotationParser(xml_dir, img_dir)
        record = parser.parse_single_file(xml_file)

        assert record["valid"] is True
        assert record["image_filename"] == "crazing_1.jpg"
        assert record["width"] == 200
        assert record["height"] == 200
        assert len(record["objects"]) == 1
        obj = record["objects"][0]
        assert obj["class_name"] == "crazing"
        assert obj["xmin"] == 18
        assert obj["ymin"] == 10
        assert obj["xmax"] == 150
        assert obj["ymax"] == 180
        assert obj["bbox_width"] == 132
        assert obj["bbox_height"] == 170

def test_validator_invalid_box():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        xml_dir = tmp_path / "ANNOTATIONS"
        img_dir = tmp_path / "IMAGES"
        xml_dir.mkdir()
        img_dir.mkdir()

        xml_file = xml_dir / "crazing_2.xml"
        xml_file.write_text(SAMPLE_XML_INVALID_BOX)
        
        # Create empty image file
        (img_dir / "crazing_2.jpg").touch()

        parser = XMLAnnotationParser(xml_dir, img_dir)
        records = parser.parse_all()

        validator = DatasetValidator(records, img_dir)
        report = validator.validate()

        assert report["invalid_boxes_count"] == 1
        assert len(records[0]["valid_objects"]) == 0

def test_splitter_zero_leakage():
    records = []
    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    for c_idx, cls in enumerate(classes):
        for i in range(20):
            img_name = f"{cls}_{i}.jpg"
            records.append({
                "valid": True,
                "image_filename": img_name,
                "valid_objects": [{"class_name": cls}]
            })

    splitter = StratifiedDatasetSplitter(records, seed=42)
    splits = splitter.create_splits(0.70, 0.15, 0.15)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    assert len(train_set) + len(val_set) + len(test_set) == 120
