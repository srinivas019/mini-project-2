"""
tests/test_pipeline.py - Automated Unit & Integration Tests
===========================================================

Comprehensive test suite verifying all critical functional requirements:
1. Multiple detected defects handling.
2. Zero defect handling (no crashes on clean roads).
3. Invalid file uploads (corrupt data, invalid extensions, zero-byte files).
4. Overlapping bounding boxes (validating that union area <= sum of individual areas).
5. Edge case dimensions (small 10x10 and large 4000x3000 images).
6. Severity threshold boundary logic (Low, Medium, High).
7. Missing model weights handling (helpful exception message).
8. Confidence threshold filtering changes.
"""

import sys
import io
import unittest
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from severity import (
    calculate_bbox_area,
    calculate_coverage_percentage,
    classify_severity,
    calculate_union_coverage,
    enrich_detections,
)
from utils import (
    validate_image_file,
    draw_annotations,
    detections_to_dataframe,
    export_dataframe_to_csv,
    convert_image_to_bytes,
)
from detector import PotholeDetector


class TestSeverityAndAreaCalculations(unittest.TestCase):
    """Test mathematical formulas and severity logic."""

    def test_bbox_area_standard(self):
        """Verify area = (x2 - x1) * (y2 - y1)."""
        area = calculate_bbox_area(10, 20, 60, 80)
        self.assertEqual(area, 50 * 60)

    def test_bbox_area_inverted_or_zero(self):
        """Verify degenerate or inverted boxes return 0 area."""
        self.assertEqual(calculate_bbox_area(100, 100, 50, 50), 0)
        self.assertEqual(calculate_bbox_area(50, 50, 50, 50), 0)

    def test_coverage_percentage(self):
        """Verify coverage percentage calculation and boundary clamping."""
        # 1,000 px bbox in 100,000 px image = 1.0%
        self.assertAlmostEqual(calculate_coverage_percentage(1000, 100000), 1.0)
        # Division by zero guard
        self.assertEqual(calculate_coverage_percentage(500, 0), 0.0)
        # Clamping
        self.assertEqual(calculate_coverage_percentage(200000, 100000), 100.0)

    def test_severity_classification_boundaries(self):
        """Verify configurable severity thresholds."""
        # Default: Low < 1.0%, Medium 1.0% to < 5.0%, High >= 5.0%
        self.assertEqual(classify_severity(0.4, 1.0, 5.0), "Low")
        self.assertEqual(classify_severity(0.999, 1.0, 5.0), "Low")
        self.assertEqual(classify_severity(1.0, 1.0, 5.0), "Medium")
        self.assertEqual(classify_severity(4.999, 1.0, 5.0), "Medium")
        self.assertEqual(classify_severity(5.0, 1.0, 5.0), "High")
        self.assertEqual(classify_severity(12.5, 1.0, 5.0), "High")

        # Custom thresholds: Low < 2.0%, Medium 2.0% to < 8.0%, High >= 8.0%
        self.assertEqual(classify_severity(1.5, 2.0, 8.0), "Low")
        self.assertEqual(classify_severity(3.0, 2.0, 8.0), "Medium")
        self.assertEqual(classify_severity(9.0, 2.0, 8.0), "High")

    def test_overlapping_bounding_boxes_union(self):
        """
        Verify that overlapping boxes are NOT double-counted.
        Box 1: [0, 0, 100, 100] -> area 10,000
        Box 2: [50, 0, 150, 100] -> area 10,000
        Overlap: [50, 0, 100, 100] -> area 5,000
        Expected Union: 15,000 (strictly less than sum 20,000)
        """
        boxes = [[0, 0, 100, 100], [50, 0, 150, 100]]
        w, h = 200, 100
        union_px, union_pct = calculate_union_coverage(boxes, w, h)

        self.assertEqual(union_px, 15000)
        # Image area is 200 * 100 = 20,000. 15,000 / 20,000 = 75.0%
        self.assertAlmostEqual(union_pct, 75.0, places=1)

    def test_enrich_detections_multiple_defects(self):
        """Verify complete enrichment pipeline with multiple defects."""
        raw = [
            {"id": "DEF-001", "class_name": "pothole", "confidence": 0.92, "bbox": [10, 10, 50, 50]},
            {"id": "DEF-002", "class_name": "crack", "confidence": 0.81, "bbox": [100, 100, 400, 400]},
        ]
        w, h = 1000, 1000  # Total area = 1,000,000
        enriched, summary = enrich_detections(raw, w, h, low_threshold=1.0, medium_threshold=5.0)

        self.assertEqual(len(enriched), 2)
        self.assertEqual(summary["total_detections"], 2)
        # Box 1: 40x40 = 1600 px -> 0.16% -> Low
        self.assertEqual(enriched[0]["severity"], "Low")
        # Box 2: 300x300 = 90,000 px -> 9.0% -> High
        self.assertEqual(enriched[1]["severity"], "High")
        self.assertEqual(summary["low_severity_count"], 1)
        self.assertEqual(summary["high_severity_count"], 1)
        self.assertEqual(summary["medium_severity_count"], 0)

    def test_zero_detections_handling(self):
        """Verify enrichment gracefully handles zero detections without crashing."""
        raw = []
        enriched, summary = enrich_detections(raw, 800, 600)
        self.assertEqual(len(enriched), 0)
        self.assertEqual(summary["total_detections"], 0)
        self.assertEqual(summary["total_union_area_px"], 0)
        self.assertEqual(summary["total_union_coverage_pct"], 0.0)

    def test_extreme_dimensions(self):
        """Verify handling of tiny (10x10) and large (4000x3000) dimensions."""
        # Tiny image
        boxes = [[2, 2, 8, 8]]
        union_px, union_pct = calculate_union_coverage(boxes, 10, 10)
        self.assertEqual(union_px, 36)
        self.assertAlmostEqual(union_pct, 36.0)

        # Large image
        boxes_large = [[100, 100, 1100, 1100]]
        union_px_l, union_pct_l = calculate_union_coverage(boxes_large, 4000, 3000)
        self.assertEqual(union_px_l, 1000000)
        self.assertAlmostEqual(union_pct_l, (1000000 / 12000000) * 100.0, places=2)


class TestUtilsAndValidation(unittest.TestCase):
    """Test image validation, annotation rendering, and exports."""

    def test_valid_image_upload(self):
        """Verify valid JPEG file passes validation."""
        img = Image.new("RGB", (300, 200), color=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        buf.name = "test_road.jpg"

        is_valid, img_np, err = validate_image_file(buf)
        self.assertTrue(is_valid)
        self.assertIsNotNone(img_np)
        self.assertEqual(img_np.shape, (200, 300, 3))
        self.assertEqual(err, "")

    def test_invalid_extension(self):
        """Verify non-supported extension is rejected."""
        buf = io.BytesIO(b"dummy pdf content")
        buf.name = "document.pdf"
        is_valid, img_np, err = validate_image_file(buf)
        self.assertFalse(is_valid)
        self.assertIn("Unsupported file format", err)

    def test_corrupted_image_file(self):
        """Verify corrupt byte stream is caught gracefully."""
        buf = io.BytesIO(b"not an actual jpeg image data header")
        buf.name = "corrupt.jpg"
        is_valid, img_np, err = validate_image_file(buf)
        self.assertFalse(is_valid)
        self.assertIn("Failed to parse image file", err)

    def test_empty_file_upload(self):
        """Verify 0-byte file is caught gracefully."""
        buf = io.BytesIO(b"")
        buf.name = "empty.png"
        is_valid, img_np, err = validate_image_file(buf)
        self.assertFalse(is_valid)
        self.assertIn("empty", err)

    def test_annotation_drawing_and_export(self):
        """Verify bounding box rendering does not crash and produces valid bytes."""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        detections = [
            {
                "id": "DEF-001",
                "class_name": "pothole",
                "confidence_pct": 88.5,
                "bbox": [50, 50, 150, 150],
                "bbox_area_px": 10000,
                "coverage_pct": 6.25,
                "severity": "High",
            }
        ]
        annotated = draw_annotations(img, detections)
        self.assertEqual(annotated.shape, img.shape)

        # Test byte conversion
        jpg_bytes = convert_image_to_bytes(annotated, "JPEG")
        self.assertGreater(len(jpg_bytes), 0)

        # Test DataFrame and CSV export
        df = detections_to_dataframe(detections)
        self.assertEqual(len(df), 1)
        self.assertIn("Detection ID", df.columns)
        csv_bytes = export_dataframe_to_csv(df)
        self.assertGreater(len(csv_bytes), 0)


class TestDetectorEdgeCases(unittest.TestCase):
    """Test detector instantiation, missing weights, and demo fallback."""

    def test_missing_weights_raises_file_not_found(self):
        """Verify missing model weights file raises clear FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as ctx:
            _ = PotholeDetector(model_path="models/non_existent_weights_xyz123.pt")
        self.assertIn("Model weights file not found", str(ctx.exception))

    def test_yolo_pothole_detector_inference(self):
        """Verify PotholeDetector loads best.pt and runs inference on a test road image."""
        model_path = Path("models/best.pt")
        if not model_path.exists():
            self.skipTest("models/best.pt not present for inference test.")

        detector = PotholeDetector(model_path=str(model_path))
        import cv2
        img = cv2.imread("sample_images/single_severe_pothole.jpg")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Predict with threshold 0.25
        dets = detector.predict(img_rgb, conf_threshold=0.25)
        self.assertIsInstance(dets, list)
        self.assertGreater(len(dets), 0)

        first = dets[0]
        self.assertIn("id", first)
        self.assertEqual(first["class_name"], "pothole")
        self.assertIn("confidence", first)
        self.assertIn("bbox", first)
        self.assertEqual(len(first["bbox"]), 4)

        # Very high confidence cutoff -> filters out detections
        dets_strict = detector.predict(img_rgb, conf_threshold=0.99)
        self.assertLessEqual(len(dets_strict), len(dets))


if __name__ == "__main__":
    unittest.main(verbosity=2)
