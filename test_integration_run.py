"""
test_integration_run.py - End-to-End Pipeline Verification
==========================================================
Runs the full YOLO detection, enrichment, annotation, and export pipeline
on all sample images generated in sample_images/.
"""

import sys
from pathlib import Path
import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from detector import PotholeDetector
from severity import enrich_detections
from utils import (
    validate_image_file,
    draw_annotations,
    detections_to_dataframe,
    export_dataframe_to_csv,
    convert_image_to_bytes,
)

def run_integration_test():
    print("[*] Starting End-to-End YOLO Integration Test...")
    model_path = Path("models/best.pt")
    if not model_path.exists():
        print(f"[!] Error: Model weights not found at {model_path}")
        sys.exit(1)

    detector = PotholeDetector(model_path=str(model_path))
    sample_files = list(Path("sample_images").glob("*.jpg"))

    if not sample_files:
        print("[!] No sample images found in sample_images/")
        sys.exit(1)

    for img_path in sample_files:
        print(f"\n---> Testing on: {img_path.name}")
        with open(img_path, "rb") as f:
            is_valid, img_np, err = validate_image_file(f)
            assert is_valid, f"Validation failed: {err}"

        h, w = img_np.shape[:2]
        raw_dets = detector.predict(img_np, conf_threshold=0.25)
        print(f"     Raw YOLO detections: {len(raw_dets)}")

        enriched, summary = enrich_detections(
            raw_dets, w, h, low_threshold=1.0, medium_threshold=5.0
        )
        print(f"     Enriched detections: {len(enriched)}")
        print(f"     Severity summary: Low={summary['low_severity_count']}, Med={summary['medium_severity_count']}, High={summary['high_severity_count']}")
        print(f"     Total Union Coverage: {summary['total_union_coverage_pct']:.2f}%")

        annotated = draw_annotations(img_np, enriched)
        assert annotated.shape == img_np.shape

        df = detections_to_dataframe(enriched)
        csv_bytes = export_dataframe_to_csv(df)
        jpg_bytes = convert_image_to_bytes(annotated, "JPEG")

        assert len(csv_bytes) > 0
        assert len(jpg_bytes) > 0

        # Save sample outputs to outputs/
        out_img_path = Path("outputs") / f"test_{img_path.stem}_annotated.jpg"
        out_csv_path = Path("outputs") / f"test_{img_path.stem}_report.csv"
        with open(out_img_path, "wb") as f_img:
            f_img.write(jpg_bytes)
        with open(out_csv_path, "wb") as f_csv:
            f_csv.write(csv_bytes)
        print(f"     [OK] Saved annotated outputs to {out_img_path.name}")

    print("\n[OK] All Integration Tests Passed Successfully!")

if __name__ == "__main__":
    run_integration_test()
