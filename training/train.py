"""
training/train.py - YOLO Pothole Detection Training Pipeline
============================================================

This script trains a custom YOLOv8 model for road defect and pothole detection
using the Ultralytics framework.

Key Capabilities:
1. Validates the existence and format of dataset YAML files.
2. Selects appropriate hardware (GPU if CUDA is available, otherwise CPU).
3. Executes training with data augmentations suitable for road surfaces.
4. Evaluates final validation metrics (Precision, Recall, mAP50, mAP50-95).
5. Automatically exports the top checkpoint to 'models/best.pt'.

Usage:
    python training/train.py --data datasets/data.yaml --epochs 50 --batch 16 --imgsz 640
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import yaml
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 Model on Pothole and Road Defect Datasets."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/data.yaml",
        help="Path to dataset configuration YAML file (default: datasets/data.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base pretrained architecture to fine-tune (e.g., yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16; use 8 if running on CPU or low VRAM).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution in pixels (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: '0' for GPU, 'cpu', or 'auto' (default: auto).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/best.pt",
        help="Destination path to copy best weights (default: models/best.pt).",
    )
    return parser.parse_args()


def validate_dataset_yaml(yaml_path: Path) -> dict:
    """
    Validate that the dataset YAML file exists and contains essential keys.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found at: '{yaml_path}'.\n"
            f"Please download or prepare a YOLO-format dataset and place its YAML configuration at '{yaml_path}'.\n"
            f"See datasets/README.md for download instructions."
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    if not isinstance(data_cfg, dict):
        raise ValueError(f"Invalid YAML structure in '{yaml_path}'.")

    # Check required keys
    required_keys = ["names"]
    for key in required_keys:
        if key not in data_cfg:
            raise KeyError(
                f"Missing required key '{key}' in '{yaml_path}'. "
                f"Ensure the YAML specifies class names (e.g., names: ['pothole'])."
            )

    return data_cfg


def main():
    args = parse_args()

    print("=" * 70)
    print("🛣️  YOLO Pothole & Road Defect Detector - Training Pipeline")
    print("=" * 70)

    # 1. Device selection
    if args.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[*] Compute Device : {device} (CUDA Available: {torch.cuda.is_available()})")
    print(f"[*] Base Model     : {args.model}")
    print(f"[*] Epochs         : {args.epochs}")
    print(f"[*] Batch Size     : {args.batch}")
    print(f"[*] Image Size     : {args.imgsz}")

    # 2. Dataset validation
    data_path = Path(args.data)
    try:
        data_cfg = validate_dataset_yaml(data_path)
        class_names = data_cfg.get("names", [])
        print(f"[*] Dataset Config : {data_path}")
        print(f"[*] Classes        : {class_names}")
    except FileNotFoundError as exc:
        print(f"\n[!] Dataset Setup Error:\n{exc}")
        print("\nTip: Refer to datasets/README.md for one-click download instructions.")
        sys.exit(1)

    # 3. Import Ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] Ultralytics is not installed. Please run: pip install ultralytics")
        sys.exit(1)

    # 4. Initialize model
    print(f"\n[*] Loading base model: {args.model}...")
    model = YOLO(args.model)

    # 5. Execute Training
    print("\n[*] Starting training. This may take several minutes to hours depending on your hardware...")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        plots=True,
        save=True,
        project="runs/detect",
        name="pothole_defect_train",
        exist_ok=True,
    )

    # 6. Evaluate on validation set
    print("\n" + "=" * 70)
    print("📊 Evaluating Validation Metrics...")
    print("=" * 70)

    val_results = model.val()
    metrics = getattr(val_results, "box", None)

    if metrics is not None:
        precision = metrics.mp  # Mean precision
        recall = metrics.mr     # Mean recall
        map50 = metrics.map50   # mAP at IoU 0.50
        map50_95 = metrics.map  # mAP at IoU 0.50:0.95

        print(f"  • Precision (P)    : {precision:.4f} (Accuracy of positive detections)")
        print(f"  • Recall (R)       : {recall:.4f} (Coverage of actual defects found)")
        print(f"  • mAP@0.50         : {map50:.4f} (Standard detection benchmark)")
        print(f"  • mAP@0.50:0.95    : {map50_95:.4f} (Strict multi-threshold benchmark)")
    else:
        print("  • Validation completed. Metrics logged to runs/detect/pothole_defect_train.")

    # 7. Save best model weights
    save_dir = getattr(model.trainer, "save_dir", Path("runs/detect/pothole_defect_train"))
    trained_best_pt = Path(save_dir) / "weights" / "best.pt"

    dest_path = Path(args.output)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if trained_best_pt.exists():
        shutil.copy(trained_best_pt, dest_path)
        print(f"\n[OK] Best model weights successfully copied to: '{dest_path}'")
        print(f"[OK] You can now launch the dashboard with: streamlit run app.py")
    else:
        print(f"[!] Warning: Could not locate '{trained_best_pt}'. Check runs/ directory.")


if __name__ == "__main__":
    main()
