# Pothole & Road Defect Dataset Guide

This project is powered by the **IVCNZ Real-World Pothole Dataset** (from the 35th International Conference on Image and Vision Computing New Zealand).

- **Official Release Source**: [jaygala24/pothole-detection Releases v1.0.0](https://github.com/jaygala24/pothole-detection/releases/download/v1.0.0/Pothole.Dataset.IVCNZ.zip)
- **Dataset Size**: 1,243 real-world road photographs with corresponding YOLO bounding-box annotation text files.
- **Reference**:
  > *Chitale, Pranjal A., et al. "Pothole Detection and Dimension Estimation System using Deep Learning (YOLO) and Image Processing." 2020 35th International Conference on Image and Vision Computing New Zealand (IVCNZ).*

---

## 1. Dataset Directory Organization

The dataset has been organized into training and validation splits ready for Ultralytics YOLOv8:

```
datasets/
├── data.yaml            <- Dataset configuration specifying image paths & class 0: pothole
├── images/
│   ├── train/           <- 994 real road training photos
│   └── val/             <- 249 real road validation photos
└── labels/
    ├── train/           <- 994 corresponding YOLO format bounding-box annotation files
    └── val/             <- 249 corresponding YOLO format bounding-box annotation files
```

---

## 2. YOLO Annotation Format

Every `.jpg` file has a matching `.txt` file with identical filename:
```text
<class_id> <x_center> <y_center> <width> <height>
```
Where all bounding-box coordinates are normalized in the range `[0.0, 1.0]`:
- `class_id`: `0` (pothole)
- `x_center`, `y_center`: Center coordinates relative to image dimensions.
- `width`, `height`: Box dimensions relative to image dimensions.

---

## 3. Training with this Dataset

To train or fine-tune YOLOv8 on this real-world dataset:
```powershell
python training/train.py --data datasets/data.yaml --epochs 50 --batch 16
```
The script will evaluate precision, recall, mAP50, and mAP50-95 on the 249 real validation photos, saving the top weights to `models/best.pt`.
