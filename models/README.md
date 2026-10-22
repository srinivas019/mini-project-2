# Model Storage & Checkpoints Directory

This directory stores the trained YOLO model weights (`best.pt`) used for detecting potholes and road defects.

---

## ⚠️ Important Note on Pretrained Models

A standard pretrained YOLO model (such as `yolov8n.pt` downloaded from Ultralytics) was trained on the **COCO dataset**, which contains 80 general object classes (e.g., *person*, *bicycle*, *car*, *dog*, *traffic light*).

**Standard COCO models CANNOT detect potholes or road cracks.**
The application includes automated validation: if a user attempts to load a standard COCO model, the system flags the class mismatch and will not falsely report cars or pedestrians as road defects.

---

## 🛠️ How to Obtain or Train Model Weights

### Option 1: Train Your Own YOLOv8 Pothole Model (Recommended)
1. Download a YOLO-formatted dataset (see [`datasets/README.md`](../datasets/README.md)).
2. Run the included training script:
   ```bash
   python training/train.py --data datasets/data.yaml --epochs 50 --batch 16
   ```
3. Upon completion, the script automatically copies the best checkpoint to:
   ```
   models/best.pt
   ```

### Option 2: Download Verified Open-Source Pothole Weights
You can download verified community checkpoints trained specifically on road defect datasets:
- **Roboflow Universe Pothole Checkpoints**: [Roboflow Universe](https://universe.roboflow.com/search?q=pothole) (Filter by Model Weights -> YOLOv8 PyTorch).
- **GitHub Community Releases**: Search for open-source RDD2022 / Pothole YOLOv8 weights.
- Save the downloaded `.pt` file as `models/best.pt`.

### Option 3: Explore in Demo / Simulation Mode
If you haven't trained weights yet, simply launch the Streamlit app:
```bash
streamlit run app.py
```
In the sidebar under **Model Configuration**, select **"Demo / Simulation Mode"**. This enables an educational computer-vision pipeline that identifies surface texture anomalies and depression contours, allowing you to experience bounding-box rendering, area calculations, severity classification, and CSV reports immediately without waiting for model training.

---

## Directory Layout
```
models/
├── README.md        <- This guide
└── best.pt          <- Place your trained YOLOv8 pothole weights here
```

