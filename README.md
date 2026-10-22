# 🛣️ AI-Based Pothole & Road Defect Detector

An end-to-end Computer Vision and Deep Learning web application designed to inspect asphalt road surfaces, detect potholes and structural defects, calculate their geometric image-area coverage, and categorize severity for municipal maintenance triage.

---

## 📌 1. Problem Statement

Road surfaces across India and many developing regions suffer from rapid degradation caused by heavy monsoons, waterlogging, overloaded commercial vehicles, and inadequate drainage. Municipal corporations and highway authorities traditionally rely on manual visual surveys, which are:
- **Slow & Labor-Intensive**: Field teams cannot inspect thousands of lane-kilometers manually in real time.
- **Subjective & Inconsistent**: Human inspection results vary widely between different surveyors.
- **Hazardous**: Surveyors face high risk while inspecting active expressways and arterial corridors.

This application provides an automated, objective, AI-assisted computer vision workflow:
1. Accepts road surface photographs (from smartphone, dashcam, or drone cameras).
2. Localizes potholes, surface cracks, and structural road defects with bounding boxes.
3. Quantifies individual defect pixel area and image coverage percentage.
4. Computes the **mathematical union coverage** across all defects to eliminate double-counting overlapping boxes.
5. Triages defects into **Low**, **Medium**, and **High** severity tiers.
6. Generates downloadable inspection reports (CSV) and annotated imagery for municipal work orders.

---

## 🏗️ 2. Technology Stack

- **Python 3.10 / 3.11**: Core runtime environment.
- **Ultralytics YOLO (YOLOv8)**: Deep learning object detection framework for road defect localization.
- **OpenCV (cv2)**: Image processing, color space transformations, morphological analysis, and text badge rendering.
- **Streamlit**: Responsive, interactive web dashboard.
- **NumPy & Pandas**: Array operations, discrete 2D binary mask union calculations, and tabular report generation.
- **Pillow (PIL)**: Image header validation, EXIF orientation correction, and byte-stream encoding.

---

## 📁 3. Project Structure

```text
pothole-defect-detector/
├── app.py                      # Main Streamlit web application & dashboard
├── detector.py                 # YOLO inference engine & fallback anomaly detector
├── severity.py                 # Geometric area math, union calculation & severity triage
├── utils.py                    # Image validation, OpenCV drawing & CSV/JPG exports
├── requirements.txt            # Project dependencies
├── README.md                   # Complete documentation & beginner guide
├── .gitignore                  # Git exclusions (weights, datasets, caches)
├── create_sample_assets.py     # Generator for realistic synthetic road test images
├── test_integration_run.py     # End-to-end integration test runner
├── models/
│   ├── README.md               # Guide on custom weights, COCO validation & checkpoints
│   └── best.pt                 # (Target location for trained YOLOv8 weights)
├── datasets/
│   ├── README.md               # Public datasets, data.yaml template & download guides
│   └── data.yaml               # YOLO dataset configuration
├── training/
│   └── train.py                # Standalone YOLOv8 training & evaluation script
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py        # Automated test suite (15 unit & integration tests)
├── sample_images/              # Sample road images (clean, potholes, cracks, overlapping)
└── outputs/                    # Exported inspection reports and annotated images
```

---

## 📐 4. Mathematical Methodology

### A. Bounding-Box Area
For an input image with width $W$ and height $H$, the total image area in pixels is:
$$\text{Area}_{\text{Image}} = W \times H$$

For an axis-aligned bounding box defined by coordinates $(x_1, y_1, x_2, y_2)$:
$$\text{Area}_{\text{BBox}} = \max(0, x_2 - x_1) \times \max(0, y_2 - y_1)$$

### B. Individual Defect Coverage Percentage
The fraction of the photograph covered by an individual defect is:
$$\text{Coverage}_{\text{Individual}} = \left( \frac{\text{Area}_{\text{BBox}}}{\text{Area}_{\text{Image}}} \right) \times 100\%$$

### C. Non-Overlapping Total Defect Coverage (Union Area)
When two or more defects intersect or cluster together, naively summing their individual bounding-box areas leads to **double-counting** and inflates the estimated damage:
$$\sum_{i=1}^n \text{Area}(B_i) \ge \text{Area}\left(\bigcup_{i=1}^n B_i\right)$$

To ensure mathematical precision, this application instantiates a 2D discrete binary pixel mask $M \in \{0, 1\}^{H \times W}$:
1. For every detected bounding box $[x_1, y_1, x_2, y_2]$, pixel locations in that rectangle are set to $1$:
   $$M[y_1:y_2, x_1:x_2] = 1$$
2. The true non-overlapping union area is simply the count of non-zero pixels:
   $$\text{Area}_{\text{Union}} = \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} M[y, x]$$
3. Total Union Coverage Percentage:
   $$\text{Coverage}_{\text{Union}} = \left( \frac{\text{Area}_{\text{Union}}}{\text{Area}_{\text{Image}}} \right) \times 100\%$$

### D. Severity Triage Thresholds
The default severity classification thresholds are:
- **Low Severity**: $\text{Coverage} < 1.0\%$
- **Medium Severity**: $1.0\% \le \text{Coverage} < 5.0\%$
- **High Severity**: $\text{Coverage} \ge 5.0\%$

*Note: All severity boundaries are dynamically configurable via the sidebar sliders.*

---

## ⚠️ 5. Engineering & Road Safety Disclaimer

> [!WARNING]
> **Image-area-based severity is a preliminary 2D visual indicator.**  
> A single 2D photograph taken from a mobile phone or dashboard camera **cannot measure 3D pothole depth, volumetric asphalt loss, or subsurface structural base failure**. Official road engineering safety standards (e.g., IRC / AASHTO Pavement Condition Index) require field calibration, road profilometers, stereo-depth sensors, or LiDAR.

---

## 🚀 6. Installation & Execution Guide

### Step 1: Prerequisites
- Python 3.10 or 3.11 installed.
- Ensure `git` and `pip` are accessible from your terminal.

### Step 2: Create a Virtual Environment (Windows)

Open **PowerShell** or **Command Prompt** in the project folder:

```powershell
# In PowerShell:
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*(If using Command Prompt: `venv\Scripts\activate.bat`)*

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Generate Sample Images (Optional but Recommended)

Run the included synthetic asset generator to populate `sample_images/` with test road images:

```powershell
python create_sample_assets.py
```

### Step 5: Run Automated Tests

Verify that all math formulas, image validators, and edge cases pass:

```powershell
python tests/test_pipeline.py
```

You should observe `Ran 15 tests ... OK`.

### Step 6: Launch the Streamlit Web Dashboard

```powershell
streamlit run app.py
```

Your web browser will automatically open to `http://localhost:8501`.

---

## 🎯 7. Model Training & Custom Weights Workflow

### Why Base Pretrained Models Cannot Detect Potholes
Ultralytics base models (like `yolov8n.pt`) are trained on the **COCO dataset** (80 general classes: *person*, *car*, *dog*, *bottle*, etc.). They **do not have a pothole class**. If loaded into this app, the built-in validation engine will flag the class mismatch and reject it to prevent reporting false detections.

### Training Your Own Custom Pothole Detector

1. **Download a Pothole Dataset**:
   - Visit [Roboflow Universe Pothole Datasets](https://universe.roboflow.com/search?q=pothole) or [Kaggle](https://www.kaggle.com/datasets).
   - Export the dataset in **YOLOv8 PyTorch** format.
   - Extract files into the `datasets/` directory (see [`datasets/README.md`](datasets/README.md)).

2. **Configure `datasets/data.yaml`**:
   ```yaml
   path: ./datasets
   train: images/train
   val: images/val
   nc: 2
   names:
     0: pothole
     1: surface_crack
   ```

3. **Launch the Training Pipeline**:
   ```powershell
   python training/train.py --data datasets/data.yaml --epochs 50 --batch 16 --imgsz 640
   ```

4. **Automatic Checkpoint Export**:
   When training completes, `training/train.py` prints:
   - Precision (P)
   - Recall (R)
   - mAP@0.50
   - mAP@0.50:0.95  
   And automatically copies the highest-performing weights to:
   ```
   models/best.pt
   ```

5. **Inference**:
   In the Streamlit app sidebar, choose **"Custom YOLO Model (.pt)"** and point to `models/best.pt`.

---

## 🧪 8. Testing Edge Cases Covered

The test suite in [`tests/test_pipeline.py`](tests/test_pipeline.py) validates:
1. **Multiple Detections**: Verifies correct classification of multiple bounding boxes on one road surface.
2. **Clean Roads (0 Defects)**: Ensures the system reports 0 detections cleanly without crashing.
3. **Invalid Files**: Rejects unsupported formats (`.pdf`, `.txt`), zero-byte files, and corrupted headers.
4. **Overlapping Boxes**: Mathematical assertion proving $\text{Union Area} \le \sum \text{Individual Areas}$.
5. **Extreme Resolutions**: Confirms numerical stability on tiny images ($10 \times 10$) and large images ($4000 \times 3000$).
6. **Severity Threshold Boundaries**: Rigorous testing across default and customized percentage cutoffs.
7. **Missing Weights**: Verifies helpful exception handling when model files are not found on disk.
8. **Sensitivity Adjustment**: Verifies that varying confidence cutoffs dynamically filters detections.

---

## 🛠️ 9. Troubleshooting & FAQ

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| `FileNotFoundError: models/best.pt not found` | Custom weights have not been trained or placed yet. | Select **"Demo / Simulation Mode"** in the sidebar, or run `python training/train.py` to train custom weights. |
| `ValueError: Loaded model appears to be generic COCO` | A standard YOLO model (`yolov8n.pt`) with COCO classes was provided. | Base COCO models cannot detect potholes. Train a pothole-specific model on a road dataset. |
| `ExecutionPolicy Error in PowerShell` | Windows PowerShell restricts script execution by default. | Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate `venv`. |
| Image upload fails or shows format error | File is not JPG, JPEG, or PNG, or file is empty. | Use standard `.jpg` or `.png` photographs with valid headers. |

---

## 🔮 10. Future Improvements

1. **Stereo-Vision & Depth Sensors**: Integrate iPhone LiDAR or stereo cameras to compute true metric pothole depth (cm) and cubic volume ($m^3$) of asphalt required for patching.
2. **GPS Tagging & GIS Mapping**: Extract EXIF GPS coordinates from smartphone photos and plot road defect hotspots on an interactive OpenStreetMap / Folium dashboard.
3. **Instance Segmentation (YOLOv8-Seg)**: Replace rectangular bounding boxes with polygonal defect masks for exact pixel-perfect surface crack boundary delineation.
4. **Edge Deployment**: Quantize model to ONNX / TensorRT for real-time 30+ FPS edge inference on Raspberry Pi or NVIDIA Jetson mounted on municipal vehicles.

