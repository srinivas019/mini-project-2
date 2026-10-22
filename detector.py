"""
detector.py - YOLO Object Detection & Model Inference Engine
=============================================================

This module manages YOLO model loading, validation, and inference for detecting
potholes and road surface defects.

Key Design Principles:
1. Pothole-Specific Enforcement:
   A standard pretrained COCO YOLO model detects objects like 'person', 'car', or
   'dog' — it CANNOT detect potholes without custom training. This module verifies
   model classes to prevent misinterpreting general models as pothole detectors.
2. Standardized Output Contract:
   Outputs standardized detection dictionaries with unique IDs, coordinates,
   confidence scores, and defect types.
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Known road defect keywords to verify model specialization
ROAD_DEFECT_KEYWORDS = {
    "pothole",
    "potholes",
    "crack",
    "cracks",
    "alligator_crack",
    "alligator crack",
    "longitudinal_crack",
    "transverse_crack",
    "rutting",
    "bump",
    "patch",
    "manhole",
    "surface defect",
    "defect",
    "road_defect",
    "0",  # common index label for single-class pothole datasets
}

# Standard COCO 80 classes indicator
COCO_SAMPLE_CLASSES = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train"}


class PotholeDetector:
    """
    Ultralytics YOLO-based road defect detector.
    """

    def __init__(self, model_path: str = "models/best.pt"):
        """
        Initialize the detector with weights from model_path.

        Args:
            model_path: File path to the trained YOLO weights (.pt file).

        Raises:
            FileNotFoundError: If the model weight file does not exist.
            ValueError: If the loaded model is a generic COCO model instead of a
                        pothole/road defect model.
        """
        self.model_path = Path(model_path)
        self.model = None
        self.class_names: Dict[int, str] = {}
        self.is_custom_pothole_model = False

        self._load_model()

    def _load_model(self) -> None:
        """Load and validate the YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights file not found at: '{self.model_path}'.\n"
                f"Please ensure trained YOLO weights are placed at '{self.model_path}', "
                f"or run 'python training/train.py' to train on a custom dataset."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is not installed. Run 'pip install ultralytics' to install it."
            ) from exc

        # Load weights
        self.model = YOLO(str(self.model_path))

        # Inspect and normalize class names
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            self.class_names = names
        elif isinstance(names, list):
            self.class_names = {idx: name for idx, name in enumerate(names)}
        else:
            self.class_names = {0: "pothole"}

        # If model class label is numeric ('0') or generic, map it cleanly to 'pothole'
        for k, v in list(self.class_names.items()):
            if str(v).strip() in {"0", "defect", "road_defect"}:
                self.class_names[k] = "pothole"

        # Check if model has pothole/defect classes or if it is generic COCO
        detected_names_lower = {str(n).lower().strip() for n in self.class_names.values()}

        has_defect_class = any(
            any(keyword in name for keyword in ROAD_DEFECT_KEYWORDS)
            for name in detected_names_lower
        )
        is_generic_coco = bool(COCO_SAMPLE_CLASSES.intersection(detected_names_lower))

        if is_generic_coco and not has_defect_class:
            raise ValueError(
                f"The model at '{self.model_path}' appears to be a generic COCO object detector "
                f"with {len(self.class_names)} classes (e.g. 'person', 'car', 'bicycle').\n"
                f"It is NOT trained on potholes. Please provide weights trained specifically on "
                f"a pothole dataset (e.g., trained via training/train.py)."
            )

        self.is_custom_pothole_model = True

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Run object detection on an input road image.

        Args:
            image: RGB NumPy array representing the road image.
            conf_threshold: Detection confidence cutoff (0.0 to 1.0).
            iou_threshold: Non-Maximum Suppression (NMS) IoU cutoff.

        Returns:
            List of detection dicts:
            [
                {
                    "id": "DEF-001",
                    "class_id": 0,
                    "class_name": "pothole",
                    "confidence": 0.88,
                    "bbox": [x1, y1, x2, y2]
                },
                ...
            ]
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return []

        # Run inference via Ultralytics YOLO
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if not results:
            return detections

        first_result = results[0]
        boxes = getattr(first_result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            cls_int = int(cls_id)
            class_name = self.class_names.get(cls_int, "pothole")

            x1 = max(0, min(w, int(round(box[0]))))
            y1 = max(0, min(h, int(round(box[1]))))
            x2 = max(0, min(w, int(round(box[2]))))
            y2 = max(0, min(h, int(round(box[3]))))

            # Skip degenerate zero-area boxes
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "id": f"DEF-{len(detections) + 1:03d}",
                "class_id": cls_int,
                "class_name": class_name,
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
            })

        return detections
