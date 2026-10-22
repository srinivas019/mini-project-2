"""
utils.py - Image Validation, Visualization, and Export Utilities
================================================================

This module provides helper functions for:
1. Validating uploaded image files (extension, format, integrity, dimensions).
2. Drawing crisp, professional bounding boxes and severity badges on images.
3. Converting detection results into structured Pandas DataFrames.
4. Exporting reports to CSV and annotated images to downloadable byte buffers.
"""

import io
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2


# Color palette for defect severity (in RGB and BGR formats)
# High: Crisp Crimson Red
# Medium: Vibrant Amber Orange
# Low: Bright Forest Green / Lime
SEVERITY_COLORS = {
    "High": {
        "rgb": (230, 40, 40),
        "bgr": (40, 40, 230),
        "hex": "#E62828",
    },
    "Medium": {
        "rgb": (245, 140, 20),
        "bgr": (20, 140, 245),
        "hex": "#F58C14",
    },
    "Low": {
        "rgb": (34, 180, 85),
        "bgr": (85, 180, 34),
        "hex": "#22B455",
    },
}

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def validate_image_file(uploaded_file: Any) -> Tuple[bool, Optional[np.ndarray], str]:
    """
    Validate that an uploaded file is a legitimate, non-corrupt road image.

    Checks:
    1. File is provided and not empty.
    2. File extension is in [jpg, jpeg, png].
    3. PIL can open and verify the image headers.
    4. Image dimensions (width, height) are strictly > 0.
    5. Converts to standard 3-channel RGB numpy array.

    Args:
        uploaded_file: Streamlit UploadedFile object or file-like object with .read()/.name.

    Returns:
        Tuple of (is_valid: bool, image_array: np.ndarray | None, error_message: str).
    """
    if uploaded_file is None:
        return False, None, "No file was uploaded."

    filename = getattr(uploaded_file, "name", "uploaded_image.jpg")
    extension = filename.split(".")[-1].lower() if "." in filename else ""

    if extension not in ALLOWED_EXTENSIONS:
        return (
            False,
            None,
            f"Unsupported file format '.{extension}'. Please upload a JPG, JPEG, or PNG image.",
        )

    try:
        # Reset file pointer to beginning in case it was read previously
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        file_bytes = uploaded_file.read()
        if len(file_bytes) == 0:
            return False, None, "The uploaded file is empty (0 bytes)."

        # Open and verify with PIL
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.verify()  # Verifies image integrity

        # Reopen to load pixel data (verify() closes or invalidates the handle)
        pil_img = Image.open(io.BytesIO(file_bytes))
        # Handle EXIF orientation if present
        pil_img = ImageOps.exif_transpose(pil_img)

        # Convert to RGB (handles grayscale, RGBA, palette modes cleanly)
        pil_img = pil_img.convert("RGB")
        img_np = np.array(pil_img)

        h, w, c = img_np.shape
        if h <= 0 or w <= 0:
            return False, None, f"Invalid image dimensions: {w}x{h}."

        return True, img_np, ""

    except Exception as exc:
        return False, None, f"Failed to parse image file: {str(exc)}"


def draw_annotations(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    show_labels: bool = True,
    show_severity: bool = True,
    show_coverage: bool = True,
    line_thickness: Optional[int] = None,
) -> np.ndarray:
    """
    Render bounding boxes, detection IDs, class labels, and severity tags
    directly onto an image with high-contrast text badges.

    Args:
        image: RGB NumPy array representing the road image.
        detections: List of enriched detection dictionaries from severity.py.
        show_labels: Whether to render class names.
        show_severity: Whether to append severity rating (Low/Medium/High).
        show_coverage: Whether to show individual coverage percentage.
        line_thickness: Optional line thickness. Defaults dynamically based on image size.

    Returns:
        Annotated RGB NumPy array.
    """
    annotated = image.copy()
    h, w, _ = annotated.shape

    # Determine dynamic line thickness and font scale based on resolution
    base_dim = max(h, w)
    thickness = line_thickness if line_thickness else max(2, int(round(base_dim / 400)))
    font_scale = max(0.45, base_dim / 1200.0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        severity = det.get("severity", "Low")
        det_id = det.get("id", "DEF")
        class_name = det.get("class_name", "defect").title()
        conf_pct = det.get("confidence_pct", 0.0)
        coverage_pct = det.get("coverage_pct", 0.0)

        x1, y1, x2, y2 = bbox

        # Get color in RGB
        color_rgb = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["Low"])["rgb"]

        # Draw main bounding box
        cv2.rectangle(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color=color_rgb,
            thickness=thickness,
        )

        if not show_labels:
            continue

        # Build informative label string
        parts = [f"[{det_id}] {class_name} {conf_pct:.0f}%"]
        tag_parts = []
        if show_severity:
            tag_parts.append(severity)
        if show_coverage:
            tag_parts.append(f"{coverage_pct:.1f}% area")

        if tag_parts:
            parts.append(f"({' | '.join(tag_parts)})")

        label_text = " ".join(parts)

        # Calculate text dimensions for background badge
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, font, fontScale=font_scale, thickness=max(1, thickness - 1)
        )

        # Place badge above box if space permits, otherwise just inside top of box
        badge_y1 = max(0, y1 - text_h - baseline - 6)
        badge_y2 = badge_y1 + text_h + baseline + 6
        badge_x1 = max(0, x1)
        badge_x2 = min(w, badge_x1 + text_w + 10)

        # Draw filled badge rectangle
        cv2.rectangle(
            annotated,
            (badge_x1, badge_y1),
            (badge_x2, badge_y2),
            color=color_rgb,
            thickness=-1,
        )

        # Put high-contrast white text on badge
        text_origin = (badge_x1 + 5, badge_y2 - baseline - 3)
        cv2.putText(
            annotated,
            label_text,
            text_origin,
            font,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=max(1, thickness - 1),
            lineType=cv2.LINE_AA,
        )

    return annotated


def detections_to_dataframe(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of enriched detections into a clean, display-ready Pandas DataFrame.

    Args:
        detections: List of enriched detection dictionaries.

    Returns:
        pd.DataFrame formatted for Streamlit table display and CSV export.
    """
    if not detections:
        return pd.DataFrame(
            columns=[
                "Detection ID",
                "Defect Type",
                "Confidence (%)",
                "BBox [x1, y1, x2, y2]",
                "Area (Pixels)",
                "Image Coverage (%)",
                "Severity",
            ]
        )

    rows = []
    for det in detections:
        bbox_str = f"[{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]"
        rows.append({
            "Detection ID": det["id"],
            "Defect Type": det["class_name"].capitalize(),
            "Confidence (%)": f"{det['confidence_pct']:.1f}%",
            "BBox [x1, y1, x2, y2]": bbox_str,
            "Area (Pixels)": f"{det['bbox_area_px']:,}",
            "Image Coverage (%)": f"{det['coverage_pct']:.2f}%",
            "Severity": det["severity"],
        })

    return pd.DataFrame(rows)


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convert a Pandas DataFrame into UTF-8 encoded CSV bytes for download.

    Args:
        df: Pandas DataFrame.

    Returns:
        bytes representing CSV content.
    """
    return df.to_csv(index=False).encode("utf-8")


def convert_image_to_bytes(image: np.ndarray, format: str = "JPEG") -> bytes:
    """
    Convert an RGB numpy array image into byte buffer for download.

    Args:
        image: RGB NumPy array.
        format: Format string, e.g. 'JPEG' or 'PNG'.

    Returns:
        Bytes of encoded image.
    """
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    if format.upper() in ["JPG", "JPEG"]:
        pil_img.save(buf, format="JPEG", quality=92)
    else:
        pil_img.save(buf, format="PNG")
    return buf.getvalue()

