"""
severity.py - Defect Area Calculation & Severity Assessment Module
===================================================================

This module provides functions to:
1. Calculate individual bounding-box areas and image coverage percentages.
2. Classify individual defect severity (Low, Medium, High) based on configurable thresholds.
3. Compute the non-overlapping union coverage of multiple bounding boxes to avoid
   double-counting overlapping defects.
4. Generate structured summary metrics for reporting and dashboard display.

Note:
Image-area-based severity is a 2D geometric visual indicator. It does NOT claim
to measure 3D pothole depth, structural pavement volume loss, or official road
safety safety indices, which require stereoscopic depth sensors or LiDAR.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def calculate_bbox_area(x1: float, y1: float, x2: float, y2: float) -> int:
    """
    Calculate the area of a single axis-aligned bounding box in pixels.

    Formula:
        Area = max(0, x2 - x1) * max(0, y2 - y1)

    Args:
        x1: Left x-coordinate.
        y1: Top y-coordinate.
        x2: Right x-coordinate.
        y2: Bottom y-coordinate.

    Returns:
        Area in integer pixels. Always >= 0.
    """
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    return int(round(width * height))


def calculate_coverage_percentage(bbox_area: float, image_area: float) -> float:
    """
    Calculate the percentage of the total image area occupied by a defect.

    Formula:
        Coverage % = (Bounding-box area / Image area) * 100

    Args:
        bbox_area: Area of the defect bounding box in pixels.
        image_area: Total area of the image in pixels (Width * Height).

    Returns:
        Percentage as a float in range [0.0, 100.0]. Returns 0.0 if image_area <= 0.
    """
    if image_area <= 0:
        return 0.0
    coverage = (float(bbox_area) / float(image_area)) * 100.0
    return min(100.0, max(0.0, coverage))


def classify_severity(
    coverage_percentage: float,
    low_threshold: float = 1.0,
    medium_threshold: float = 5.0,
) -> str:
    """
    Classify the severity rating of a defect based on its image coverage percentage.

    Threshold Logic:
        - Low: coverage < low_threshold (default < 1.0%)
        - Medium: low_threshold <= coverage < medium_threshold (default 1.0% to < 5.0%)
        - High: coverage >= medium_threshold (default >= 5.0%)

    Args:
        coverage_percentage: Percentage of image covered by the defect.
        low_threshold: Upper bound for Low severity.
        medium_threshold: Lower bound for High severity.

    Returns:
        One of 'Low', 'Medium', 'High'.
    """
    if coverage_percentage < low_threshold:
        return "Low"
    elif coverage_percentage < medium_threshold:
        return "Medium"
    else:
        return "High"


def calculate_union_coverage(
    boxes: List[List[int]],
    image_width: int,
    image_height: int,
) -> Tuple[int, float]:
    """
    Calculate the total non-overlapping union area and coverage percentage
    for all detected bounding boxes using a discrete binary pixel mask.

    Why a union mask?
        If two bounding boxes overlap, summing their individual areas would double-count
        the overlapping region. A binary mask marks each pixel as defective at most once,
        giving the exact mathematical union area:
            Total Defective Area = Area(Box_1 ∪ Box_2 ∪ ... ∪ Box_N)

    Args:
        boxes: List of bounding boxes in format [[x1, y1, x2, y2], ...].
        image_width: Total image width in pixels.
        image_height: Total image height in pixels.

    Returns:
        Tuple of:
            - total_union_pixels (int): Non-overlapping pixel count.
            - total_coverage_pct (float): Percentage of image covered [0.0, 100.0].
    """
    if image_width <= 0 or image_height <= 0 or not boxes:
        return 0, 0.0

    # Create binary mask of the image dimensions
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = box

        # Clip coordinates strictly within image boundaries
        x1_clip = max(0, min(image_width, int(round(x1))))
        y1_clip = max(0, min(image_height, int(round(y1))))
        x2_clip = max(0, min(image_width, int(round(x2))))
        y2_clip = max(0, min(image_height, int(round(y2))))

        if x2_clip > x1_clip and y2_clip > y1_clip:
            mask[y1_clip:y2_clip, x1_clip:x2_clip] = 1

    total_union_pixels = int(np.count_nonzero(mask))
    total_image_pixels = image_width * image_height
    total_coverage_pct = (total_union_pixels / total_image_pixels) * 100.0

    return total_union_pixels, round(total_coverage_pct, 3)


def enrich_detections(
    raw_detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    low_threshold: float = 1.0,
    medium_threshold: float = 5.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Take raw model detections and enrich them with geometric area calculations,
    coverage percentages, individual severity ratings, and overall summary stats.

    Args:
        raw_detections: List of detection dicts containing 'id', 'class_name',
                        'confidence', 'bbox' ([x1, y1, x2, y2]).
        image_width: Width of image in pixels.
        image_height: Height of image in pixels.
        low_threshold: Configurable Low severity threshold in %.
        medium_threshold: Configurable Medium severity threshold in %.

    Returns:
        Tuple of:
            - enriched_detections (List[Dict]): List of enriched detection dictionaries.
            - summary (Dict): Overall aggregation including counts, union coverage, etc.
    """
    image_area = max(1, image_width * image_height)
    enriched = []
    boxes_for_union = []
    total_individual_area = 0

    severity_counts = {"Low": 0, "Medium": 0, "High": 0}

    for idx, det in enumerate(raw_detections):
        det_id = det.get("id", f"DEF-{idx + 1:03d}")
        class_name = det.get("class_name", "pothole")
        confidence = float(det.get("confidence", 0.0))
        bbox = det.get("bbox", [0, 0, 0, 0])

        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]

        # Clip box to image bounds for valid area computation
        x1_c = max(0, min(image_width, x1))
        y1_c = max(0, min(image_height, y1))
        x2_c = max(0, min(image_width, x2))
        y2_c = max(0, min(image_height, y2))

        area_px = calculate_bbox_area(x1_c, y1_c, x2_c, y2_c)
        coverage_pct = calculate_coverage_percentage(area_px, image_area)
        severity = classify_severity(coverage_pct, low_threshold, medium_threshold)

        severity_counts[severity] += 1
        total_individual_area += area_px
        boxes_for_union.append([x1_c, y1_c, x2_c, y2_c])

        enriched.append({
            "id": det_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100.0, 1),
            "bbox": [x1_c, y1_c, x2_c, y2_c],
            "bbox_area_px": area_px,
            "coverage_pct": round(coverage_pct, 3),
            "severity": severity,
        })

    # Compute non-overlapping union coverage
    union_pixels, union_coverage_pct = calculate_union_coverage(
        boxes_for_union, image_width, image_height
    )

    overlap_pixels = max(0, total_individual_area - union_pixels)
    overlap_pct = (overlap_pixels / image_area) * 100.0 if image_area > 0 else 0.0

    summary = {
        "total_detections": len(enriched),
        "low_severity_count": severity_counts["Low"],
        "medium_severity_count": severity_counts["Medium"],
        "high_severity_count": severity_counts["High"],
        "image_width": image_width,
        "image_height": image_height,
        "image_area_px": image_area,
        "total_individual_area_px": total_individual_area,
        "total_union_area_px": union_pixels,
        "total_union_coverage_pct": union_coverage_pct,
        "overlap_pixels_saved": overlap_pixels,
        "overlap_pct_saved": round(overlap_pct, 3),
        "low_threshold": low_threshold,
        "medium_threshold": medium_threshold,
    }

    return enriched, summary

