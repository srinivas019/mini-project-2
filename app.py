"""
app.py - AI-Based Pothole & Road Defect Detector
================================================

Streamlit Dashboard for Road Surface Quality Assessment, Defect Detection,
Area Coverage Analysis, and Severity Classification.

Author: Senior Computer Vision & ML Engineering Team
Framework: Streamlit, Ultralytics YOLO, OpenCV, Pandas, NumPy, PIL
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Import internal modules
from detector import PotholeDetector
from severity import enrich_detections
from utils import (
    validate_image_file,
    draw_annotations,
    detections_to_dataframe,
    export_dataframe_to_csv,
    convert_image_to_bytes,
    SEVERITY_COLORS,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pothole & Road Defect Detector",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a clean, modern aesthetic
st.markdown(
    """
    <style>
    /* Metric Card Enhancements */
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: #555;
    }
    .metric-card-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 12px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 12px;
    }
    .disclaimer-box {
        background-color: #FFF9C4;
        border-left: 5px solid #FBC02D;
        padding: 14px 16px;
        border-radius: 6px;
        margin-top: 20px;
        margin-bottom: 20px;
        color: #424242;
        font-size: 0.92rem;
    }
    .formula-badge {
        background-color: #EDE7F6;
        color: #4A148C;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_detection_model(model_path_str: str = "models/best.pt"):
    """
    Load and cache the YOLO detection model.

    Args:
        model_path_str: Path to weights file (.pt).

    Returns:
        Tuple of (detector_instance, error_message: str).
    """
    path = Path(model_path_str)
    if not path.exists():
        return None, (
            f"Weights file not found at '{model_path_str}'.\n\n"
            f"Please ensure trained YOLO weights are placed at '{model_path_str}', "
            f"or run `python training/train.py` to train custom weights."
        )

    try:
        detector = PotholeDetector(model_path=str(path))
        return detector, ""
    except Exception as exc:
        return None, f"Failed to initialize YOLO model: {str(exc)}"


# -----------------------------------------------------------------------------
# 3. Sidebar Configuration Controls
# -----------------------------------------------------------------------------
def render_sidebar() -> Dict[str, Any]:
    """Render sidebar inputs and return user configuration parameters."""
    st.sidebar.title("⚙️ Inspection Controls")

    # A. Model Weights Configuration
    st.sidebar.subheader("1. Model Configuration")
    model_path = st.sidebar.text_input(
        "YOLO Weights File Path:",
        value="models/best.pt",
        help="Path to your trained Ultralytics YOLOv8 weights (.pt).",
    )
    if Path(model_path).exists():
        st.sidebar.success(f"✓ Active Model: `{Path(model_path).name}`")
    else:
        st.sidebar.error(f"⚠️ Model not found at '{model_path}'")

    # B. Detection Confidence Slider
    st.sidebar.subheader("2. Detection Sensitivity")
    conf_threshold = st.sidebar.slider(
        "Confidence Cutoff:",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Higher values suppress uncertain detections. Lower values catch fainter defects.",
    )

    # C. Severity Thresholds Configuration
    st.sidebar.subheader("3. Severity Thresholds (% Area)")
    st.sidebar.caption("Define the image-area boundaries for Low, Medium, and High ratings:")
    low_thresh = st.sidebar.slider(
        "Low Severity Upper Bound (%):",
        min_value=0.2,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Defects covering less than this percentage are categorized as Low severity.",
    )
    med_thresh = st.sidebar.slider(
        "Medium Severity Upper Bound (%):",
        min_value=low_thresh + 0.5,
        max_value=12.0,
        value=5.0,
        step=0.5,
        help="Defects between Low Upper Bound and this value are Medium. Greater or equal is High.",
    )

    # D. Image Input Selection
    st.sidebar.subheader("4. Road Image Source")
    input_source = st.sidebar.radio(
        "Select Image Input:",
        options=["Sample Test Images", "Upload Image"],
        index=0,
    )

    selected_sample = None
    uploaded_file = None

    sample_dir = Path("sample_images")
    if input_source == "Sample Test Images":
        sample_files = {
            "Real Road - Multiple Potholes Cluster (IVCNZ)": "pothole_cluster_road.jpg",
            "Real Road - Severe Highway Pothole (IVCNZ)": "single_severe_pothole.jpg",
            "Real Road - Highway Defects (IVCNZ)": "multiple_potholes_highway.jpg",
            "Real Road - Urban Street Pothole (IVCNZ)": "urban_road_pothole.jpg",
            "Real Road - Intact Surface (Zero Defects Cutoff)": "intact_road_surface.jpg",
        }
        sample_name = st.sidebar.selectbox("Choose Real Road Sample:", list(sample_files.keys()))
        selected_sample = sample_dir / sample_files[sample_name]
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Road Photo (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"],
            help="Upload a road surface photograph taken from vehicle or smartphone perspective.",
        )

    # E. Display Settings
    st.sidebar.subheader("5. Visual Display Options")
    show_labels = st.sidebar.checkbox("Show Class & ID Labels", value=True)
    show_severity = st.sidebar.checkbox("Show Severity Badge", value=True)
    show_coverage = st.sidebar.checkbox("Show Area % on Badge", value=True)

    return {
        "model_path": model_path,
        "conf_threshold": conf_threshold,
        "low_thresh": low_thresh,
        "med_thresh": med_thresh,
        "input_source": input_source,
        "selected_sample": selected_sample,
        "uploaded_file": uploaded_file,
        "show_labels": show_labels,
        "show_severity": show_severity,
        "show_coverage": show_coverage,
    }


# -----------------------------------------------------------------------------
# 4. Main Application UI & Workflow
# -----------------------------------------------------------------------------
def main():
    # Header Section
    st.title("🛣️ AI-Based Pothole & Road Defect Detector")
    st.markdown(
        """
        **Automated Visual Quality & Surface Integrity Inspection System**  
        Road surfaces continuously degrade due to monsoon rains, heavy vehicle traffic, poor drainage, 
        and deferred maintenance. Manual road audits are slow, hazardous, and difficult to scale. 
        This application leverages Computer Vision to localize road defects, compute image-area coverage, 
        and provide severity triage for road maintenance prioritization.
        """
    )

    # Render sidebar and gather configurations
    cfg = render_sidebar()

    # Step 1: Resolve and Validate Input Image
    img_np: Optional[np.ndarray] = None
    image_name = "road_inspection"

    if cfg["input_source"] == "Sample Test Images":
        sample_path = cfg["selected_sample"]
        if sample_path and sample_path.exists():
            image_name = sample_path.stem
            with open(sample_path, "rb") as f:
                is_valid, img_np, err_msg = validate_image_file(f)
                if not is_valid:
                    st.error(f"Error loading sample image: {err_msg}")
                    return
        else:
            st.warning("Sample images not found. Run `python create_sample_assets.py` to generate them.")
            return
    else:
        if cfg["uploaded_file"] is None:
            st.info(
                "👈 Please upload a road photograph (JPG, JPEG, or PNG) via the sidebar "
                "or switch to **Sample Test Images** to inspect."
            )
            return
        image_name = cfg["uploaded_file"].name.rsplit(".", 1)[0]
        is_valid, img_np, err_msg = validate_image_file(cfg["uploaded_file"])
        if not is_valid:
            st.error(f"⚠️ Invalid image file: {err_msg}")
            return

    # Step 2: Load YOLO Model
    detector, model_err = load_detection_model(cfg["model_path"])
    if model_err:
        st.error(f"⚠️ Model Initialization Failed:\n\n{model_err}")
        return

    # Step 3: Run Inference with Spinner
    with st.spinner("🔍 Running YOLOv8 road defect detection..."):
        try:
            raw_detections = detector.predict(
                image=img_np,
                conf_threshold=cfg["conf_threshold"],
            )
        except Exception as exc:
            st.error(f"Inference error during defect detection: {str(exc)}")
            return

    # Step 4: Enrich Detections with Area Math & Severity Ratings
    img_h, img_w = img_np.shape[:2]
    enriched_detections, summary = enrich_detections(
        raw_detections=raw_detections,
        image_width=img_w,
        image_height=img_h,
        low_threshold=cfg["low_thresh"],
        medium_threshold=cfg["med_thresh"],
    )

    # Step 5: Render Annotations
    annotated_np = draw_annotations(
        image=img_np,
        detections=enriched_detections,
        show_labels=cfg["show_labels"],
        show_severity=cfg["show_severity"],
        show_coverage=cfg["show_coverage"],
    )

    # -------------------------------------------------------------------------
    # Visual Dashboard Layout
    # -------------------------------------------------------------------------
    st.markdown("---")

    # A. Top KPI Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Defects", summary["total_detections"])
    with m2:
        st.metric("Low Severity", summary["low_severity_count"], help=f"Coverage < {cfg['low_thresh']}%")
    with m3:
        st.metric("Medium Severity", summary["medium_severity_count"], help=f"{cfg['low_thresh']}% to < {cfg['med_thresh']}%")
    with m4:
        st.metric("High Severity", summary["high_severity_count"], help=f"Coverage ≥ {cfg['med_thresh']}%")
    with m5:
        st.metric(
            "Total Defect Coverage",
            f"{summary['total_union_coverage_pct']:.2f}%",
            help="Non-overlapping union area of all detected bounding boxes",
        )

    # Status Notification if no defects detected
    if summary["total_detections"] == 0:
        st.success(
            "✅ **Road Surface Clear**: No potholes or road defects detected at confidence "
            f"threshold `{cfg['conf_threshold']:.2f}`. The analyzed pavement appears intact, "
            "or defects fall below the current sensitivity cutoff."
        )

    # B. Image Comparison Columns
    col_orig, col_annotated = st.columns(2)
    with col_orig:
        st.subheader("📷 Original Road Photograph")
        st.image(img_np, caption=f"Resolution: {img_w} × {img_h} px", use_container_width=True)

    with col_annotated:
        st.subheader("🎯 YOLOv8 Defect Localization & Severity Map")
        st.image(
            annotated_np,
            caption=(
                f"YOLOv8 Engine | Detections: {summary['total_detections']} | "
                f"Union Coverage: {summary['total_union_coverage_pct']:.2f}%"
            ),
            use_container_width=True,
        )

    # C. Action Downloads (Image & CSV)
    down_col1, down_col2 = st.columns(2)
    with down_col1:
        img_bytes = convert_image_to_bytes(annotated_np, format="JPEG")
        st.download_button(
            label="📥 Download Annotated Image (JPG)",
            data=img_bytes,
            file_name=f"{image_name}_annotated.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

    df_detections = detections_to_dataframe(enriched_detections)
    with down_col2:
        csv_bytes = export_dataframe_to_csv(df_detections)
        st.download_button(
            label="📊 Download Detection Report (CSV)",
            data=csv_bytes,
            file_name=f"{image_name}_defect_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # D. Detailed Detection Table
    st.markdown("### 📋 Defect Inventory & Geometric Measurements")
    if summary["total_detections"] > 0:
        # Style dataframe severity column
        def highlight_severity(row):
            sev = row["Severity"]
            if sev == "High":
                return ["background-color: #ffebee; font-weight: bold;"] * len(row)
            elif sev == "Medium":
                return ["background-color: #fff3e0;"] * len(row)
            else:
                return ["background-color: #e8f5e9;"] * len(row)

        styled_df = df_detections.style.apply(highlight_severity, axis=1)
        st.dataframe(styled_df, use_container_width=True)

        if summary["overlap_pixels_saved"] > 0:
            st.info(
                f"ℹ️ **Overlap Elimination**: {summary['overlap_pixels_saved']:,} overlapping pixels "
                f"({summary['overlap_pct_saved']:.2f}% of image) were deduplicated via binary mask union, "
                f"preventing inflated coverage calculations."
            )
    else:
        st.dataframe(df_detections, use_container_width=True)

    # E. Engineering Methodology & Disclaimers
    with st.expander("📐 Technical Methodology: Area Calculation & Severity Logic"):
        st.markdown(
            r"""
            #### 1. Image & Bounding Box Dimensions
            For an input image with width $W$ and height $H$, the total image area is:
            $$\text{Area}_{\text{Image}} = W \times H$$

            For each detected bounding box defined by coordinates $(x_1, y_1, x_2, y_2)$:
            $$\text{Area}_{\text{BBox}} = \max(0, x_2 - x_1) \times \max(0, y_2 - y_1)$$

            #### 2. Individual Defect Coverage Percentage
            $$\text{Coverage}_{\text{Individual}} = \left( \frac{\text{Area}_{\text{BBox}}}{\text{Area}_{\text{Image}}} \right) \times 100\%$$

            #### 3. Non-Overlapping Total Defect Coverage (Union Area)
            When multiple defects are located close to each other, simply summing individual bounding-box areas double-counts overlapping regions.
            To compute the true mathematical union:
            $$\text{Area}_{\text{Union}} = \text{Area}(B_1 \cup B_2 \cup \dots \cup B_n)$$
            This application generates a discrete binary pixel mask $M \in \{0, 1\}^{H \times W}$, painting each detected bounding box onto the mask. The total union coverage is computed as:
            $$\text{Total Coverage \%} = \left( \frac{\sum_{i=1}^H \sum_{j=1}^W M_{i,j}}{W \times H} \right) \times 100\%$$

            #### 4. Severity Threshold Classification
            - **Low Severity**: $\text{Coverage} < 1.0\%$ *(configurable in sidebar)*
            - **Medium Severity**: $1.0\% \le \text{Coverage} < 5.0\%$
            - **High Severity**: $\text{Coverage} \ge 5.0\%$
            """
        )

    # Mandatory Engineering Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Engineering & Road Safety Disclaimer:</strong><br>
            Image-area-based severity calculations in this software represent <em>preliminary 2D visual indicators</em>. 
            A standard monocular photograph cannot measure <strong>pothole depth</strong>, volumetric material displacement, 
            or subsurface structural base deterioration. Official municipal maintenance classifications require 
            field survey validation or sensor-fused 3D instrumentation (such as stereo cameras, road profilometers, or LiDAR).
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
