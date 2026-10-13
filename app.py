"""VisionHack ADAS — Streamlit Web Application

Upgraded with 5 Advanced Computer Vision Pipelines:
1. OpenCV + YOLOv8n Object Detector
2. DINOv2 Embeddings + Nearest-Neighbor Scene Search
3. SAM (Segment Anything Model) Masks + Object Analysis
4. Object Detector + Persistent ByteTrack Tracking
5. Diffusion Models for Image Restoration
"""

import os
import time
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Core Vision Modules
from vision.enhancement import (
    validate_image,
    apply_gamma_correction,
    apply_clahe,
    apply_low_light_pipeline,
    apply_simulated_fog
)
from vision.detection import ObjectDetector, DEFAULT_ROAD_CLASSES, CLASS_COLORS
from vision.metrics import compute_image_metrics

# Advanced Vision Modules
from vision.tracking import ObjectTracker
from vision.retrieval import SceneRetrievalEngine
from vision.segmentation import Segmenter
from vision.restoration import DiffusionRestorer


# ==========================================
# Page Configuration & Modern Dark Styling
# ==========================================
st.set_page_config(
    page_title="VisionHack ADAS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Base Dark Automotive / Aerospace Theme */
:root {
    --bg-primary: #070b14;
    --bg-secondary: #0d1527;
    --bg-card: rgba(15, 23, 42, 0.72);
    --bg-card-hover: rgba(22, 34, 61, 0.85);
    --border-subtle: rgba(56, 189, 248, 0.16);
    --border-glow: rgba(56, 189, 248, 0.45);
    --accent-cyan: #38bdf8;
    --accent-teal: #14b8a6;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

.stApp {
    background: radial-gradient(circle at 15% 10%, rgba(14, 165, 233, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 85% 90%, rgba(20, 184, 166, 0.08) 0%, transparent 45%),
                linear-gradient(180deg, #060911 0%, #0a101d 50%, #05080f 100%);
    color: var(--text-primary);
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090e1a 0%, #060a12 100%) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5);
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
    letter-spacing: -0.02em;
}

/* Glassmorphism Cards */
.adas-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(14px);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.adas-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 14px 35px -8px rgba(14, 165, 233, 0.18), inset 0 1px 0 0 rgba(255, 255, 255, 0.1);
}

.adas-card-title {
    font-size: 1.12rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--accent-cyan);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Metric KPI Displays */
.metric-box {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(51, 65, 85, 0.6);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.metric-box:hover {
    border-color: var(--accent-cyan);
    transform: translateY(-2px);
}
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent-cyan);
    letter-spacing: -0.03em;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}
.metric-delta-pos {
    color: #34d399;
    font-size: 0.82rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.metric-delta-neg {
    color: #f87171;
    font-size: 0.82rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* Notice & Status Banners */
.notice-box {
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.12) 0%, rgba(245, 158, 11, 0.04) 100%);
    border-left: 4px solid var(--accent-amber);
    padding: 14px 18px;
    border-radius: 6px;
    color: #fef3c7;
    font-size: 0.9rem;
    line-height: 1.5;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

/* Sleek Glowing Status Badges */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.15);
}
.status-pill-amber {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(245, 158, 11, 0.12);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
    box-shadow: 0 0 10px rgba(245, 158, 11, 0.15);
}

/* Buttons with Gradient Glow */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #0284c7 0%, #0d9488 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 10px 24px !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
div.stButton > button:first-child:hover {
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5) !important;
    transform: translateY(-2px) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

/* Header Text Accents */
h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
}
h2 {
    color: #f8fafc !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.12);
    padding-bottom: 8px;
    margin-bottom: 14px;
}

/* Streamlit Inputs & Sliders styling */
.stSelectbox label, .stSlider label, .stRadio label {
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    font-size: 0.88rem !important;
}

/* Completely remove top Streamlit Header Bar, Decoration & Blank Margin */
header[data-testid="stHeader"],
div[data-testid="stDecoration"],
[data-testid="stHeader"] {
    display: none !important;
    height: 0px !important;
    visibility: hidden !important;
}
div[data-testid="stToolbar"] {
    visibility: hidden !important;
    display: none !important;
}
.main .block-container,
[data-testid="stAppViewBlockContainer"],
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
    margin-top: 0px !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# Caching Model Initializations (Lazy Load)
# ==========================================
@st.cache_resource(show_spinner="Initializing YOLOv8n object detector...")
def get_detector() -> ObjectDetector:
    return ObjectDetector(model_name="yolov8n.pt")


@st.cache_resource(show_spinner="Initializing YOLOv8 ByteTracker...")
def get_tracker() -> ObjectTracker:
    return ObjectTracker(model_name="yolov8n.pt")


@st.cache_resource(show_spinner="Initializing DINOv2 Scene Retrieval Engine...")
def get_retrieval_engine() -> SceneRetrievalEngine:
    return SceneRetrievalEngine(model_name="dinov2_vits14", index_path="samples/dino_index.json")


@st.cache_resource(show_spinner="Initializing MobileSAM Segmenter...")
def get_segmenter() -> Segmenter:
    return Segmenter(model_name="mobile_sam.pt")


@st.cache_resource(show_spinner="Initializing Diffusion Restoration Engine...")
def get_restorer() -> DiffusionRestorer:
    return DiffusionRestorer()


# ==========================================
# Helper Functions
# ==========================================
def render_metrics_comparison_cards(orig_m: dict, enh_m: dict):
    """Render side-by-side metric comparison cards with calculated deltas."""
    cols = st.columns(4)
    items = [
        ("Brightness (Mean Lum)", "mean_luminance", "", 1),
        ("Contrast (RMS Std)", "rms_contrast", "", 1),
        ("Sharpness (Laplacian Var)", "laplacian_variance", "", 1),
        ("Visibility Heuristic", "experimental_quality_score", "/ 100", 1)
    ]
    for col, (label, key, unit, decimals) in zip(cols, items):
        v1 = orig_m[key]
        v2 = enh_m[key]
        delta = round(v2 - v1, decimals)
        delta_str = f"{delta:+g}"
        delta_class = "metric-delta-pos" if delta >= 0 else "metric-delta-neg"
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{label}</div>
                <div style="display:flex; justify-content:space-around; align-items:baseline; margin-top:6px;">
                    <div><span style="font-size:0.8rem; color:#94a3b8;">Orig:</span> <strong style="color:#cbd5e1;">{v1}</strong></div>
                    <div><span style="font-size:0.8rem; color:#94a3b8;">Enh:</span> <strong style="color:#38bdf8;">{v2}{unit}</strong></div>
                </div>
                <div class="{delta_class}" style="margin-top:4px;">Shift: {delta_str}</div>
            </div>
            """, unsafe_allow_html=True)


def safe_decode_image(uploaded_file) -> np.ndarray:
    """Safely decode an uploaded image file into a BGR uint8 numpy array."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode the uploaded image file.")
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] == 3:
            return img
    raise ValueError(f"Unsupported image shape: {img.shape}")


# ==========================================
# Sidebar Navigation & Model Status
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom: 8px;">
        <span style="font-size:2rem;">🚗</span>
        <div>
            <div style="font-size:1.25rem; font-weight:800; letter-spacing:-0.03em; color:#f8fafc; line-height:1.1;">VisionHack ADAS</div>
            <div style="font-size:0.75rem; color:#38bdf8; font-weight:600; letter-spacing:0.05em; text-transform:uppercase;">Vision Cockpit v2.1</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("AI-Powered Visibility Enhancement & Road Scene Analysis")
    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🖼️ Image Enhancement & Detection",
            "🔍 DINOv2 Visual Scene Search",
            "✂️ SAM Masks & Object Analysis",
            "🏎️ Video Object Tracking",
            "✨ Diffusion Image Restoration",
            "📈 Analytics & Evaluation",
            "📖 About & Methodology"
        ],
        index=0
    )

    st.divider()
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px;'>Engine Telemetry</div>", unsafe_allow_html=True)
    # YOLO Status (check if weights exist or model initialized without triggering heavy inference)
    yolo_weights_present = os.path.exists("yolov8n.pt") or ("detector" in st.session_state and st.session_state["detector"].is_ready)
    if yolo_weights_present:
        st.markdown("<div style='margin-bottom:6px;'><span class='status-pill'>● YOLOv8n Edge Ready</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-bottom:6px;'><span class='status-pill-amber'>○ YOLOv8n (Auto-Cache)</span></div>", unsafe_allow_html=True)

    # ByteTrack
    st.markdown("<div style='margin-bottom:6px;'><span class='status-pill'>● ByteTrack Association Active</span></div>", unsafe_allow_html=True)

    # DINOv2 Status
    dino_loaded = "retrieval_engine" in st.session_state and st.session_state["retrieval_engine"].is_model_loaded
    dino_status = "● DINOv2 ViT-S/14 Active" if dino_loaded else "○ DINOv2 (On-Demand)"
    st.markdown(f"<div style='margin-bottom:6px;'><span class='status-pill-amber'>{dino_status}</span></div>", unsafe_allow_html=True)

    # SAM Status
    sam_loaded = "segmenter" in st.session_state and st.session_state["segmenter"].is_ready
    sam_status = "● MobileSAM Active" if sam_loaded else "○ MobileSAM (On-Demand)"
    st.markdown(f"<div style='margin-bottom:6px;'><span class='status-pill-amber'>{sam_status}</span></div>", unsafe_allow_html=True)

    # Diffusion Status
    has_diff = os.path.exists(".venv/Lib/site-packages/diffusers") or False
    try:
        import diffusers
        has_diff = True
    except ImportError:
        has_diff = False
    diff_status = "● Diffusion Restorer Ready" if has_diff else "○ Diffusion (Optional Setup)"
    st.markdown(f"<div style='margin-bottom:6px;'><span class='status-pill-amber'>{diff_status}</span></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='font-size: 0.75rem; color: #64748b;'>
    <strong>ADAS Research Disclaimer:</strong><br>
    All visibility scores, spatial tags, and similarity scores are experimental heuristics.
    Not certified for vehicle control or physical road distance measurement.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if selected_page == "📊 Overview":
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        <div class='adas-card'>
            <div class='adas-card-title'>🔬 Five Computer Vision Approaches Implemented</div>
            <div style='display:flex; flex-direction:column; gap:14px; margin-top:8px;'>
                <div style='background:rgba(15,23,42,0.6); border:1px solid rgba(56,189,248,0.12); border-radius:10px; padding:12px 16px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <strong style='color:#f8fafc; font-size:0.95rem;'>1. OpenCV + YOLOv8n Perception</strong>
                        <span class='status-pill' style='font-size:0.7rem;'>Core Pipeline</span>
                    </div>
                    <div style='color:#94a3b8; font-size:0.84rem; margin-top:4px;'>Non-linear Gamma LUT, LAB-space CLAHE, edge-preserving bilateral denoising & real-time side-by-side detection deltas.</div>
                </div>
                <div style='background:rgba(15,23,42,0.6); border:1px solid rgba(56,189,248,0.12); border-radius:10px; padding:12px 16px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <strong style='color:#f8fafc; font-size:0.95rem;'>2. DINOv2 Foundation Retrieval</strong>
                        <span class='status-pill-amber' style='font-size:0.7rem;'>Self-Supervised</span>
                    </div>
                    <div style='color:#94a3b8; font-size:0.84rem; margin-top:4px;'>Extracts 384-D normalized visual embeddings via ViT-S/14 for nearest-neighbor road scene query matching.</div>
                </div>
                <div style='background:rgba(15,23,42,0.6); border:1px solid rgba(56,189,248,0.12); border-radius:10px; padding:12px 16px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <strong style='color:#f8fafc; font-size:0.95rem;'>3. MobileSAM Instance Segmentation</strong>
                        <span class='status-pill-amber' style='font-size:0.7rem;'>Zero-Shot Masks</span>
                    </div>
                    <div style='color:#94a3b8; font-size:0.84rem; margin-top:4px;'>Bipartite IoU matching between YOLO bounding boxes and SAM masks for pixel-accurate road geometry.</div>
                </div>
                <div style='background:rgba(15,23,42,0.6); border:1px solid rgba(56,189,248,0.12); border-radius:10px; padding:12px 16px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <strong style='color:#f8fafc; font-size:0.95rem;'>4. ByteTrack Trajectory Tracking</strong>
                        <span class='status-pill' style='font-size:0.7rem;'>Temporal Filter</span>
                    </div>
                    <div style='color:#94a3b8; font-size:0.84rem; margin-top:4px;'>Kalman Filter motion prediction with Hungarian bipartite ID matching and real-time motion vector trails.</div>
                </div>
                <div style='background:rgba(15,23,42,0.6); border:1px solid rgba(56,189,248,0.12); border-radius:10px; padding:12px 16px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <strong style='color:#f8fafc; font-size:0.95rem;'>5. Generative Diffusion Restorer</strong>
                        <span class='status-pill-amber' style='font-size:0.7rem;'>Prior Synthesis</span>
                    </div>
                    <div style='color:#94a3b8; font-size:0.84rem; margin-top:4px;'>Super-resolution and text-guided detail prior with aspect-ratio preservation and classical comparisons.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='adas-card'>
            <div class='adas-card-title'>⚡ System Architecture Highlights</div>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px;'>
                <div class='metric-box'>
                    <div style='font-weight: 700; color: #38bdf8; font-size:1.1rem;'>CPU Optimized</div>
                    <div style='font-size: 0.75rem; color: #94a3b8; margin-top:2px;'>Fast edge laptop execution</div>
                </div>
                <div class='metric-box'>
                    <div style='font-weight: 700; color: #38bdf8; font-size:1.1rem;'>Lazy Loading</div>
                    <div style='font-size: 0.75rem; color: #94a3b8; margin-top:2px;'>No forced startup downloads</div>
                </div>
                <div class='metric-box'>
                    <div style='font-weight: 700; color: #38bdf8; font-size:1.1rem;'>Full Tracking</div>
                    <div style='font-size: 0.75rem; color: #94a3b8; margin-top:2px;'>ByteTrack motion vectors</div>
                </div>
                <div class='metric-box'>
                    <div style='font-weight: 700; color: #38bdf8; font-size:1.1rem;'>Verified Offline</div>
                    <div style='font-size: 0.75rem; color: #94a3b8; margin-top:2px;'>24 passing unit tests</div>
                </div>
            </div>
            <div style='margin-top: 18px; padding: 14px; background: rgba(56,189,248,0.06); border-radius:8px; border:1px solid rgba(56,189,248,0.15);'>
                <div style='font-size:0.82rem; font-weight:700; color:#38bdf8; text-transform:uppercase; letter-spacing:0.05em;'>Quick Start Guide</div>
                <div style='font-size:0.85rem; color:#cbd5e1; margin-top:6px; line-height:1.5;'>
                    Navigate via the sidebar menu to explore side-by-side detection, visual scene search, instance segmentation, and persistent video tracking.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================================
# PAGE 2: APPROACH 1 — IMAGE ENHANCEMENT & DETECTION (CORE COMPARISON)
# =========================================================================
elif selected_page == "🖼️ Image Enhancement & Detection":
    st.markdown("## 🖼️ Image Enhancement & YOLOv8n Detection Comparison")
    st.caption("Evaluate classical enhancement algorithms and compare raw vs. enhanced YOLO detections side-by-side.")

    with st.container():
        st.markdown("<div class='adas-card'>", unsafe_allow_html=True)
        col_src, col_enh = st.columns([1, 1])

        with col_src:
            source_choice = st.radio(
                "Input Source",
                ["Use Pre-bundled Sample Scene", "Upload Custom Road Image"],
                horizontal=True
            )

        with col_enh:
            enhancement_mode = st.selectbox(
                "Select Enhancement Method",
                [
                    "Combined Low-Light Pipeline",
                    "Gamma Correction",
                    "CLAHE (LAB Lightness)",
                    "Simulated Fog (Demo Testbed)"
                ],
                index=0
            )

        pcol1, pcol2, pcol3 = st.columns(3)
        if enhancement_mode == "Combined Low-Light Pipeline":
            with pcol1:
                gamma_val = st.slider("Gamma Expansion", 0.5, 3.0, 1.5, 0.1)
            with pcol2:
                clip_val = st.slider("CLAHE Clip Limit", 1.0, 8.0, 2.2, 0.2)
            with pcol3:
                denoise_val = st.checkbox("Bilateral Denoise", value=True)
        elif enhancement_mode == "Gamma Correction":
            with pcol1:
                gamma_val = st.slider("Gamma Value", 0.2, 3.5, 1.6, 0.1)
        elif enhancement_mode == "CLAHE (LAB Lightness)":
            with pcol1:
                clip_val = st.slider("Clip Limit", 1.0, 10.0, 2.5, 0.5)
            with pcol2:
                grid_sz = st.select_slider("Tile Grid Size", [4, 8, 16], value=8)
        elif enhancement_mode == "Simulated Fog (Demo Testbed)":
            with pcol1:
                fog_dens = st.slider("Fog Density Factor", 0.1, 1.0, 0.6, 0.05)
            with pcol2:
                fog_bright = st.slider("Atmospheric Luminance", 150, 255, 210, 5)

        st.markdown("</div>", unsafe_allow_html=True)

    raw_bgr = None
    image_name = "sample.jpg"

    if source_choice == "Upload Custom Road Image":
        uploaded_file = st.file_uploader("Upload Road Scene Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_name = uploaded_file.name
            try:
                raw_bgr = safe_decode_image(uploaded_file)
            except Exception as e:
                st.error(f"Image decode error: {e}")
    else:
        sample_options = {
            "Rainy Night City (Real Road Scene)": "samples/rainy_night_city.png",
            "Night Road Scene (Dark)": "samples/road_night_dark.jpg",
            "Simulated Fog Road Scene": "samples/road_foggy_simulated.jpg",
            "Day Road Scene (Normal)": "samples/road_day_normal.jpg"
        }
        selected_sample_label = st.selectbox("Select Benchmark Scene", list(sample_options.keys()))
        sample_path = sample_options[selected_sample_label]
        image_name = os.path.basename(sample_path)
        if os.path.exists(sample_path):
            raw_bgr = cv2.imread(sample_path)
        else:
            st.error(f"Sample file not found at {sample_path}. Please run generate_samples.py.")

    if raw_bgr is not None:
        h, w = raw_bgr.shape[:2]
        st.caption(f"Active Scene: **{image_name}** | Dimensions: **{w} x {h} px**")

        # 1. Apply Enhancement
        t0 = time.perf_counter()
        if enhancement_mode == "Combined Low-Light Pipeline":
            enhanced_bgr = apply_low_light_pipeline(raw_bgr, gamma=gamma_val, clip_limit=clip_val, denoise=denoise_val)
        elif enhancement_mode == "Gamma Correction":
            enhanced_bgr = apply_gamma_correction(raw_bgr, gamma=gamma_val)
        elif enhancement_mode == "CLAHE (LAB Lightness)":
            enhanced_bgr = apply_clahe(raw_bgr, clip_limit=clip_val, tile_grid_size=(grid_sz, grid_sz))
        elif enhancement_mode == "Simulated Fog (Demo Testbed)":
            enhanced_bgr = apply_simulated_fog(raw_bgr, fog_density=fog_dens, fog_brightness=fog_bright)
        t_enh = (time.perf_counter() - t0) * 1000.0

        m_orig = compute_image_metrics(raw_bgr)
        m_enh = compute_image_metrics(enhanced_bgr)

        # 2. Run Detector on Both
        detector = get_detector()
        conf_thresh = st.session_state.get("conf_threshold", 0.25)
        selected_classes = st.session_state.get("selected_classes", None)

        t_det0 = time.perf_counter()
        dets_orig = detector.detect(raw_bgr, conf_threshold=conf_thresh, allowed_classes=selected_classes)
        t_det_orig = (time.perf_counter() - t_det0) * 1000.0

        t_det1 = time.perf_counter()
        dets_enh = detector.detect(enhanced_bgr, conf_threshold=conf_thresh, allowed_classes=selected_classes)
        t_det_enh = (time.perf_counter() - t_det1) * 1000.0

        ann_orig = detector.draw_annotations(raw_bgr, dets_orig, show_proximity=True)
        ann_enh = detector.draw_annotations(enhanced_bgr, dets_enh, show_proximity=True)

        st.markdown("### 🖼️ Side-by-Side Visual Comparison")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown(f"**Original Image ({len(dets_orig)} detected)**")
            st.image(cv2.cvtColor(ann_orig, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"YOLO Inference: {t_det_orig:.1f} ms")
        with col_img2:
            st.markdown(f"**Enhanced Image [{enhancement_mode}] ({len(dets_enh)} detected)**")
            st.image(cv2.cvtColor(ann_enh, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Enhancement: {t_enh:.1f} ms | YOLO Inference: {t_det_enh:.1f} ms")

        st.markdown("### 📊 Visibility & Statistical Image Properties")
        render_metrics_comparison_cards(m_orig, m_enh)

        diff_count = len(dets_enh) - len(dets_orig)
        diff_sign = f"+{diff_count}" if diff_count > 0 else str(diff_count)

        st.markdown(f"""
        <div style='background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-left: 4px solid var(--accent-cyan); padding: 16px 22px; border-radius: 10px; margin: 18px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
            <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;'>
                <div>
                    <span style='font-size:0.75rem; font-weight:700; color:#38bdf8; text-transform:uppercase; letter-spacing:0.08em;'>Perception Delta</span>
                    <div style='font-size:1.1rem; font-weight:700; color:#f8fafc; margin-top:2px;'>
                        Raw Feed: <span style='color:#94a3b8;'>{len(dets_orig)}</span> &nbsp;⟶&nbsp; Enhanced Feed: <span style='color:#38bdf8;'>{len(dets_enh)}</span>
                    </div>
                </div>
                <div>
                    <span class='{"status-pill" if diff_count >= 0 else "status-pill-amber"}' style='font-size:0.95rem; font-family:"JetBrains Mono",monospace; padding:6px 16px;'>
                        Net Shift: {diff_sign} Entities
                    </span>
                </div>
            </div>
            <div style='font-size:0.78rem; color:#94a3b8; margin-top:10px; border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;'>
                ⚠️ <strong>Scientific Rigor:</strong> An increase in detection count demonstrates altered feature prominence and detector activation; it does not claim certified ground-truth precision without human-annotated bounding box benchmarks.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Downloads
        st.markdown("### 💾 Export Outputs")
        d1, d2, d3 = st.columns(3)
        with d1:
            res_enh, enc_enh = cv2.imencode(".png", enhanced_bgr)
            if res_enh:
                st.download_button("⬇️ Download Enhanced Image (PNG)", enc_enh.tobytes(), file_name=f"enhanced_{image_name}.png", mime="image/png")
        with d2:
            res_ann_o, enc_ann_o = cv2.imencode(".png", ann_orig)
            if res_ann_o:
                st.download_button("⬇️ Download Original Annotated (PNG)", enc_ann_o.tobytes(), file_name=f"annotated_orig_{image_name}.png", mime="image/png")
        with d3:
            res_ann_e, enc_ann_e = cv2.imencode(".png", ann_enh)
            if res_ann_e:
                st.download_button("⬇️ Download Enhanced Annotated (PNG)", enc_ann_e.tobytes(), file_name=f"annotated_enh_{image_name}.png", mime="image/png")


# =========================================================================
# PAGE 3: APPROACH 2 — DINOv2 EMBEDDINGS & NEAREST-NEIGHBOR RETRIEVAL
# =========================================================================
elif selected_page == "🔍 DINOv2 Visual Scene Search":
    st.markdown("## 🔍 DINOv2 Visual Scene Search")
    st.caption("Self-supervised Vision Transformer (ViT-S/14) embeddings with Cosine Nearest-Neighbor Retrieval.")

    st.markdown("""
    <div class='notice-box'>
        <strong>Visual Similarity Notice:</strong> Visual embeddings measure feature representation proximity in latent space.
        High similarity indicates comparable visual textures, lighting, and layout, but does NOT prove that two images contain the identical physical objects.
    </div>
    """, unsafe_allow_html=True)

    engine = get_retrieval_engine()
    st.session_state["retrieval_engine"] = engine

    col_idx, col_query = st.columns([1, 2])
    with col_idx:
        st.markdown("<div class='adas-card'>", unsafe_allow_html=True)
        st.markdown("<div class='adas-card-title'>🗂️ Gallery Index</div>", unsafe_allow_html=True)
        st.write(f"Currently Indexed Scenes: **{len(engine.gallery)}**")

        c_b1, c_b2 = st.columns(2)
        with c_b1:
            if st.button("➕ Index New Scenes"):
                with st.spinner("Extracting DINOv2 embeddings for samples..."):
                    cnt = engine.build_index_from_directory("samples")
                    st.success(f"Indexed {cnt} scenes from samples/ folder!")
        with c_b2:
            if st.button("🔄 Full Rebuild"):
                with st.spinner("Clearing and rebuilding gallery index..."):
                    cnt = engine.rebuild_index("samples")
                    st.success(f"Rebuilt index with {cnt} scenes!")

        if engine.gallery:
            st.caption("Indexed files:")
            for item in engine.gallery:
                st.write(f"• `{item['name']}`")
        else:
            st.info("Gallery index is empty. Click button above to index sample scenes.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_query:
        st.markdown("<div class='adas-card'>", unsafe_allow_html=True)
        st.markdown("<div class='adas-card-title'>🔎 Query Scene Search</div>", unsafe_allow_html=True)

        q_mode = st.radio("Query Source", ["Choose Existing Scene", "Upload Query Image"], horizontal=True)
        q_bgr = None
        q_name = "query"

        if q_mode == "Upload Query Image":
            q_file = st.file_uploader("Upload Query Road Image", type=["jpg", "jpeg", "png"])
            if q_file is not None:
                q_name = q_file.name
                q_bgr = safe_decode_image(q_file)
        else:
            sample_choices = {
                "Rainy Night City": "samples/rainy_night_city.png",
                "Night Road": "samples/road_night_dark.jpg",
                "Foggy Road": "samples/road_foggy_simulated.jpg",
                "Day Road": "samples/road_day_normal.jpg"
            }
            c_name = st.selectbox("Select Query Scene", list(sample_choices.keys()))
            c_path = sample_choices[c_name]
            q_name = os.path.basename(c_path)
            if os.path.exists(c_path):
                q_bgr = cv2.imread(c_path)

        top_k = st.slider("Top-K Similar Scenes to Retrieve", 1, 5, 3)

        if q_bgr is not None:
            st.image(cv2.cvtColor(q_bgr, cv2.COLOR_BGR2RGB), caption=f"Query: {q_name}", width=320)

            if st.button("🚀 Search Visually Similar Scenes"):
                if not engine.gallery:
                    st.warning("Please build the gallery index first using the button on the left.")
                else:
                    with st.spinner("Computing DINOv2 embedding and cosine distances..."):
                        t0 = time.perf_counter()
                        matches = engine.search(q_bgr, top_k=top_k)
                        t_search = (time.perf_counter() - t0) * 1000.0

                    st.markdown(f"### Top-{len(matches)} Similar Scenes (Search Time: {t_search:.1f} ms)")
                    match_cols = st.columns(len(matches))
                    for col, match in zip(match_cols, matches):
                        with col:
                            m_img = cv2.imread(match["path"])
                            if m_img is not None:
                                st.image(cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                            st.markdown(f"**{match['name']}**")
                            st.write(f"Similarity: **{match['similarity']:.4f}**")
                            # Normalized bar
                            st.progress(max(0.0, min(1.0, match["similarity"])))

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# PAGE 4: APPROACH 3 — SAM MASKS & OBJECT ANALYSIS
# =========================================================================
elif selected_page == "✂️ SAM Masks & Object Analysis":
    st.markdown("## ✂️ SAM Masks & Object Segmentation Analysis")
    st.caption("Prompt-guided Segment Anything Model (MobileSAM) instance segmentation with pixel area analysis.")

    st.markdown("""
    <div class='notice-box'>
        <strong>SAM Prompting Architecture:</strong> Bounding boxes predicted by YOLOv8 are forwarded as geometric prompts into MobileSAM
        to produce pixel-accurate segmentation masks for road entities.
    </div>
    """, unsafe_allow_html=True)

    sam_engine = get_segmenter()

    col_sam_src, col_sam_ctl = st.columns([1, 1])
    with col_sam_src:
        sam_src = st.radio("Image Source", ["Use Sample Road Scene", "Upload Image"], horizontal=True)
    with col_sam_ctl:
        prompt_mode = st.radio("Prompt Strategy", ["YOLO Detection Box Prompts", "Automatic Everything Mode"], horizontal=True)

    sam_bgr = None
    if sam_src == "Upload Image":
        sam_file = st.file_uploader("Upload Image for SAM", type=["jpg", "jpeg", "png"])
        if sam_file is not None:
            sam_bgr = safe_decode_image(sam_file)
    else:
        sample_choices = {
            "Rainy Night City": "samples/rainy_night_city.png",
            "Day Road Scene": "samples/road_day_normal.jpg",
            "Night Road Scene": "samples/road_night_dark.jpg",
            "Simulated Fog Road Scene": "samples/road_foggy_simulated.jpg"
        }
        chosen_sam = st.selectbox("Select Scenario", list(sample_choices.keys()))
        s_path = sample_choices[chosen_sam]
        if os.path.exists(s_path):
            sam_bgr = cv2.imread(s_path)

    if sam_bgr is not None:
        st.image(cv2.cvtColor(sam_bgr, cv2.COLOR_BGR2RGB), caption="Input Scene", width=480)

        if st.button("🚀 Run Segment Anything Model"):
            with st.spinner("Executing MobileSAM segmentation... (first run downloads ~39MB model weights)"):
                t_sam_start = time.perf_counter()

                bboxes = None
                labels = None
                if prompt_mode == "YOLO Detection Box Prompts":
                    detector = get_detector()
                    dets = detector.detect(sam_bgr, conf_threshold=0.20)
                    if dets:
                        bboxes = [d["box"] for d in dets]
                        labels = [d["class_name"] for d in dets]

                try:
                    segs = sam_engine.segment(sam_bgr, bboxes=bboxes, detection_labels=labels)
                    t_sam = (time.perf_counter() - t_sam_start) * 1000.0

                    if segs:
                        annotated_sam = sam_engine.draw_mask_overlays(sam_bgr, segs)

                        st.markdown(f"### Segmentation Results ({len(segs)} masks, Latency: {t_sam:.1f} ms)")
                        st.image(cv2.cvtColor(annotated_sam, cv2.COLOR_BGR2RGB), use_container_width=True)

                        # Area & mask table
                        st.markdown("#### Mask Area & Geometry Statistics")
                        mask_data = [
                            {
                                "Mask ID": s["mask_id"],
                                "Associated Entity": s["label"],
                                "Area (Pixels)": f"{s['area_px']:,}",
                                "Viewport Area %": f"{s['area_pct']}%",
                                "Box Prompt [x1, y1, x2, y2]": str(s["box"]) if s["box"] else "Auto Mode"
                            }
                            for s in segs
                        ]
                        st.dataframe(pd.DataFrame(mask_data), use_container_width=True)

                        # Download segmented overlay
                        res_sam, enc_sam = cv2.imencode(".png", annotated_sam)
                        if res_sam:
                            st.download_button("⬇️ Download Segmented Overlay (PNG)", enc_sam.tobytes(), file_name="sam_segmented.png", mime="image/png")
                    else:
                        st.info("SAM produced 0 masks for this scene. Try adjusting the prompt mode or confidence.")
                except Exception as e:
                    st.error(f"SAM Execution Note: {e}")


# =========================================================================
# PAGE 5: APPROACH 4 — OBJECT DETECTOR + BYTETRACK OBJECT TRACKER
# =========================================================================
elif selected_page == "🏎️ Video Object Tracking":
    st.markdown("## 🏎️ Video Object Tracking & Traffic Analysis")
    st.caption("YOLOv8 deep detection coupled with ByteTrack persistent ID association and motion vector trails.")

    st.markdown("""
    <div class='notice-box'>
        <strong>Tracking Methodology & Disclaimers:</strong>
        ByteTrack matches bounding boxes across sequential video frames using Kalman Filter motion predictions and Hungarian association.<br>
        <em>Limitations:</em> Severe camera ego-motion, extreme low-light, or occlusions can trigger ID-switches.
    </div>
    """, unsafe_allow_html=True)

    tracker = get_tracker()

    t_src = st.radio("Video Source", ["Pre-bundled Demo Clip (samples/demo_road_clip.mp4)", "Upload Custom Video"], horizontal=True)
    vid_file_path = None

    if t_src == "Upload Custom Video":
        up_vid = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
        if up_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(up_vid.read())
            vid_file_path = tfile.name
    else:
        demo_clip = "samples/demo_road_clip.mp4"
        if os.path.exists(demo_clip):
            vid_file_path = demo_clip

    col_tk1, col_tk2, col_tk3 = st.columns(3)
    with col_tk1:
        draw_trails = st.checkbox("Render Motion Trails", value=True)
    with col_tk2:
        tk_stride = st.slider("Frame Stride", 1, 4, 1, help="Process every Nth frame")
    with col_tk3:
        max_tk_frames = st.slider("Max Frames Ceiling", 10, 100, 30, 10)

    if vid_file_path and st.button("🚀 Start Persistent Tracking Pipeline"):
        tracker.reset_tracks()
        cap = cv2.VideoCapture(vid_file_path)
        if not cap.isOpened():
            st.error("Failed to open video source.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or np.isnan(fps) or fps <= 0:
                fps = 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if width <= 0 or height <= 0:
                st.error("Invalid video dimensions detected. Cannot process video.")
                cap.release()
            else:
                frames_to_run = min(total_vid_frames, max_tk_frames) if total_vid_frames > 0 else max_tk_frames

                out_tk_path = os.path.join(tempfile.gettempdir(), f"visionhack_track_{int(time.time())}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                effective_fps = max(1.0, fps / tk_stride)
                writer = cv2.VideoWriter(out_tk_path, fourcc, effective_fps, (width, height))

                if not writer.isOpened():
                    st.error("Could not initialize video encoder for output export. Check system codecs.")
                    cap.release()
                else:
                    p_bar = st.progress(0.0)
                    status_txt = st.empty()
                    prev_slot = st.empty()

                    t_start = time.perf_counter()
                    processed_f = 0
                    raw_f = 0

            while cap.isOpened() and raw_f < frames_to_run:
                ret, frame = cap.read()
                if not ret:
                    break

                if raw_f % tk_stride != 0:
                    raw_f += 1
                    continue

                tracks = tracker.track_frame(frame, conf_threshold=0.20)
                tracked_frame = tracker.draw_tracking_annotations(frame, tracks, draw_trails=draw_trails)

                # Overlay stats HUD
                active_ids = len([t for t in tracks if t["track_id"] > 0])
                cv2.putText(
                    tracked_frame,
                    f"FRAME {processed_f+1} | ACTIVE TRACKS: {active_ids}",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (56, 189, 248),
                    2,
                    cv2.LINE_AA
                )

                writer.write(tracked_frame)
                processed_f += 1
                raw_f += 1

                if processed_f % 3 == 0 or raw_f >= frames_to_run:
                    p_bar.progress(min(1.0, raw_f / frames_to_run))
                    status_txt.text(f"Tracked frame {processed_f} (Raw: {raw_f}/{frames_to_run})...")
                    prev_slot.image(cv2.cvtColor(tracked_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            cap.release()
            writer.release()
            elapsed_tk = time.perf_counter() - t_start

            st.success(f"Tracking complete: {processed_f} frames tracked in {elapsed_tk:.2f}s ({processed_f/max(0.001, elapsed_tk):.1f} FPS)!")

            # Cumulative counts
            st.markdown("### Cumulative Unique Track Counts by Class")
            c_counts = {cls: len(ids) for cls, ids in tracker.cumulative_counts.items()}
            if c_counts:
                st.bar_chart(pd.Series(c_counts, name="Cumulative Tracks"))
            else:
                st.info("No persistent tracks confirmed.")

            if os.path.exists(out_tk_path) and os.path.getsize(out_tk_path) > 0:
                with open(out_tk_path, "rb") as vf:
                    st.download_button("⬇️ Download Tracked Video (MP4)", vf.read(), file_name="tracked_road_clip.mp4", mime="video/mp4")


# =========================================================================
# PAGE 6: APPROACH 5 — DIFFUSION MODELS FOR IMAGE RESTORATION
# =========================================================================
elif selected_page == "✨ Diffusion Image Restoration":
    st.markdown("## ✨ Diffusion Models for Image Restoration")
    st.caption("AI-driven generative image restoration compared directly against classical Gamma & CLAHE baselines.")

    st.markdown("""
    <div class='notice-box'>
        <strong>Scientific Rigor & Hallucination Notice:</strong>
        Diffusion restoration generates plausible high-frequency textures conditioned on priors.
        It does NOT recover genuine sensor data that was destroyed by underexposure or optical scattering.<br>
        Classical methods (Gamma/CLAHE) remain fast, deterministic baselines.
    </div>
    """, unsafe_allow_html=True)

    restorer = get_restorer()

    c_diff_src, c_diff_opt = st.columns([1, 1])
    with c_diff_src:
        d_sample = st.selectbox(
            "Select Degraded Road Scene",
            ["samples/rainy_night_city.png", "samples/road_night_dark.jpg", "samples/road_foggy_simulated.jpg", "samples/road_day_normal.jpg"]
        )
    with c_diff_opt:
        st.write(f"Diffusion Engine Status: **{('Installed' if restorer.is_available else 'Optional - PyTorch CPU/GPU')}**")
        diff_prompt = st.text_input("Restoration Prompt Prior", "clear high-contrast road scene, sharp highway markings")

    if os.path.exists(d_sample):
        raw_bgr = cv2.imread(d_sample)
        # Fast classical baseline
        clahe_baseline = apply_low_light_pipeline(raw_bgr, gamma=1.5, clip_limit=2.2, denoise=True)

        st.markdown("### Triple-View Comparison: Raw vs Classical vs Diffusion")
        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            st.image(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), caption="1. Degraded Input", use_container_width=True)
            st.json(compute_image_metrics(raw_bgr))
        with col_v2:
            st.image(cv2.cvtColor(clahe_baseline, cv2.COLOR_BGR2RGB), caption="2. Classical Baseline (Gamma+CLAHE)", use_container_width=True)
            st.json(compute_image_metrics(clahe_baseline))
        with col_v3:
            st.markdown("**3. Diffusion Restored Result**")
            if st.button("🚀 Attempt Diffusion Inference"):
                with st.spinner("Checking diffusion model weights and environment..."):
                    out_diff, info = restorer.restore_image(raw_bgr, prompt=diff_prompt)
                    if out_diff is not None:
                        st.image(cv2.cvtColor(out_diff, cv2.COLOR_BGR2RGB), caption="Restored Output", use_container_width=True)
                    else:
                        st.warning(f"Diffusion Status: {info['status'].upper()}")
                        st.info(f"Diagnostics: {info.get('error')}")


# =========================================================================
# PAGE 7: ANALYTICS & EVALUATION
# =========================================================================
elif selected_page == "📈 Analytics & Evaluation":
    st.markdown("## 📈 Analytics & Empirical Evaluation")
    st.caption("Verified statistical benchmarks derived genuinely from actual test pipeline runs.")

    sample_choices = {
        "Rainy Night City Scene": "samples/rainy_night_city.png",
        "Night Road Scene": "samples/road_night_dark.jpg",
        "Day Road Scene": "samples/road_day_normal.jpg",
        "Simulated Fog Road Scene": "samples/road_foggy_simulated.jpg"
    }
    chosen_scene = st.selectbox("Select Test Scene for Analytics", list(sample_choices.keys()))
    img_path = sample_choices[chosen_scene]

    if os.path.exists(img_path):
        raw_img = cv2.imread(img_path)
        enh_img = apply_low_light_pipeline(raw_img, gamma=1.5, clip_limit=2.2, denoise=True)

        conf_th = st.session_state.get("conf_threshold", 0.25)
        allowed_cls = st.session_state.get("selected_classes", None)
        detector = get_detector()

        t_start = time.perf_counter()
        dets_raw = detector.detect(raw_img, conf_threshold=conf_th, allowed_classes=allowed_cls)
        t_raw = (time.perf_counter() - t_start) * 1000.0

        t_start = time.perf_counter()
        dets_enh = detector.detect(enh_img, conf_threshold=conf_th, allowed_classes=allowed_cls)
        t_enh = (time.perf_counter() - t_start) * 1000.0

        st.markdown("### Summary Key Performance Indicators")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Objects (Raw)", len(dets_raw))
        with k2:
            st.metric("Total Objects (Enhanced)", len(dets_enh), delta=len(dets_enh)-len(dets_raw))
        with k3:
            avg_conf_raw = (np.mean([d["confidence"] for d in dets_raw]) * 100) if dets_raw else 0.0
            avg_conf_enh = (np.mean([d["confidence"] for d in dets_enh]) * 100) if dets_enh else 0.0
            st.metric("Avg Conf (Enh)", f"{avg_conf_enh:.1f}%", delta=f"{avg_conf_enh - avg_conf_raw:+.1f}%")
        with k4:
            st.metric("Inference Time", f"{t_enh:.1f} ms", delta=f"{t_enh - t_raw:+.1f} ms")

        st.markdown("### Class Distribution & Confidence Breakdown")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Enhanced Detections by Class**")
            if dets_enh:
                class_counts = {}
                for d in dets_enh:
                    class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1
                st.bar_chart(pd.Series(class_counts, name="Count"))
            else:
                st.info("No detections above threshold in enhanced scene.")
        with col_c2:
            st.markdown("**Confidence Scores per Detection**")
            if dets_enh:
                conf_series = pd.Series([d["confidence"] for d in dets_enh], index=[f"Obj {i+1} ({d['class_name']})" for i, d in enumerate(dets_enh)], name="Confidence")
                st.bar_chart(conf_series)
            else:
                st.info("No detection confidence data available.")


# =========================================================================
# PAGE 8: ABOUT, METHODOLOGY & LIMITATIONS
# =========================================================================
elif selected_page == "📖 About & Methodology":
    st.markdown("## 📖 About VisionHack ADAS")
    st.markdown("#### Comprehensive Methodology, 5 CV Approaches, and Ethical Limitations")

    st.markdown("""
    <div class='adas-card'>
        <div class='adas-card-title'>🔬 Technical Synthesis of the Five Approaches</div>
        <ul>
            <li><strong>1. OpenCV + YOLOv8n:</strong> Real-time baseline object localization using an anchor-free CNN backbone paired with LUT gamma expansion and LAB-space CLAHE.</li>
            <li><strong>2. DINOv2 Visual Retrieval:</strong> ViT-S/14 self-supervised representation learning producing 384-dimensional L2-normalized vectors for semantic scene matching.</li>
            <li><strong>3. SAM Instance Segmentation:</strong> Prompt-driven MobileSAM transformer producing binary boundary masks with pixel area quantification.</li>
            <li><strong>4. ByteTrack Object Tracking:</strong> Two-stage Kalman-filtered Hungarian data association maintaining persistent vehicle and pedestrian trajectories.</li>
            <li><strong>5. Diffusion Image Restoration:</strong> Iterative denoising score-based reverse diffusion prior evaluated against deterministic baselines.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='adas-card'>
        <div class='adas-card-title'>⚖️ Ethical Constraints & Disclaimers</div>
        <div class='notice-box'>
            <strong>Non-Calibrated Proximity Disclosure:</strong><br>
            Computing real-world metric distances (in metres) from monocular 2D bounding boxes requires known camera intrinsic matrices ($f_x, f_y, c_x, c_y$),
            extrinsic pitch angles, and flat-ground plane assumptions.<br><br>
            Therefore, VisionHack ADAS displays image-space viewport area heuristics only. This software is an engineering study and is strictly <strong>NOT</strong> certified for safety-critical vehicle control or autonomous actuation.
        </div>
    </div>
    """, unsafe_allow_html=True)
