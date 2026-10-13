"""VisionHack ADAS — Streamlit Web Application

AI-Powered Visibility Enhancement and Road-Scene Analysis Dashboard.
"""

import os
import time
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from vision.enhancement import (
    validate_image,
    apply_gamma_correction,
    apply_clahe,
    apply_low_light_pipeline,
    apply_simulated_fog
)
from vision.detection import ObjectDetector, DEFAULT_ROAD_CLASSES, CLASS_COLORS
from vision.metrics import compute_image_metrics


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
/* Base Dark Theme (Navy / Slate / Teal) */
.stApp {
    background: linear-gradient(180deg, #0b1120 0%, #070d19 100%);
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}

/* Glassmorphism Cards */
.adas-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(10px);
}

.adas-card-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #38bdf8;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Metric Badges */
.metric-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.metric-val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}
.metric-delta-pos {
    color: #34d399;
    font-size: 0.8rem;
    font-weight: 600;
}
.metric-delta-neg {
    color: #f87171;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Notice Banner */
.notice-box {
    background: rgba(245, 158, 11, 0.1);
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 4px;
    color: #fde68a;
    font-size: 0.9rem;
    margin-bottom: 16px;
}

/* Info Pill */
.status-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

/* Buttons */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #0284c7 0%, #0f766e 100%);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
    transition: all 0.2s ease-in-out;
}
div.stButton > button:first-child:hover {
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    transform: translateY(-1px);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# Caching Model Initialization
# ==========================================
@st.cache_resource(show_spinner="Initializing YOLOv8n object detector...")
def get_detector() -> ObjectDetector:
    return ObjectDetector(model_name="yolov8n.pt")


detector = get_detector()


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
    st.title("🚗 VisionHack ADAS")
    st.caption("AI-Powered Visibility Enhancement & Road Scene Analysis")
    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "📊 Dashboard Overview",
            "⚖️ Original vs Enhanced Detection",
            "🖼️ Image Enhancement",
            "📈 Detection Analytics",
            "🎥 Video Analysis",
            "⚙️ Detection Settings",
            "📖 About & Methodology"
        ],
        index=1
    )

    st.divider()
    st.markdown("### Model Status")
    if detector.is_ready:
        st.markdown("<span class='status-pill'>● YOLOv8n Active (CPU)</span>", unsafe_allow_html=True)
        st.caption("Ultralytics Pretrained Engine Ready")
    else:
        st.error(f"Detector Offline: {detector.error_message}")

    st.divider()
    st.markdown("""
    <div style='font-size: 0.75rem; color: #64748b;'>
    <strong>ADAS Prototype Disclaimer:</strong><br>
    All visibility scores and proximity tags are experimental heuristics.
    Not calibrated to real-world metric distances. Not certified for vehicle control.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 1: DASHBOARD OVERVIEW
# ==========================================
if selected_page == "📊 Dashboard Overview":
    st.markdown("## 📊 VisionHack ADAS Overview")
    st.markdown("#### Advanced Driver Assistance System Visibility & Perception Pipeline")

    st.markdown("""
    <div class='notice-box'>
        ⚠️ <strong>Notice:</strong> This dashboard is an engineering and research prototype for adverse weather and low-light road analysis.
        All contrast scores and spatial heuristics are experimental and not calibrated to real-world metric distances.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class='adas-card'>
            <div class='adas-card-title'>🎯 System Objectives</div>
            <p>
            Autonomous vehicles and Advanced Driver Assistance Systems (ADAS) experience severe perception degradation under low-light, nighttime, and adverse atmospheric conditions (fog/rain).
            VisionHack demonstrates a modular, dual-stage vision architecture:
            </p>
            <ul>
                <li><strong>Stage 1 — Adaptive Pre-processing:</strong> Non-linear gamma expansion and LAB-space CLAHE to recover edge contrast and shadow features without washing out highlights.</li>
                <li><strong>Stage 2 — Perception & Object Detection:</strong> Real-time YOLOv8n inference to localize road entities (vehicles, pedestrians, cyclists, signs).</li>
                <li><strong>Stage 3 — Quality & Proximity Heuristics:</strong> Laplacian edge metrics and image-space bounding box sizing heuristics to guide driver awareness.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='adas-card'>
            <div class='adas-card-title'>⚡ Key Capabilities</div>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px;'>
                <div class='metric-box'>
                    <div style='font-weight: 600; color: #38bdf8;'>Gamma Correction</div>
                    <div style='font-size: 0.8rem; color: #94a3b8; margin-top: 4px;'>Lookup-table accelerated luminance redistribution</div>
                </div>
                <div class='metric-box'>
                    <div style='font-weight: 600; color: #38bdf8;'>LAB CLAHE</div>
                    <div style='font-size: 0.8rem; color: #94a3b8; margin-top: 4px;'>Lightness-confined contrast equalization preserving color fidelity</div>
                </div>
                <div class='metric-box'>
                    <div style='font-weight: 600; color: #38bdf8;'>Real-Time YOLOv8</div>
                    <div style='font-size: 0.8rem; color: #94a3b8; margin-top: 4px;'>Single-stage deep neural detection of road actors</div>
                </div>
                <div class='metric-box'>
                    <div style='font-weight: 600; color: #38bdf8;'>Statistical Metrics</div>
                    <div style='font-size: 0.8rem; color: #94a3b8; margin-top: 4px;'>RMS contrast, dark ratio, and Laplacian sharpness variance</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='adas-card'>
            <div class='adas-card-title'>🚀 Quick Navigation</div>
            <ol style='color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;'>
                <li><strong>⚖️ Original vs Enhanced Detection:</strong> Highest-priority comparison workflow running YOLOv8n on both original and enhanced inputs.</li>
                <li><strong>🖼️ Image Enhancement:</strong> Dedicated tuning for Gamma, CLAHE, and Bilateral low-light filters.</li>
                <li><strong>📈 Detection Analytics:</strong> Class distributions, confidence breakdowns, and comparative metrics.</li>
                <li><strong>🎥 Video Analysis:</strong> Frame-by-frame video processing with configurable sampling intervals.</li>
            </ol>
            <div style='text-align: center; margin-top: 15px;'>
                <span class='status-pill'>Architecture Verified & Testable</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='adas-card'>
            <div class='adas-card-title'>📁 Sample Assets Ready</div>
            <div style='font-size: 0.85rem; color: #94a3b8;'>
                Pre-generated test scenes in <code>samples/</code>:
                <ul style='margin-top: 6px;'>
                    <li><code>road_day_normal.jpg</code> (Clear day benchmark)</li>
                    <li><code>road_night_dark.jpg</code> (Low-light underexposure)</li>
                    <li><code>road_foggy_simulated.jpg</code> (Atmospheric scattering demo)</li>
                    <li><code>demo_road_clip.mp4</code> (30-frame road clip)</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PAGE 2: FEATURE 1 — ORIGINAL VS ENHANCED DETECTION (HIGH PRIORITY)
# =========================================================
elif selected_page == "⚖️ Original vs Enhanced Detection":
    st.markdown("## ⚖️ Original vs Enhanced Detection Comparison")
    st.caption("Side-by-side detection evaluation using the identical YOLOv8n model on raw and enhanced inputs.")

    # Top Controls Card
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

        # Method parameters
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

    # Image loading
    raw_bgr = None
    image_name = "sample.jpg"

    if source_choice == "Upload Custom Road Image":
        uploaded_file = st.file_uploader(
            "Upload Road Scene Image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image_name = uploaded_file.name
            try:
                raw_bgr = safe_decode_image(uploaded_file)
            except Exception as e:
                st.error(f"Image decode error: {e}")
    else:
        sample_options = {
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

        # 2. Quality Metrics
        m_orig = compute_image_metrics(raw_bgr)
        m_enh = compute_image_metrics(enhanced_bgr)

        # 3. Model Inference on Both Images
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

        # 4. Display Side-by-Side Images
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

        # 5. Metric Cards Comparison
        st.markdown("### 📊 Visibility & Statistical Image Properties")
        render_metrics_comparison_cards(m_orig, m_enh)

        # 6. Detection Difference & Scientific Disclaimer
        diff_count = len(dets_enh) - len(dets_orig)
        diff_sign = f"+{diff_count}" if diff_count > 0 else str(diff_count)

        st.markdown(f"""
        <div style='background: #1e293b; border-left: 4px solid #38bdf8; padding: 12px 18px; border-radius: 6px; margin: 15px 0;'>
            <strong>Detection Count Shift:</strong>
            Original: <strong style='color:#cbd5e1;'>{len(dets_orig)}</strong> |
            Enhanced: <strong style='color:#38bdf8;'>{len(dets_enh)}</strong> |
            Difference: <strong style='color:{"#34d399" if diff_count >= 0 else "#f87171"};'>{diff_sign} objects</strong>
            <br>
            <span style='font-size:0.8rem; color:#94a3b8;'>
            ⚠️ Scientific Note: A change in detection count does not claim or prove improved accuracy without verified ground-truth labels.
            </span>
        </div>
        """, unsafe_allow_html=True)

        # 7. Comparison Table
        st.markdown("### 📋 Detailed Detection Comparison Table")
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.markdown("**Original Image Detections**")
            if dets_orig:
                df_orig = pd.DataFrame([
                    {
                        "Class": d["class_name"],
                        "Confidence": f"{d['confidence']:.1%}",
                        "Box [x1, y1, x2, y2]": str(d["box"]),
                        "Area %": f"{d['relative_area_pct']}%",
                        "Proximity": d["proximity_heuristic"].split()[0]
                    }
                    for d in dets_orig
                ])
                st.dataframe(df_orig, use_container_width=True)
            else:
                st.info("No objects detected in original image.")

        with tcol2:
            st.markdown("**Enhanced Image Detections**")
            if dets_enh:
                df_enh = pd.DataFrame([
                    {
                        "Class": d["class_name"],
                        "Confidence": f"{d['confidence']:.1%}",
                        "Box [x1, y1, x2, y2]": str(d["box"]),
                        "Area %": f"{d['relative_area_pct']}%",
                        "Proximity": d["proximity_heuristic"].split()[0]
                    }
                    for d in dets_enh
                ])
                st.dataframe(df_enh, use_container_width=True)
            else:
                st.info("No objects detected in enhanced image.")

        # 8. Export Downloads
        st.markdown("### 💾 Export Outputs")
        d1, d2, d3 = st.columns(3)
        with d1:
            res_enh, enc_enh = cv2.imencode(".png", enhanced_bgr)
            if res_enh:
                st.download_button(
                    "⬇️ Download Enhanced Image (PNG)",
                    enc_enh.tobytes(),
                    file_name=f"enhanced_{image_name}.png",
                    mime="image/png"
                )
        with d2:
            res_ann_o, enc_ann_o = cv2.imencode(".png", ann_orig)
            if res_ann_o:
                st.download_button(
                    "⬇️ Download Original Annotated (PNG)",
                    enc_ann_o.tobytes(),
                    file_name=f"annotated_orig_{image_name}.png",
                    mime="image/png"
                )
        with d3:
            res_ann_e, enc_ann_e = cv2.imencode(".png", ann_enh)
            if res_ann_e:
                st.download_button(
                    "⬇️ Download Enhanced Annotated (PNG)",
                    enc_ann_e.tobytes(),
                    file_name=f"annotated_enh_{image_name}.png",
                    mime="image/png"
                )


# =========================================================
# PAGE 3: FEATURE 2 — IMAGE ENHANCEMENT & QUALITY ANALYSIS
# =========================================================
elif selected_page == "🖼️ Image Enhancement":
    st.markdown("## 🖼️ Image Enhancement & Quality Analysis")
    st.caption("Deep-dive inspection into non-linear gamma, LAB-space CLAHE, and image-quality indicators.")

    src_mode = st.radio("Source", ["Pre-bundled Benchmark Scenario", "Upload Image"], horizontal=True)
    raw_bgr = None
    image_name = "test.jpg"

    if src_mode == "Upload Image":
        up_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if up_file is not None:
            image_name = up_file.name
            try:
                raw_bgr = safe_decode_image(up_file)
            except Exception as e:
                st.error(f"Image load error: {e}")
    else:
        sample_options = {
            "Low-Light Night Road": "samples/road_night_dark.jpg",
            "Atmospheric Fog Scenario": "samples/road_foggy_simulated.jpg",
            "Clear Day Benchmark": "samples/road_day_normal.jpg"
        }
        chosen = st.selectbox("Scenario", list(sample_options.keys()))
        s_path = sample_options[chosen]
        image_name = os.path.basename(s_path)
        if os.path.exists(s_path):
            raw_bgr = cv2.imread(s_path)

    if raw_bgr is not None:
        c_mode, c_p1, c_p2 = st.columns([2, 1, 1])
        with c_mode:
            enh_algo = st.selectbox(
                "Enhancement Pipeline",
                ["Combined Low-Light", "Gamma Correction", "CLAHE (LAB)", "Simulated Fog"]
            )
        if enh_algo == "Combined Low-Light":
            with c_p1:
                g_val = st.slider("Gamma", 0.5, 3.0, 1.5, 0.1)
            with c_p2:
                c_val = st.slider("CLAHE Clip", 1.0, 8.0, 2.2, 0.2)
            enhanced = apply_low_light_pipeline(raw_bgr, gamma=g_val, clip_limit=c_val, denoise=True)
        elif enh_algo == "Gamma Correction":
            with c_p1:
                g_val = st.slider("Gamma", 0.2, 3.5, 1.6, 0.1)
            enhanced = apply_gamma_correction(raw_bgr, gamma=g_val)
        elif enh_algo == "CLAHE (LAB)":
            with c_p1:
                c_val = st.slider("Clip Limit", 1.0, 10.0, 2.5, 0.5)
            enhanced = apply_clahe(raw_bgr, clip_limit=c_val)
        else:
            with c_p1:
                f_dens = st.slider("Density", 0.1, 1.0, 0.6, 0.05)
            enhanced = apply_simulated_fog(raw_bgr, fog_density=f_dens)

        m_raw = compute_image_metrics(raw_bgr)
        m_proc = compute_image_metrics(enhanced)

        st.markdown("### Visual Comparison")
        v1, v2 = st.columns(2)
        with v1:
            st.image(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), caption="Original Input", use_container_width=True)
        with v2:
            st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), caption=f"Enhanced Output ({enh_algo})", use_container_width=True)

        st.markdown("### Quality Metrics Breakdown")
        render_metrics_comparison_cards(m_raw, m_proc)

        # Comparison Bar Chart
        st.markdown("#### Metrics Comparison Chart")
        chart_df = pd.DataFrame({
            "Metric": ["Mean Luminance", "RMS Contrast", "Laplacian Var", "Visibility Index"],
            "Original": [m_raw["mean_luminance"], m_raw["rms_contrast"], min(250.0, m_raw["laplacian_variance"]), m_raw["experimental_quality_score"]],
            "Enhanced": [m_proc["mean_luminance"], m_proc["rms_contrast"], min(250.0, m_proc["laplacian_variance"]), m_proc["experimental_quality_score"]],
        }).set_index("Metric")
        st.bar_chart(chart_df)


# =========================================================
# PAGE 4: FEATURE 3 — DETECTION ANALYTICS DASHBOARD
# =========================================================
elif selected_page == "📈 Detection Analytics":
    st.markdown("## 📈 Detection Analytics Dashboard")
    st.caption("Quantitative analytics derived genuinely from YOLOv8n object detection runs.")

    sample_choices = {
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

        t_start = time.perf_counter()
        dets_raw = detector.detect(raw_img, conf_threshold=conf_th, allowed_classes=allowed_cls)
        t_raw = (time.perf_counter() - t_start) * 1000.0

        t_start = time.perf_counter()
        dets_enh = detector.detect(enh_img, conf_threshold=conf_th, allowed_classes=allowed_cls)
        t_enh = (time.perf_counter() - t_start) * 1000.0

        # Summary KPIs
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

        # Class Breakdown
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

        # Proximity heuristics breakdown
        st.markdown("### Image-Space Proximity Heuristics (Visual Area %)")
        if dets_enh:
            prox_df = pd.DataFrame([
                {
                    "Object": f"{d['class_name']} #{i+1}",
                    "Viewport Area %": d["relative_area_pct"],
                    "Proximity Tag": d["proximity_heuristic"]
                }
                for i, d in enumerate(dets_enh)
            ])
            st.dataframe(prox_df, use_container_width=True)
        else:
            st.info("No objects available for proximity analysis.")


# =========================================================
# PAGE 5: FEATURE 4 — VIDEO ANALYSIS IMPROVEMENTS
# =========================================================
elif selected_page == "🎥 Video Analysis":
    st.markdown("## 🎥 Video Stream & Batch Analysis")
    st.caption("Frame-by-frame pipeline processing with configurable sampling interval, frame ceilings, and export.")

    st.markdown("""
    <div class='notice-box'>
        ⚙️ <strong>Frame-Sampling Interval & Frame Ceilings:</strong> To prevent memory overload and long processing times on laptops,
        configure the sampling stride and maximum frame ceiling. Output video is encoded frame-by-frame using OpenCV.
    </div>
    """, unsafe_allow_html=True)

    v_src = st.radio("Video Source", ["Use Pre-bundled Demo Clip", "Upload Video File"], horizontal=True)

    video_path_to_process = None
    if v_src == "Use Pre-bundled Demo Clip":
        demo_path = "samples/demo_road_clip.mp4"
        if os.path.exists(demo_path):
            video_path_to_process = demo_path
            st.info(f"Loaded demonstration clip: `{demo_path}`")
        else:
            st.warning("Demo video not found. Run generate_samples.py to build it.")
    else:
        uploaded_vid = st.file_uploader("Upload Video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
        if uploaded_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())
            video_path_to_process = tfile.name

    # Video Controls
    v_col1, v_col2, v_col3, v_col4 = st.columns(4)
    with v_col1:
        v_enhancement = st.selectbox(
            "Enhancement Mode",
            ["Combined Low-Light", "Gamma Correction", "CLAHE", "None (Raw)"]
        )
    with v_col2:
        frame_sampling_stride = st.slider("Sampling Stride (Every Nth Frame)", min_value=1, max_value=5, value=1, help="1 = process every frame, 2 = process every 2nd frame.")
    with v_col3:
        max_frames_to_process = st.slider("Max Frames Ceiling", min_value=10, max_value=150, value=30, step=10)
    with v_col4:
        run_v_detection = st.checkbox("Run YOLO Detection", value=True)

    if video_path_to_process and st.button("🚀 Start Frame-by-Frame Pipeline"):
        cap = cv2.VideoCapture(video_path_to_process)
        if not cap.isOpened():
            st.error("Failed to open video file. Ensure valid codec (H.264 / MP4V).")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_limit = min(total_frames, max_frames_to_process) if total_frames > 0 else max_frames_to_process

            st.write(f"Processing up to **{frames_limit}** frames ({width}x{height} @ {fps:.1f} FPS, Stride: {frame_sampling_stride})...")

            # Temporary output video file
            out_temp_path = os.path.join(tempfile.gettempdir(), f"visionhack_out_{int(time.time())}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_temp_path, fourcc, fps / frame_sampling_stride, (width, height))

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            preview_slot = st.empty()

            processed_count = 0
            raw_frame_idx = 0
            t_start = time.perf_counter()
            total_dets_found = 0

            conf_thresh = st.session_state.get("conf_threshold", 0.25)
            selected_classes = st.session_state.get("selected_classes", None)

            while cap.isOpened() and raw_frame_idx < frames_limit:
                ret, frame = cap.read()
                if not ret:
                    break

                # Frame-sampling stride
                if raw_frame_idx % frame_sampling_stride != 0:
                    raw_frame_idx += 1
                    continue

                # Apply enhancement
                if v_enhancement == "Gamma Correction":
                    proc_frame = apply_gamma_correction(frame, gamma=1.5)
                elif v_enhancement == "CLAHE":
                    proc_frame = apply_clahe(frame, clip_limit=2.0)
                elif v_enhancement == "Combined Low-Light":
                    proc_frame = apply_low_light_pipeline(frame, gamma=1.4, clip_limit=2.0, denoise=False)
                else:
                    proc_frame = frame.copy()

                # Detection
                frame_dets = []
                if run_v_detection and detector.is_ready:
                    frame_dets = detector.detect(proc_frame, conf_threshold=conf_thresh, allowed_classes=selected_classes)
                    total_dets_found += len(frame_dets)
                    final_frame = detector.draw_annotations(proc_frame, frame_dets, show_proximity=True)
                else:
                    final_frame = proc_frame

                # Overlay frame counter
                cv2.putText(
                    final_frame,
                    f"FRAME {processed_count+1} (Dets: {len(frame_dets)})",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (56, 189, 248),
                    2,
                    cv2.LINE_AA
                )

                writer.write(final_frame)
                processed_count += 1
                raw_frame_idx += 1

                # Live preview
                if processed_count % 3 == 0 or raw_frame_idx >= frames_limit:
                    progress_bar.progress(min(1.0, raw_frame_idx / frames_limit))
                    status_text.text(f"Processed {processed_count} sampled frames (Raw frame: {raw_frame_idx}/{frames_limit})...")
                    preview_slot.image(
                        cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Live Preview (Sampled Frame {processed_count})",
                        use_container_width=True
                    )

            cap.release()
            writer.release()
            elapsed = time.perf_counter() - t_start

            st.success(f"Processing complete! Exported {processed_count} frames in {elapsed:.2f}s ({processed_count / max(0.001, elapsed):.1f} FPS, {total_dets_found} cumulative detections).")

            # Video Download
            if os.path.exists(out_temp_path) and os.path.getsize(out_temp_path) > 0:
                with open(out_temp_path, "rb") as vf:
                    st.download_button(
                        label="⬇️ Download Annotated Video (MP4)",
                        data=vf.read(),
                        file_name="visionhack_adas_annotated.mp4",
                        mime="video/mp4"
                    )


# ==========================================
# PAGE 6: DETECTION SETTINGS
# ==========================================
elif selected_page == "⚙️ Detection Settings":
    st.markdown("## ⚙️ ADAS Perception & Detection Settings")
    st.caption("Tune YOLOv8 confidence thresholds, target road classes, and inference parameters.")

    st.markdown("<div class='adas-card'>", unsafe_allow_html=True)
    st.markdown("<div class='adas-card-title'>🎯 Confidence & Filtering</div>", unsafe_allow_html=True)

    current_conf = st.slider(
        "Detection Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=st.session_state.get("conf_threshold", 0.25),
        step=0.05,
        help="Higher values reduce false positives in noisy night frames; lower values detect subtle distant objects."
    )
    st.session_state["conf_threshold"] = current_conf

    all_classes = detector.get_available_classes()
    road_class_ids = list(DEFAULT_ROAD_CLASSES.keys())
    road_class_names = [all_classes.get(cid, f"Class {cid}") for cid in road_class_ids]

    st.markdown("<div class='adas-card-title' style='margin-top:20px;'>🚗 Active Road Classes</div>", unsafe_allow_html=True)
    selected_names = st.multiselect(
        "Filter target classes for ADAS detection:",
        options=road_class_names,
        default=road_class_names
    )

    name_to_id = {all_classes.get(cid): cid for cid in road_class_ids}
    st.session_state["selected_classes"] = {name_to_id[n] for n in selected_names if n in name_to_id}

    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 7: ABOUT & METHODOLOGY
# ==========================================
elif selected_page == "📖 About & Methodology":
    st.markdown("## 📖 About VisionHack ADAS")
    st.markdown("#### Research Methodology, Mathematical Foundations, and Ethical Constraints")

    st.markdown("""
    <div class='adas-card'>
        <div class='adas-card-title'>🔬 Problem Statement & Context</div>
        <p>
        Autonomous driving systems and level-2+ ADAS rely heavily on optical cameras for lane keeping, forward collision warnings, and pedestrian detection.
        Under low ambient illumination (night driving) and suspended atmospheric aerosols (fog/smog), optical cameras capture low-contrast, noisy images with clipped dynamic ranges.
        Off-the-shelf deep neural networks (e.g. YOLO) often experience domain shift due to degraded edge gradients in dark regions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='adas-card'>
        <div class='adas-card-title'>📐 Enhancement Mathematics</div>
        <ul>
            <li>
                <strong>Gamma Correction:</strong> Non-linear mapping $V_{out} = 255 \times (V_{in} / 255)^{1/\gamma}$.
                When $\gamma > 1$, shadow values are amplified while preserving peak highlight values, expanding dark detail.
            </li>
            <li>
                <strong>LAB-Space CLAHE:</strong> Isolates the Lightness ($L^*$) channel in CIE $L^*a^*b^*$ space and redistributes local histograms across tile grids (e.g., $8 \\times 8$)
                with clip limits to prevent amplifying background sensor noise.
            </li>
            <li>
                <strong>Simulated Fog Physics:</strong> Koschmieder's Law formulation:
                $I(x) = J(x)e^{-\\beta d(x)} + A(1 - e^{-\\beta d(x)})$,
                modeled as an atmospheric veil testbed to evaluate detector resilience under low visibility.
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='adas-card'>
        <div class='adas-card-title'>⚖️ Ethical Constraints & Disclaimers</div>
        <div class='notice-box'>
            <strong>Non-Calibrated Proximity Disclosure:</strong><br>
            Computing real-world metric distances (in metres) from monocular 2D bounding boxes requires known camera intrinsic matrices ($f_x, f_y, c_x, c_y$),
            extrinsic pitch angles, and flat-ground plane assumptions. Arbitrary scalar formulas are mathematically unfounded and dangerous.<br><br>
            Therefore, VisionHack ADAS displays image-space viewport area heuristics only. This software is an engineering study and is strictly <strong>NOT</strong> certified for safety-critical vehicle control or autonomous actuation.
        </div>
    </div>
    """, unsafe_allow_html=True)
