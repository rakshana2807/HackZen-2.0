"""VisionHack ADAS — Streamlit Web Application

AI-Powered Visibility Enhancement and Road-Scene Analysis Dashboard.
"""

import os
import time
import tempfile
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from vision.enhancement import (
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
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* Warning / Notice Banner */
.notice-box {
    background: rgba(245, 158, 11, 0.1);
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 4px;
    color: #fde68a;
    font-size: 0.9rem;
    margin-bottom: 16px;
}

/* Success / Status Pill */
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

/* Button & Widget Overrides */
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
def render_metrics_row(metrics: dict, title: str = "Image Statistics"):
    st.markdown(f"**{title}**")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{metrics['mean_luminance']}</div>
            <div class="metric-label">Mean Luminance (0-255)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{metrics['rms_contrast']}</div>
            <div class="metric-label">RMS Contrast (Std Dev)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{metrics['laplacian_variance']}</div>
            <div class="metric-label">Laplacian Var (Sharpness)</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{metrics['experimental_quality_score']} <span style='font-size:0.9rem;'>/ 100</span></div>
            <div class="metric-label">Exp. Visibility Score</div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# Sidebar Navigation & Settings
# ==========================================
with st.sidebar:
    st.title("🚗 VisionHack ADAS")
    st.caption("AI-Powered Visibility Enhancement & Road Scene Analysis")
    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "📊 Dashboard Overview",
            "🖼️ Image Analysis",
            "🎥 Video Analysis",
            "⚙️ Detection Settings",
            "📖 About & Methodology"
        ],
        index=0
    )

    st.divider()
    st.markdown("### Model Status")
    if detector.is_ready:
        st.markdown("<span class='status-pill'>● YOLOv8n Active</span>", unsafe_allow_html=True)
        st.caption("Ultralytics Pretrained Engine Ready")
    else:
        st.error(f"Detector Offline: {detector.error_message}")

    st.divider()
    st.markdown("""
    <div style='font-size: 0.75rem; color: #64748b;'>
    <strong>ADAS Prototype Disclaimer:</strong><br>
    All visibility indices and proximity tags are experimental heuristics.
    Not certified for vehicle control or physical road distance measurement.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 1: DASHBOARD OVERVIEW
# ==========================================
if selected_page == "📊 Dashboard Overview":
    st.markdown("## 📊 VisionHack ADAS Overview")
    st.markdown("#### Advanced Driver Assistance System Visibility Pipeline")

    st.markdown("""
    <div class='notice-box'>
        ⚠️ <strong>Notice:</strong> This dashboard is an engineering prototype for adverse weather and low-light road analysis.
        All contrast scores and spatial heuristics are experimental and not calibrated to real-world metric distances.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class='adas-card'>
            <div class='adas-card-title'>🎯 System Objectives</div>
            <p>
            Autonomous vehicles and Advanced Driver Assistance Systems (ADAS) experience performance degradation in low-light, nighttime, and adverse atmospheric conditions (fog/rain).
            VisionHack demonstrates a dual-stage vision architecture:
            </p>
            <ul>
                <li><strong>Stage 1 — Adaptive Pre-processing:</strong> Non-linear gamma expansion and LAB-space CLAHE to recover edge contrast and shadow features without washing out highlights.</li>
                <li><strong>Stage 2 — Perception & Object Detection:</strong> Real-time YOLOv8 inference to localize road entities (vehicles, pedestrians, cyclists, signs).</li>
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
            <div class='adas-card-title'>🚀 Quick Start Guide</div>
            <ol style='color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;'>
                <li>Navigate to <strong>🖼️ Image Analysis</strong> to upload road images or load bundled sample road scenes.</li>
                <li>Compare <strong>Gamma</strong>, <strong>CLAHE</strong>, or <strong>Low-Light</strong> enhancement pipelines side-by-side.</li>
                <li>Toggle YOLO detection on original vs enhanced images to observe contrast-driven bounding box recall.</li>
                <li>Visit <strong>🎥 Video Analysis</strong> to process multi-frame road footage with download support.</li>
            </ol>
            <div style='text-align: center; margin-top: 15px;'>
                <span class='status-pill'>Pipeline Ready for Testing</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick sample check
        st.markdown("""
        <div class='adas-card'>
            <div class='adas-card-title'>📁 Sample Assets Available</div>
            <div style='font-size: 0.85rem; color: #94a3b8;'>
                Pre-generated test scenes in <code>samples/</code>:
                <ul style='margin-top: 6px;'>
                    <li><code>road_day_normal.jpg</code></li>
                    <li><code>road_night_dark.jpg</code></li>
                    <li><code>road_foggy_simulated.jpg</code></li>
                    <li><code>demo_road_clip.mp4</code></li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 2: IMAGE ANALYSIS
# ==========================================
elif selected_page == "🖼️ Image Analysis":
    st.markdown("## 🖼️ Image Analysis & Visibility Pipeline")
    st.caption("Upload a road-scene image or test with pre-bundled scenarios.")

    # Top Controls Card
    with st.container():
        st.markdown("<div class='adas-card'>", unsafe_allow_html=True)
        col_src, col_enh, col_det_target = st.columns([2, 2, 2])

        with col_src:
            source_choice = st.radio(
                "Image Source",
                ["Upload Image", "Use Pre-bundled Sample Scene"],
                horizontal=True
            )

        with col_enh:
            enhancement_mode = st.selectbox(
                "Enhancement Pipeline",
                [
                    "None (Raw Input)",
                    "Gamma Correction",
                    "CLAHE (LAB Lightness)",
                    "Combined Low-Light Pipeline",
                    "Simulated Fog (Demo Only)"
                ],
                index=3
            )

        with col_det_target:
            detection_target = st.radio(
                "Run YOLO Detection On:",
                ["Enhanced Image", "Original Image", "Both Side-by-Side", "None (Enhance Only)"],
                index=0
            )

        # Parameter Sliders
        col_p1, col_p2, col_p3 = st.columns(3)
        if enhancement_mode == "Gamma Correction":
            with col_p1:
                gamma_val = st.slider("Gamma Value", min_value=0.2, max_value=3.5, value=1.6, step=0.1)
        elif enhancement_mode == "CLAHE (LAB Lightness)":
            with col_p1:
                clahe_clip = st.slider("Clip Limit", min_value=1.0, max_value=10.0, value=2.5, step=0.5)
            with col_p2:
                grid_sz = st.select_slider("Tile Grid Size", options=[4, 8, 16], value=8)
        elif enhancement_mode == "Combined Low-Light Pipeline":
            with col_p1:
                ll_gamma = st.slider("Gamma", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
            with col_p2:
                ll_clip = st.slider("CLAHE Clip", min_value=1.0, max_value=8.0, value=2.2, step=0.2)
            with col_p3:
                ll_denoise = st.checkbox("Bilateral Filter (Denoise)", value=True)
        elif enhancement_mode == "Simulated Fog (Demo Only)":
            with col_p1:
                fog_dens = st.slider("Fog Density Factor", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
            with col_p2:
                fog_bright = st.slider("Atmospheric Luminance", min_value=150, max_value=255, value=210, step=5)
            st.caption("⚠️ Note: Synthetic demonstration only. Not environmental fog detection.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Input Image Loading
    raw_bgr = None
    image_name = "image.jpg"

    if source_choice == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload Road Scene Image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image_name = uploaded_file.name
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if raw_bgr is None:
                st.error("Could not decode the uploaded image. Please check the file format.")
    else:
        sample_options = {
            "Night Road Scene (Dark)": "samples/road_night_dark.jpg",
            "Day Road Scene (Normal)": "samples/road_day_normal.jpg",
            "Simulated Fog Road Scene": "samples/road_foggy_simulated.jpg"
        }
        selected_sample_label = st.selectbox("Select Sample Scenario", list(sample_options.keys()))
        sample_path = sample_options[selected_sample_label]
        image_name = os.path.basename(sample_path)
        if os.path.exists(sample_path):
            raw_bgr = cv2.imread(sample_path)
        else:
            st.error(f"Sample file not found at {sample_path}. Please run generate_samples.py.")

    if raw_bgr is not None:
        h, w = raw_bgr.shape[:2]
        st.caption(f"Loaded Image: **{image_name}** | Dimensions: **{w} x {h} px**")

        # 1. Image Enhancement Execution
        t0 = time.perf_counter()
        if enhancement_mode == "None (Raw Input)":
            enhanced_bgr = raw_bgr.copy()
        elif enhancement_mode == "Gamma Correction":
            enhanced_bgr = apply_gamma_correction(raw_bgr, gamma=gamma_val)
        elif enhancement_mode == "CLAHE (LAB Lightness)":
            enhanced_bgr = apply_clahe(raw_bgr, clip_limit=clahe_clip, tile_grid_size=(grid_sz, grid_sz))
        elif enhancement_mode == "Combined Low-Light Pipeline":
            enhanced_bgr = apply_low_light_pipeline(
                raw_bgr, gamma=ll_gamma, clip_limit=ll_clip, denoise=ll_denoise
            )
        elif enhancement_mode == "Simulated Fog (Demo Only)":
            enhanced_bgr = apply_simulated_fog(raw_bgr, fog_density=fog_dens, fog_brightness=fog_bright)
        t_enh = (time.perf_counter() - t0) * 1000.0

        # 2. Compute Metrics
        orig_metrics = compute_image_metrics(raw_bgr)
        enh_metrics = compute_image_metrics(enhanced_bgr)

        # 3. Object Detection Inference
        conf_thresh = st.session_state.get("conf_threshold", 0.25)
        selected_classes = st.session_state.get("selected_classes", None)

        orig_annotated = None
        enh_annotated = None
        orig_dets = []
        enh_dets = []
        t_det = 0.0

        if detector.is_ready and detection_target != "None (Enhance Only)":
            t_det_start = time.perf_counter()
            if detection_target in ["Original Image", "Both Side-by-Side"]:
                orig_dets = detector.detect(raw_bgr, conf_threshold=conf_thresh, allowed_classes=selected_classes)
                orig_annotated = detector.draw_annotations(raw_bgr, orig_dets, show_proximity=True)

            if detection_target in ["Enhanced Image", "Both Side-by-Side"]:
                enh_dets = detector.detect(enhanced_bgr, conf_threshold=conf_thresh, allowed_classes=selected_classes)
                enh_annotated = detector.draw_annotations(enhanced_bgr, enh_dets, show_proximity=True)
            t_det = (time.perf_counter() - t_det_start) * 1000.0

        # Display side-by-side or individual outputs
        st.markdown("### Visual Inspection & Comparison")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 1. Input Image")
            if orig_annotated is not None:
                st.image(cv2.cvtColor(orig_annotated, cv2.COLOR_BGR2RGB), caption=f"Original + YOLO ({len(orig_dets)} detected)", use_container_width=True)
            else:
                st.image(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), caption="Original Input", use_container_width=True)
            render_metrics_row(orig_metrics, "Original Image Statistics")

        with c2:
            st.markdown(f"#### 2. Processed Output ({enhancement_mode})")
            if enh_annotated is not None:
                st.image(cv2.cvtColor(enh_annotated, cv2.COLOR_BGR2RGB), caption=f"Enhanced + YOLO ({len(enh_dets)} detected)", use_container_width=True)
            else:
                st.image(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB), caption="Enhanced Output", use_container_width=True)
            render_metrics_row(enh_metrics, "Enhanced Image Statistics")

        # Performance summary pill
        st.markdown(f"""
        <div style='background: #1e293b; padding: 10px 15px; border-radius: 8px; margin: 15px 0; font-size: 0.85rem; color: #94a3b8;'>
            ⏱️ <strong>Processing Latency:</strong> Enhancement: <span style='color:#38bdf8;'>{t_enh:.1f} ms</span> |
            YOLO Inference: <span style='color:#38bdf8;'>{t_det:.1f} ms</span> |
            Contrast Shift: <span style='color:#34d399;'>{enh_metrics['rms_contrast'] - orig_metrics['rms_contrast']:+.2f} RMS</span>
        </div>
        """, unsafe_allow_html=True)

        # Detection Summary Breakdown
        active_dets = enh_dets if detection_target == "Enhanced Image" else (orig_dets if detection_target == "Original Image" else enh_dets)
        if detection_target != "None (Enhance Only)":
            st.markdown("### 🎯 Detection Summary")
            dcol1, dcol2 = st.columns([1, 2])
            with dcol1:
                st.metric("Total Objects Detected", len(active_dets))
                # Class counts breakdown
                counts = {}
                for d in active_dets:
                    counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
                for cname, count in counts.items():
                    st.write(f"• **{cname}**: {count}")
            with dcol2:
                if active_dets:
                    table_data = [
                        {
                            "Class": d["class_name"],
                            "Confidence": f"{d['confidence']:.1%}",
                            "Bounding Box (x1, y1, x2, y2)": str(d["box"]),
                            "Relative Viewport Area": f"{d['relative_area_pct']}%",
                            "Spatial Proximity Tag": d["proximity_heuristic"]
                        }
                        for d in active_dets
                    ]
                    st.dataframe(table_data, use_container_width=True)
                else:
                    st.info("No objects detected above the selected confidence threshold.")

        # Download Options
        st.markdown("### 💾 Export Results")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            res, enc_img = cv2.imencode(".png", enhanced_bgr)
            if res:
                st.download_button(
                    label="⬇️ Download Enhanced Image (PNG)",
                    data=enc_img.tobytes(),
                    file_name=f"enhanced_{image_name}.png",
                    mime="image/png"
                )
        with d_col2:
            annotated_export = enh_annotated if enh_annotated is not None else (orig_annotated if orig_annotated is not None else enhanced_bgr)
            res2, enc_ann = cv2.imencode(".png", annotated_export)
            if res2:
                st.download_button(
                    label="⬇️ Download Annotated Detection (PNG)",
                    data=enc_ann.tobytes(),
                    file_name=f"annotated_{image_name}.png",
                    mime="image/png"
                )


# ==========================================
# PAGE 3: VIDEO ANALYSIS
# ==========================================
elif selected_page == "🎥 Video Analysis":
    st.markdown("## 🎥 Video Stream & Batch Analysis")
    st.caption("Process multi-frame road footage through the enhancement and detection pipeline.")

    st.markdown("""
    <div class='notice-box'>
        ⚙️ <strong>Frame Limiter:</strong> To avoid long processing times in demo environments,
        you can set a maximum frame ceiling. Output video is re-encoded with OpenCV.
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
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        v_enhancement = st.selectbox(
            "Video Enhancement Mode",
            ["Combined Low-Light", "Gamma Correction", "CLAHE", "None (Raw)"]
        )
    with v_col2:
        max_frames_to_process = st.slider("Max Frames to Process", min_value=10, max_value=150, value=30, step=10)
    with v_col3:
        run_v_detection = st.checkbox("Run YOLO Detection on Frames", value=True)

    if video_path_to_process and st.button("🚀 Start Video Processing Pipeline"):
        cap = cv2.VideoCapture(video_path_to_process)
        if not cap.isOpened():
            st.error("Failed to open video file. Ensure valid codec (H.264 / MP4V).")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_limit = min(total_frames, max_frames_to_process) if total_frames > 0 else max_frames_to_process

            st.write(f"Processing up to **{frames_limit}** frames ({width}x{height} @ {fps:.1f} FPS)...")

            # Temporary output video file
            out_temp_path = os.path.join(tempfile.gettempdir(), f"visionhack_out_{int(time.time())}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_temp_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            preview_slot = st.empty()

            processed_count = 0
            t_start = time.perf_counter()
            total_dets_found = 0

            conf_thresh = st.session_state.get("conf_threshold", 0.25)
            selected_classes = st.session_state.get("selected_classes", None)

            while cap.isOpened() and processed_count < frames_limit:
                ret, frame = cap.read()
                if not ret:
                    break

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
                if run_v_detection and detector.is_ready:
                    dets = detector.detect(proc_frame, conf_threshold=conf_thresh, allowed_classes=selected_classes)
                    total_dets_found += len(dets)
                    final_frame = detector.draw_annotations(proc_frame, dets, show_proximity=True)
                else:
                    final_frame = proc_frame

                # Overlay frame counter
                cv2.putText(
                    final_frame,
                    f"ADAS FRAME {processed_count+1}/{frames_limit}",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (56, 189, 248),
                    2,
                    cv2.LINE_AA
                )

                writer.write(final_frame)
                processed_count += 1

                # Update live preview every few frames
                if processed_count % 3 == 0 or processed_count == frames_limit:
                    progress_bar.progress(processed_count / frames_limit)
                    status_text.text(f"Processed frame {processed_count} / {frames_limit}...")
                    preview_slot.image(
                        cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Live Preview (Frame {processed_count})",
                        use_container_width=True
                    )

            cap.release()
            writer.release()
            elapsed = time.perf_counter() - t_start

            st.success(f"Processing complete! Processed {processed_count} frames in {elapsed:.2f}s ({processed_count / max(0.001, elapsed):.1f} FPS).")

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
# PAGE 4: DETECTION SETTINGS
# ==========================================
elif selected_page == "⚙️ Detection Settings":
    st.markdown("## ⚙️ ADAS Perception & Detection Settings")
    st.caption("Tune YOLOv8 confidence thresholds, NMS filters, and target road classes.")

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

    # Class selection
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
# PAGE 5: ABOUT & METHODOLOGY
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
        Off-the-shelf deep neural networks (e.g. YOLO) often fail to detect vehicles and pedestrians due to missing edge gradients in dark regions.
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
                <strong>LAB-Space CLAHE:</strong> Traditional RGB histogram equalization distorts color hue and saturation.
                VisionHack isolates the Lightness ($L^*$) channel in CIE $L^*a^*b^*$ space and redistributes local histograms across tile grids (e.g., $8 \\times 8$)
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
            extrinsic pitch angles, and flat-ground plane assumptions. Arbitrary scalar formulas (e.g. $1200 / w$) are mathematically unfounded and dangerous.<br><br>
            Therefore, VisionHack ADAS displays image-space viewport area heuristics only. This software is an engineering study and is strictly <strong>NOT</strong> certified for safety-critical vehicle control or autonomous actuation.
        </div>
    </div>
    """, unsafe_allow_html=True)
