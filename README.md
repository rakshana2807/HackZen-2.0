# VisionHack ADAS — AI-Powered Visibility Enhancement & Road Scene Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg)](https://opencv.org/)

VisionHack ADAS is a computer vision and deep learning prototype designed to enhance roadway scene visibility and improve object detection recall under adverse lighting, night conditions, and atmospheric disturbances.

---

## 🌟 Key Features

1. **Original vs Enhanced Detection Comparison (High-Priority Workflow)**:
   - Run the identical pretrained Ultralytics **YOLOv8n** model on both raw and enhanced road scenes simultaneously.
   - Dual side-by-side annotated views with bounding boxes, labels, confidence percentages, and visual area tags.
   - Comprehensive comparative detection tables and delta counts.
   - Direct download buttons for enhanced images and both annotated outputs (PNG).

2. **Visibility Enhancement & Statistical Quality Analysis**:
   - **Gamma Correction**: Dynamic non-linear shadow expansion via precomputed lookup tables (LUT).
   - **LAB-Space CLAHE**: Contrast-limited adaptive histogram equalization isolated to the luminance channel ($L^*$) to prevent chromatic distortion.
   - **Composite Low-Light Pipeline**: Bilateral edge-preserving noise suppression followed by gamma and CLAHE.
   - **Simulated Fog Mode**: Koschmieder's Law atmospheric veil simulation for evaluating pipeline robustness.
   - Comparative metric cards and visual bar charts displaying mean luminance, RMS contrast, Laplacian sharpness variance, and relative quality heuristics.

3. **Detection Analytics Dashboard**:
   - Quantitative breakdown showing total detected objects, class frequency histograms, average detection confidence, and per-object confidence distributions.
   - Direct comparative KPIs between raw and enhanced inferences with measurable execution latency.

4. **Robust Video Stream & Batch Analysis**:
   - Frame-by-frame pipeline processing without loading entire video files into memory.
   - Configurable frame-sampling stride (every $N^{\text{th}}$ frame) and max frame ceilings to ensure fast, responsive laptop execution.
   - Real-time frame counter, progress indicator, and downloadable annotated MP4 video re-encoded via OpenCV.

---

## 📂 Project Architecture

```
HackZen 2.0/
├── app.py                          # Unified Streamlit dashboard (8 sections)
├── requirements.txt                # Core CPU-friendly dependencies
├── requirements-optional.txt       # Optional heavy neural model dependencies
├── README.md                       # Comprehensive guide & 5 approaches walkthrough
├── DOCUMENTATION.md                # Research methodology, math, and limitations
├── generate_samples.py             # Synthetic road scenes & video generator
├── vision/
│   ├── __init__.py                 # Vision package initialization
│   ├── enhancement.py              # Gamma, CLAHE, Low-light & fog algorithms
│   ├── detection.py                # Ultralytics YOLOv8 detector & bounding box overlays
│   ├── metrics.py                  # Statistical contrast, sharpness, and visibility indicators
│   ├── tracking.py                 # ByteTrack persistent tracking & motion trails
│   ├── retrieval.py                # DINOv2 (ViT-S/14) embeddings & nearest-neighbor search
│   ├── segmentation.py             # MobileSAM instance mask generator & area analysis
│   └── restoration.py              # Generative diffusion restoration & classical comparisons
├── tests/
│   ├── test_enhancement.py         # Unit tests for enhancement algorithms
│   ├── test_detection.py           # Unit tests for YOLO detection & rendering
│   ├── test_metrics.py             # Unit tests for image statistical metrics
│   ├── test_comparison_and_video.py# Unit tests for comparison logic and video sampling
│   └── test_advanced_vision.py     # Unit tests for retrieval, SAM geometry, tracking & restoration
├── samples/
│   ├── README.md                   # Sample assets description
│   ├── road_day_normal.jpg         # Clear daylight road scene
│   ├── road_night_dark.jpg         # Low-light nighttime road scene
│   ├── road_foggy_simulated.jpg    # Atmospheric fog scenario
│   └── demo_road_clip.mp4          # Multi-frame video clip
└── outputs/
    └── .gitkeep                    # Directory for exported annotations
```

---

## 🛠️ Prerequisites & Installation

### Requirements
- Python 3.11, 3.12, or 3.13
- Git

### Installation Steps

1. **Clone or Navigate to the Workspace**:
   ```bash
   cd "HackZen 2.0"
   ```

2. **Create and Activate a Virtual Environment (Recommended)**:
   ```bash
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Application

### 1. Launch the Dashboard
Run the Streamlit application:
```bash
python -m streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 2. Generate Sample Media

Sample images and a short demonstration video can be generated locally using the included script:

```bash
python generate_samples.py
```

This creates synthetic daylight, low-light, and simulated-fog road scenes, along with a short video clip, in the `samples/` directory. These are demonstration assets, not real-world road footage.

### 3. Run Automated Tests
Execute the full test suite with `pytest`:
```bash
python -m pytest tests/ -v
```

---

## 🧭 Hackathon Demo Walkthrough

1. **Dashboard Overview**: Start on the overview tab to introduce the problem space: sensor degradation in night and adverse road scenarios.
2. **Original vs Enhanced Detection Comparison (Core Demo)**:
   - Navigate to `⚖️ Original vs Enhanced Detection`.
   - Select the pre-bundled `Night Road Scene (Dark)` and the `Combined Low-Light Pipeline`.
   - Observe the dual annotated panels side by side: raw image vs enhanced image running the identical YOLOv8n detector.
   - Review the detailed comparison table and delta count (+/- detections) while pointing out the scientific disclaimer.
   - Click to download the enhanced image and annotated outputs.
3. **Image Enhancement & Quality Analysis**:
   - Open `🖼️ Image Enhancement` to demonstrate the contrast and sharpness transformations.
   - Inspect the interactive comparison bar chart comparing mean luminance, RMS contrast, and Laplacian sharpness.
4. **Detection Analytics Dashboard**:
   - Open `📈 Detection Analytics` to review genuine quantitative metrics: average confidence percentages, class frequency bars, and inference latencies.
5. **Video Stream Analysis**:
   - Open `🎥 Video Analysis`, select `demo_road_clip.mp4`, configure the frame-sampling stride (e.g. 1 or 2), and click `Start Frame-by-Frame Pipeline`.
   - Watch the live frame preview and download the finished MP4 video.
6. **Methodology & Ethics**:
   - Conclude on `About & Methodology` to explain the mathematical foundations and transparent image-space proximity heuristics.

---

## ⚠️ Limitations & Disclaimers

- **Non-Calibrated Proximity**: Depth cannot be accurately calculated from single bounding box widths without extrinsic pitch, roll, camera height, and intrinsic focal length ($f_x, f_y$). Heuristic proximity tags are purely illustrative.
- **Atmospheric Fog**: Koschmieder simulation is an optical testbed, not an active sensor for meteorological fog forecasting.
- **Safety**: VisionHack ADAS is an educational and engineering prototype and must **never** be deployed for direct physical control of an automobile.
