# VisionHack ADAS — AI-Powered Visibility Enhancement & Road Scene Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg)](https://opencv.org/)

VisionHack ADAS is a computer vision and deep learning prototype designed to enhance roadway scene visibility and improve object detection recall under adverse lighting, night conditions, and atmospheric disturbances.

---

## 🌟 Key Features

1. **Adaptive Image Enhancement**:
   - **Gamma Correction**: Dynamic non-linear shadow expansion via precomputed lookup tables (LUT).
   - **LAB-Space CLAHE**: Contrast-limited adaptive histogram equalization isolated to the luminance channel ($L^*$) to prevent color distortions.
   - **Composite Low-Light Pipeline**: Bilateral edge-preserving noise suppression followed by gamma and CLAHE.
   - **Simulated Fog Mode**: Koschmieder's Law atmospheric veil simulation for evaluating pipeline robustness.

2. **Real-Time Object Detection**:
   - Integrated with Ultralytics **YOLOv8n** for single-stage deep neural inference.
   - Real-time road actor classification (cars, pedestrians, buses, trucks, cyclists, traffic signals).
   - User-configurable confidence thresholds and class filtering.
   - Side-by-side detector comparison (Original vs. Enhanced input).

3. **Experimental Quality & Proximity Heuristics**:
   - Quantitative image metrics: RMS luminance contrast, Laplacian sharpness variance, dark/saturated pixel ratios.
   - Viewport-area proximity heuristic tags (explaining clearly that real-world metric distances require calibrated stereo/monocular camera intrinsics).

4. **Multi-Media Processing**:
   - Interactive single-image analysis with side-by-side metrics inspection and PNG exports.
   - Frame-by-frame video processing engine with frame limiters and downloadable annotated MP4 export.

---

## 📂 Project Architecture

```
HackZen 2.0/
├── app.py                     # Main Streamlit ADAS dashboard
├── requirements.txt           # Python dependency specifications
├── README.md                  # Project overview and quickstart guide
├── DOCUMENTATION.md           # Engineering methodology, math, and ethics
├── generate_samples.py        # Synthetic road scenes & video generator
├── vision/
│   ├── __init__.py            # Vision package init
│   ├── enhancement.py         # Gamma, CLAHE, Low-light & fog algorithms
│   ├── detection.py           # Ultralytics YOLOv8 detector & bounding box overlays
│   └── metrics.py             # Contrast, sharpness, and visibility indicators
├── tests/
│   ├── test_enhancement.py    # Unit tests for enhancement algorithms
│   ├── test_detection.py      # Unit tests for YOLO detection & rendering
│   └── test_metrics.py        # Unit tests for image statistical metrics
├── samples/
│   ├── README.md              # Sample assets description
│   ├── road_day_normal.jpg    # Clear daylight road scene
│   ├── road_night_dark.jpg    # Low-light nighttime road scene
│   ├── road_foggy_simulated.jpg # Atmospheric fog scenario
│   └── demo_road_clip.mp4     # Multi-frame video clip
└── outputs/
    └── .gitkeep               # Directory for exported annotations
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

### 2. Generate Sample Media (Optional)
Pre-generated samples are included, but you can regenerate them anytime:
```bash
python generate_samples.py
```

### 3. Run Automated Tests
Execute the full test suite with `pytest`:
```bash
python -m pytest tests/ -v
```

---

## 🧭 Hackathon Demo Walkthrough

1. **Dashboard Overview**: Start on the overview tab to introduce the dual-stage ADAS architecture.
2. **Image Analysis (Night Scene)**:
   - Select the pre-bundled `Night Road Scene (Dark)`.
   - Choose `Combined Low-Light Pipeline`.
   - Set detection target to `Both Side-by-Side`.
   - Highlight the contrast shift and how enhancement increases feature prominence for YOLOv8.
3. **Image Analysis (Simulated Fog)**:
   - Select `Simulated Fog Road Scene`.
   - Note the Laplacian variance drop indicating haze/blur.
   - Apply CLAHE with clip limit `3.0` to recover road boundaries.
4. **Video Stream Analysis**:
   - Open `Video Analysis` and click `Start Video Processing Pipeline`.
   - Watch the live frame-by-frame processing counter and download the annotated MP4.
5. **Methodology & Ethics**:
   - Open `About & Methodology` to explain why uncalibrated monocular distances are avoided in favor of transparent viewport heuristics.

---

## ⚠️ Limitations & Disclaimers

- **Non-Calibrated Proximity**: Depth cannot be accurately calculated from single bounding box widths without extrinsic pitch, roll, camera height, and intrinsic focal length ($f_x, f_y$). Heuristic proximity tags are purely illustrative.
- **Atmospheric Fog**: Koschmieder simulation is an optical testbed, not an active sensor for meteorological fog forecasting.
- **Safety**: VisionHack ADAS is an educational and engineering prototype and must **never** be deployed for direct physical control of an automobile.
