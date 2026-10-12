# VisionHack ADAS — Engineering & Research Documentation

## 1. Problem Statement

Modern Advanced Driver Assistance Systems (ADAS) and Level 2+ semi-autonomous driving platforms rely heavily on visible-spectrum optical cameras to perform lane detection, traffic sign recognition, vehicle tracking, and vulnerable road user (VRU) detection.

Under degraded atmospheric and environmental conditions:
1. **Low-Light / Night Driving**: Low signal-to-noise ratios (SNR), underexposed sensors, and headlight glares suppress edge gradients, leading to missed detections.
2. **Fog & Atmospheric Scattering**: Suspended aerosol particles cause Mie scattering, attenuating reflected target luminance and imposing an ambient veil that reduces spatial frequency contrast.
3. **Deep Learning Latency & Fragility**: Neural networks trained primarily on daylight datasets suffer domain shift when evaluating high-noise or washed-out frames.

---

## 2. Proposed Architecture & Solution

VisionHack ADAS addresses this through a modular, dual-stage visual enhancement and perception pipeline:

```
[Raw Camera / Video Stream]
            │
            ▼
┌───────────────────────────────────────────────┐
│       Stage 1: Pre-Processing Pipeline        │
│  - Edge-Preserving Bilateral Denoising        │
│  - Non-Linear Gamma Expansion (LUT)           │
│  - Luminance-Channel LAB-Space CLAHE          │
└───────────────────────┬───────────────────────┘
                        │
                        ├───────────────────────────────┐
                        ▼                               ▼
        ┌──────────────────────────────┐ ┌──────────────────────────────┐
        │ Stage 2: Deep Perception     │ │ Stage 3: Statistical Quality │
        │  - Ultralytics YOLOv8n       │ │  - RMS Luminance Contrast    │
        │  - Real-time Road Classes    │ │  - Laplacian Variance        │
        │  - NMS & Confidence Filter   │ │  - Viewport Proximity Score  │
        └───────────────┬──────────────┘ └──────────────┬───────────────┘
                        │                               │
                        └───────────────┬───────────────┘
                                        ▼
                        [Streamlit Real-Time Dashboard]
```

---

## 3. Mathematical Foundations of Enhancement Algorithms

### 3.1 Gamma Correction (Non-Linear Luminance Expansion)
Standard 8-bit digital imaging maps luminance linearly to $[0, 255]$. In underexposed scenes, critical road boundary information is compressed into low numerical bins.

Gamma correction applies a power-law transformation:
$$I_{out}(x, y) = 255 \times \left(\frac{I_{in}(x, y)}{255}\right)^{\frac{1}{\gamma}}$$

- When $\gamma > 1.0$, the function exhibits high derivative at the origin, amplifying dark and mid-tone regions while preventing highlight saturation.
- Computational efficiency: Computed using an 8-bit precalculated lookup table ($256$ elements) executed via `cv2.LUT`, yielding $<1.0\text{ ms}$ latency on standard CPUs.

### 3.2 LAB-Space CLAHE (Contrast-Limited Adaptive Histogram Equalization)
Global histogram equalization frequently amplifies background sensor noise and causes unnatural color shifts when applied naively to RGB channels.

VisionHack ADAS separates chromatic and achromatic signals by converting the image to CIE $L^*a^*b^*$ color space:
1. Extract channel $L^*$ (Lightness, perceptual luminance).
2. Divide $L^*$ into non-overlapping contextual tiles (e.g. $8 \times 8$).
3. Apply contrast limiting by clipping the histogram at threshold $\beta_{clip}$. Clipped pixels are redistributed uniformly across all bins.
4. Apply bilinear interpolation across tile boundaries to eliminate blocking artifacts.
5. Reconstruct the image via inverse transformation $\text{LAB} \rightarrow \text{BGR}$.

### 3.3 Atmospheric Scattering Physics (Koschmieder's Law)
For controlled resilience testing, atmospheric fog is modeled using Koschmieder's formulation:
$$I(x) = J(x)e^{-\beta d(x)} + A\left(1 - e^{-\beta d(x)}\right)$$
where $J(x)$ is scene radiance, $\beta$ is the atmospheric extinction coefficient, $d(x)$ is optical path distance (modeled via a vertical horizon-to-bumper depth gradient), and $A$ is the atmospheric light intensity ($A \approx 210$).

---

## 4. Object Detection Methodology

VisionHack ADAS employs **YOLOv8n** (Ultralytics), a state-of-the-art anchor-free convolutional architecture optimized for high-speed edge inference:
- **Input Resolution**: $640 \times 640$ adaptive letterbox scaling.
- **Road Entity Filtering**: Filtered to critical COCO road classes: Person (0), Bicycle (1), Car (2), Motorcycle (3), Bus (5), Truck (7), Traffic Light (9), Stop Sign (11).
- **Inference Latency**: $\approx 25\text{--}40\text{ ms}$ on standard modern x86 CPU architectures without hardware acceleration.

---

## 5. Image-Quality Metrics vs Physical Visibility

### Measurable Statistical Properties
1. **RMS Contrast**:
   $$\sigma_{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (I_i - \bar{I})^2}$$
   Quantifies overall spread of luminance values across the grayscale range $[0, 127.5]$.
2. **Laplacian Variance (Edge Energy / Sharpness)**:
   $$\text{Var}(\nabla^2 I) = \text{Var}\left(\frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}\right)$$
   Sharp, unhazed scenes feature steep intensity transitions yielding high variance. Fog, heavy haze, and severe blur suppress high-frequency spatial gradients, driving variance towards zero.

### Scientific Limitations
- **No Physical Distance Claim**: Grayscale variance does not equal atmospheric optical range (MOR) in meters. A flat white wall on a sunny day produces zero Laplacian variance despite clear visibility.
- **No Calibrated Metres**: Without multi-camera stereo disparity or calibrated LiDAR/radar, single 2D bounding boxes cannot provide reliable metric distances in meters. Proximity is reported strictly as an image-space viewport area heuristic.

---

## 6. Experimental Evaluation and Results

### 6.1 Evaluation Method

The prototype was evaluated across clear, low-light, and simulated-fog road-scene images using the synthetic test generator (`generate_samples.py`) for repeatable functional testing. These scenes demonstrate pipeline operation and metric stability under controlled conditions, serving as a functional evaluation rather than an exhaustive real-world road-image benchmark.

For each test scenario:
1. The original image's mean luminance, RMS contrast, Laplacian variance, dark-pixel ratio, and saturated-pixel ratio were measured.
2. The selected enhancement pipeline was executed with exact processing time recorded.
3. The identical metrics were calculated for the enhanced output.
4. Ultralytics YOLOv8n object detection was executed on both inputs at a confidence threshold of 0.20.
5. Detected classes, confidence scores, latencies, and observed behavior were logged.

### 6.2 Recorded Empirical Results

The following values were populated directly from live runs of the application pipeline on the test scenes:

| Evaluation Scenario | Pipeline / Condition | Mean Luminance | RMS Contrast | Laplacian Var (Sharpness) | Dark Pixels (<30) | Processing Time | Detections & Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Low-Light Road** | Original (Raw Input) | 24.59 | 11.81 | 31.82 | 69.35% | — | 1 detection: `stop sign` (conf: 25.1%) |
| **Low-Light Road** | Combined Low-Light ($\gamma=1.5$, CLAHE) | 55.83 | 17.11 | 32.60 | 0.00% | 175.9 ms | 0 detections above 0.20 threshold |
| **Simulated Fog Road** | Original (Raw Input) | 150.22 | 45.33 | 51.53 | 0.00% | — | 0 detections |
| **Simulated Fog Road** | LAB CLAHE ($\text{clip}=2.5$) | 151.01 | 43.88 | 69.16 | 0.00% | 7.36 ms | 0 detections |
| **Clear Day Road** | Original (Raw Input) | 125.97 | 58.68 | 112.36 | 0.55% | — | Baseline benchmark |

#### Additional Pipeline Measurements:
- **Video Processing Performance**: 30 frames processed in ~1.5 seconds (~20.0 FPS) with full frame-by-frame enhancement, YOLO inference, and MP4 re-encoding.
- **Automated Test Suite**: 11 passed, 0 failed across 3 test suites (`test_detection.py`, `test_enhancement.py`, `test_metrics.py`).
- **Claimed Improvements**: No verified detection-accuracy improvement or real-world visibility improvement is claimed at this stage.

### 6.3 Interpretation & Scientific Limitations

1. **Luminance and Dynamic Range Expansion**:
   Gamma correction effectively brightened underexposed shadow regions (mean luminance rose from 24.59 to 55.83, eliminating the 69.35% dark-pixel clipping). CLAHE enhanced local high-frequency gradients in simulated fog, increasing Laplacian variance from 51.53 to 69.16.

2. **Divergence Between Enhancement and Detection**:
   Changes in image-quality metrics do not automatically demonstrate improved object detection. In the low-light test scene, brightening eliminated an ambiguous edge artifact that had previously produced a low-confidence false detection (`stop sign` at 25.1%). In synthetic scenes lacking photorealistic texture distributions, enhancement modifies spatial gradients without guaranteeing higher neural recall.

3. **Artifacts and Noise Amplification**:
   Aggressive non-linear stretching can amplify sensor noise, quantization banding, and atmospheric halo artifacts. Detection models must be rigorously evaluated on annotated ground-truth datasets (e.g., BDD100K, ExDark, Foggy Cityscapes) using standardized metrics such as Precision, Recall, and mean Average Precision (mAP@50, mAP@50:95).

4. **Educational & Experimental Prototype**:
   The prototype is intended strictly for educational and experimental evaluation. It has not been validated or certified for real-world driver assistance or safety-critical vehicular deployment.

---

## 7. Ethical, Legal, and Safety Considerations

1. **Human-in-the-Loop ADAS**: This prototype is intended solely for situational awareness research and driver assistance display demonstration. It must not be wired to braking, steering, or throttle control.
2. **Failure Modes**: Over-amplifying severely dark frames with high sensor noise can occasionally generate false positives. The edge-preserving bilateral filter mitigates but does not fully eliminate high-frequency noise spikes.
3. **Data Privacy**: The software does not store, transmit, or harvest processed road imagery; all inferences execute strictly in local memory.
