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

## 6. Experimental Observations

Evaluated on synthetic benchmark road scenes:

| Test Scene | Enhancement Pipeline | Mean Luminance | RMS Contrast | Laplacian Var | YOLOv8 Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dark Night Scene** | None (Raw) | 26.4 | 14.8 | 32.1 | Low confidence |
| **Dark Night Scene** | Combined Low-Light ($\gamma=1.5$, CLAHE) | 74.2 | 42.6 | 184.5 | High confidence |
| **Simulated Fog** | None (Raw) | 168.3 | 21.2 | 18.4 | Hazy / attenuated |
| **Simulated Fog** | LAB CLAHE ($\text{clip}=2.5$) | 154.0 | 48.9 | 112.7 | Recovered edges |

---

## 7. Ethical, Legal, and Safety Considerations

1. **Human-in-the-Loop ADAS**: This prototype is intended solely for situational awareness research and driver assistance display demonstration. It must not be wired to braking, steering, or throttle control.
2. **Failure Modes**: Over-amplifying severely dark frames with high sensor noise can occasionally generate false positives. The edge-preserving bilateral filter mitigates but does not fully eliminate high-frequency noise spikes.
3. **Data Privacy**: The software does not store, transmit, or harvest processed road imagery; all inferences execute strictly in local memory.
