"""Metrics for image-quality and visibility analysis.

IMPORTANT NOTICE:
All indicators in this module are experimental contrast and sharpness metrics
derived directly from digital pixel distributions.
They are NOT calibrated physical distance measurements, NOT meteorologically certified,
and NOT intended for automated vehicle navigation or life-critical decisions.
"""

from typing import Dict, Any
import cv2
import numpy as np
from .enhancement import validate_image


def compute_image_metrics(image: np.ndarray) -> Dict[str, Any]:
    """Compute measurable image statistical properties.

    Metrics computed:
    - Mean Luminance: Average pixel intensity across the grayscale conversion.
    - Global Contrast (RMS Contrast): Standard deviation of luminance values [0-127.5].
    - Laplacian Variance (Sharpness/Blur indicator): Variance of the 2D Laplacian operator.
      Higher values generally indicate sharper edge definitions, lower values imply blur or fog.
    - Dark Pixel Ratio: Percentage of pixels with luminance < 30.
    - Saturated Pixel Ratio: Percentage of pixels with luminance > 240.
    - Experimental Visibility Score: Heuristic normalized index in [0, 100] reflecting
      both contrast and sharpness for quick side-by-side relative comparisons.

    Args:
        image: uint8 input image (BGR or Grayscale).

    Returns:
        Dictionary of metric keys and computed floating-point values.
    """
    img = validate_image(image)

    # Convert to grayscale if necessary
    if len(img.shape) == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
    else:
        gray = img

    # 1. Luminance statistics
    mean_lum = float(np.mean(gray))
    std_lum = float(np.std(gray))  # Global RMS Contrast

    # 2. Laplacian operator for edge frequency / blur heuristic
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = float(laplacian.var())

    # 3. Dynamic range / clipping indicators
    total_pixels = gray.size
    dark_ratio = float(np.sum(gray < 30)) / total_pixels * 100.0
    saturated_ratio = float(np.sum(gray > 240)) / total_pixels * 100.0

    # 4. Experimental composite index (demonstration relative indicator ONLY)
    # Normalized: contrast (0-128) scaled, laplacian variance logarithmic scale
    contrast_score = min(1.0, std_lum / 64.0) * 50.0
    sharpness_score = min(1.0, np.log1p(laplacian_var) / np.log1p(1000.0)) * 50.0
    experimental_quality_score = round(contrast_score + sharpness_score, 1)

    return {
        "mean_luminance": round(mean_lum, 2),
        "rms_contrast": round(std_lum, 2),
        "laplacian_variance": round(laplacian_var, 2),
        "dark_pixel_pct": round(dark_ratio, 2),
        "saturated_pixel_pct": round(saturated_ratio, 2),
        "experimental_quality_score": experimental_quality_score,
        "is_calibrated": False,
        "label": "Experimental Image-Quality Indicator (Not physical distance)"
    }
