import pytest
import numpy as np
from vision.metrics import compute_image_metrics


def test_metrics_keys_and_ranges():
    # Synthetic checkerboard image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = 255
    img[50:, 50:] = 255

    metrics = compute_image_metrics(img)

    required_keys = [
        "mean_luminance",
        "rms_contrast",
        "laplacian_variance",
        "dark_pixel_pct",
        "saturated_pixel_pct",
        "experimental_quality_score",
        "is_calibrated",
        "label"
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric key: {key}"

    assert metrics["is_calibrated"] is False
    assert 0.0 <= metrics["mean_luminance"] <= 255.0
    assert 0.0 <= metrics["rms_contrast"] <= 128.0
    assert metrics["laplacian_variance"] >= 0.0
    assert 0.0 <= metrics["dark_pixel_pct"] <= 100.0
    assert 0.0 <= metrics["saturated_pixel_pct"] <= 100.0


def test_contrast_difference():
    # Low contrast flat gray vs high contrast noise
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    metrics_flat = compute_image_metrics(flat)

    noisy = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    metrics_noisy = compute_image_metrics(noisy)

    assert metrics_flat["rms_contrast"] == 0.0
    assert metrics_flat["laplacian_variance"] == 0.0
    assert metrics_noisy["rms_contrast"] > metrics_flat["rms_contrast"]
    assert metrics_noisy["laplacian_variance"] > metrics_flat["laplacian_variance"]
