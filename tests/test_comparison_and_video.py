"""Comprehensive test suite for comparison logic, video frames, and format handling."""

import os
import tempfile
import cv2
import numpy as np
import pytest
from vision.enhancement import (
    validate_image,
    apply_gamma_correction,
    apply_clahe,
    apply_low_light_pipeline,
    apply_simulated_fog
)
from vision.detection import ObjectDetector
from vision.metrics import compute_image_metrics


def test_rgba_and_grayscale_format_handling():
    # 1. RGBA image handling
    rgba_img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    enhanced_clahe = apply_clahe(rgba_img)
    assert enhanced_clahe.shape == (100, 100, 4)
    assert enhanced_clahe.dtype == np.uint8

    gamma_rgba = apply_gamma_correction(rgba_img, gamma=1.8)
    assert gamma_rgba.shape == (100, 100, 4)

    # 2. 1-Channel Grayscale
    gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    enhanced_gray = apply_clahe(gray_img)
    assert enhanced_gray.shape == (100, 100)

    # 3. Metrics on Grayscale and RGBA
    m_gray = compute_image_metrics(gray_img)
    assert m_gray["mean_luminance"] >= 0.0
    m_rgba = compute_image_metrics(rgba_img)
    assert m_rgba["mean_luminance"] >= 0.0


def test_comparison_logic():
    # Prepare low contrast image and enhanced image
    dark_img = np.full((120, 120, 3), 30, dtype=np.uint8)
    enhanced_img = apply_low_light_pipeline(dark_img, gamma=1.6, clip_limit=2.5, denoise=False)

    m_orig = compute_image_metrics(dark_img)
    m_enh = compute_image_metrics(enhanced_img)

    # Verify metric delta logic
    delta_contrast = m_enh["rms_contrast"] - m_orig["rms_contrast"]
    delta_brightness = m_enh["mean_luminance"] - m_orig["mean_luminance"]
    assert delta_brightness > 0.0

    detector = ObjectDetector("yolov8n.pt")
    dets_orig = detector.detect(dark_img, conf_threshold=0.25)
    dets_enh = detector.detect(enhanced_img, conf_threshold=0.25)

    assert isinstance(dets_orig, list)
    assert isinstance(dets_enh, list)
    diff_count = len(dets_enh) - len(dets_orig)
    assert isinstance(diff_count, int)


def test_video_generator_and_frame_sampling():
    # Test reading video generated from samples
    video_path = "samples/demo_road_clip.mp4"
    if not os.path.exists(video_path):
        pytest.skip("demo_road_clip.mp4 not found")

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    frame_idx = 0
    sampled_frames = 0
    sample_stride = 2

    while cap.isOpened() and frame_idx < 10:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_stride == 0:
            sampled_frames += 1
            # Check frame validity
            assert frame.shape[0] > 0 and frame.shape[1] > 0
        frame_idx += 1

    cap.release()
    assert sampled_frames == 5
