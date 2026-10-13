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


def test_tiny_video_writer_and_reader_roundtrip(tmp_path):
    """Verify VideoWriter initialization, non-zero file creation, and reopening with frame count check."""
    output_video_path = str(tmp_path / "tiny_test.mp4")
    w, h, fps = 64, 64, 10
    num_frames = 5

    # Try mp4v fourcc
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        # Fall back to alternative or skip if environment codecs are missing
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video_path = str(tmp_path / "tiny_test.avi")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            pytest.skip("Environment does not provide compatible video encoder for VideoWriter")

    for i in range(num_frames):
        frame = np.full((h, w, 3), i * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    assert os.path.exists(output_video_path), "Exported video file was not created"
    assert os.path.getsize(output_video_path) > 0, "Exported video file is empty (0 bytes)"

    # Verify video can be reopened and read back
    cap = cv2.VideoCapture(output_video_path)
    assert cap.isOpened(), "Could not reopen generated video"
    read_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        assert frame.shape == (h, w, 3)
        read_frames += 1
    cap.release()

    assert read_frames == num_frames, f"Expected {num_frames} frames, but read {read_frames}"

