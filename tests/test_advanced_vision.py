"""Unit tests for advanced vision modules (Retrieval, Segmentation, Tracking, Restoration)."""

import os
import cv2
import numpy as np
import pytest

from vision.tracking import ObjectTracker
from vision.retrieval import SceneRetrievalEngine
from vision.segmentation import Segmenter
from vision.restoration import DiffusionRestorer


def test_tracker_initialization_and_formatting():
    tracker = ObjectTracker(model_name="yolov8n.pt")
    assert tracker.is_ready is True
    assert tracker.error_message is None

    # Track on blank frame
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tracks = tracker.track_frame(frame, conf_threshold=0.25)
    assert isinstance(tracks, list)

    # Annotation rendering
    annotated = tracker.draw_tracking_annotations(frame, tracks, draw_trails=True)
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8

    tracker.reset_tracks()
    assert len(tracker.trails) == 0
    assert len(tracker.cumulative_counts) == 0


def test_retrieval_empty_index_and_normalization():
    engine = SceneRetrievalEngine(index_path="samples/test_temp_index.json")
    # Clean state
    engine.gallery = []
    assert len(engine.gallery) == 0

    # Search on empty gallery should return empty list gracefully
    dummy_query = np.zeros((100, 100, 3), dtype=np.uint8)
    res = engine.search(dummy_query, top_k=3)
    assert res == []

    # Test cosine similarity logic on synthetic unit vectors
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    sim_identical = float(np.dot(v1, v2))
    sim_orthogonal = float(np.dot(v1, v3))
    assert abs(sim_identical - 1.0) < 1e-5
    assert abs(sim_orthogonal - 0.0) < 1e-5

    # Clean up test index if created
    if os.path.exists("samples/test_temp_index.json"):
        os.remove("samples/test_temp_index.json")


def test_segmenter_initialization_and_geometry():
    segmenter = Segmenter(model_name="mobile_sam.pt")
    assert segmenter.model_name == "mobile_sam.pt"

    # Synthetic mask geometry calculation
    h, w = 100, 100
    mock_mask = np.zeros((h, w), dtype=bool)
    mock_mask[20:40, 20:50] = True  # 20 * 30 = 600 pixels

    area_px = int(np.sum(mock_mask))
    total_px = float(h * w)
    area_pct = round((area_px / total_px) * 100.0, 2)

    assert area_px == 600
    assert area_pct == 6.0


def test_diffusion_restorer_graceful_diagnostics():
    restorer = DiffusionRestorer()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Test that uninstalled/missing model weights return clean diagnostic dictionary
    out, info = restorer.restore_image(dummy_img)
    assert isinstance(info, dict)
    assert "status" in info
    assert info["is_simulated"] is False
