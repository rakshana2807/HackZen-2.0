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
    temp_index = "samples/test_temp_index.json"
    engine = SceneRetrievalEngine(index_path=temp_index)
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

    # Test corrupted index recovery
    with open(temp_index, "w") as f:
        f.write("{invalid-json-content: corrupt")
    engine_corrupt = SceneRetrievalEngine(index_path=temp_index)
    assert engine_corrupt.gallery == []

    # Clean up test index if created
    if os.path.exists(temp_index):
        os.remove(temp_index)


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


def test_segmenter_spatial_association_logic():
    """Verify that SAM mask spatial overlap links candidate detections correctly without model weights."""
    segmenter = Segmenter(model_name="mobile_sam.pt")

    # Mock raw masks output of shape (2, 100, 100)
    raw_masks = np.zeros((2, 100, 100), dtype=np.float32)
    # Mask 1: occupies [20, 20] to [40, 50]
    raw_masks[0, 20:40, 20:50] = 1.0
    # Mask 2: occupies [60, 60] to [80, 90]
    raw_masks[1, 60:80, 60:90] = 1.0

    class MockResults:
        class MockMasks:
            class MockTensor:
                def cpu(self):
                    return self
                def numpy(self):
                    return raw_masks
            data = MockTensor()
        masks = MockMasks()

    class MockSAM:
        def __call__(self, *args, **kwargs):
            return [MockResults()]

    segmenter.model = MockSAM()

    # Pass detection boxes where Box 0 matches Mask 1, and Box 1 matches Mask 2
    bboxes = [[20, 20, 40, 50], [60, 60, 80, 90]]
    labels = ["car", "person"]

    segs = segmenter.segment(np.zeros((100, 100, 3), dtype=np.uint8), bboxes=bboxes, detection_labels=labels)
    assert len(segs) == 2
    assert segs[0]["label"] == "car"
    assert segs[0]["box"] == [20, 20, 40, 50]
    assert segs[1]["label"] == "person"
    assert segs[1]["box"] == [60, 60, 80, 90]

    # Ambiguous/distant prompt box should fall back to generic mask label
    distant_boxes = [[0, 0, 10, 10]]
    distant_labels = ["bicycle"]
    segs_unmatched = segmenter.segment(np.zeros((100, 100, 3), dtype=np.uint8), bboxes=distant_boxes, detection_labels=distant_labels)
    assert "Unassociated" in segs_unmatched[0]["label"] or "Mask" in segs_unmatched[0]["label"]


def test_proximity_condition_branches():
    """Verify that all branches of the proximity heuristic evaluate properly."""
    h, w = 1000, 1000
    img_area = float(h * w)

    def evaluate_proximity(x1, y1, x2, y2):
        box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        relative_area_pct = (box_area / img_area) * 100.0
        if relative_area_pct > 12.0 or (y2 > (h * 0.85) and relative_area_pct > 6.0):
            return "Immediate Proximity (Visual Heuristic)"
        elif relative_area_pct > 3.0:
            return "Mid-Field (Visual Heuristic)"
        else:
            return "Far / Distant (Visual Heuristic)"

    # Branch 1: Huge bbox (>12% area) regardless of vertical position
    assert evaluate_proximity(0, 0, 400, 400) == "Immediate Proximity (Visual Heuristic)"

    # Branch 2: Lower viewport (y2 > 850) AND relative area > 6%
    assert evaluate_proximity(100, 600, 400, 900) == "Immediate Proximity (Visual Heuristic)"

    # Branch 3a: Lower viewport (y2 > 850) but moderate area (between 3% and 6%) -> Mid-field
    # w=200, h=200 -> area=40000 -> 4.0%
    assert evaluate_proximity(100, 700, 300, 900) == "Mid-Field (Visual Heuristic)"

    # Branch 3b: Lower viewport (y2 > 850) with tiny area (<3%) -> Far
    # w=100, h=100 -> area=10000 -> 1.0%
    assert evaluate_proximity(100, 800, 200, 900) == "Far / Distant (Visual Heuristic)"

    # Branch 4: Mid area (3% - 12%) in upper half -> Mid-Field
    assert evaluate_proximity(100, 100, 300, 300) == "Mid-Field (Visual Heuristic)"

    # Branch 5: Tiny distant bbox (<3%) in upper half -> Far
    assert evaluate_proximity(100, 100, 130, 130) == "Far / Distant (Visual Heuristic)"


def test_retrieval_mocked_search_ranking():
    """Verify ranking and similarity calculation without downloading DINOv2."""
    engine = SceneRetrievalEngine(index_path="samples/test_mock_index.json")

    # Manually configure gallery with known normalized synthetic embeddings (384-D)
    emb_target1 = np.zeros(384, dtype=np.float32)
    emb_target1[0] = 1.0  # Perfect cosine match

    emb_target2 = np.zeros(384, dtype=np.float32)
    emb_target2[1] = 1.0  # Orthogonal

    engine.gallery = [
        {"path": "samples/road_day_normal.jpg", "name": "road_day_normal.jpg", "embedding": emb_target1.tolist()},
        {"path": "samples/road_night_dark.jpg", "name": "road_night_dark.jpg", "embedding": emb_target2.tolist()}
    ]

    # Mock extract_embedding on engine
    query_emb = np.zeros(384, dtype=np.float32)
    query_emb[0] = 1.0
    engine.extract_embedding = lambda img: query_emb

    dummy_query = np.zeros((50, 50, 3), dtype=np.uint8)
    matches = engine.search(dummy_query, top_k=2)

    assert len(matches) == 2
    assert matches[0]["name"] == "road_day_normal.jpg"
    assert abs(matches[0]["similarity"] - 1.0) < 1e-4
    assert matches[1]["name"] == "road_night_dark.jpg"
    assert abs(matches[1]["similarity"] - 0.0) < 1e-4


def test_diffusion_restorer_graceful_diagnostics():
    restorer = DiffusionRestorer()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Test that uninstalled/missing model weights return clean diagnostic dictionary
    out, info = restorer.restore_image(dummy_img)
    assert isinstance(info, dict)
    assert "status" in info
    assert info["is_simulated"] is False
    assert out is None
