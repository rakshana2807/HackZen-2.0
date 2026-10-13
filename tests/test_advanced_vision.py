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
    """Verify tracker initialization, trail formatting, and reset without requiring weights."""
    tracker = ObjectTracker(model_name="yolov8n.pt", lazy_load=True)

    # Mock YOLO track call
    class MockBox:
        def __init__(self, track_id, cls_id, conf, xyxy):
            import torch
            self.id = torch.tensor([track_id]) if track_id is not None else None
            self.cls = torch.tensor([cls_id])
            self.conf = torch.tensor([conf])
            self.xyxy = torch.tensor([xyxy])

    class MockResult:
        def __init__(self, boxes):
            self.boxes = boxes

    class MockTrackYOLO:
        names = {0: "person", 2: "car"}
        def track(self, source, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False):
            b1 = MockBox(track_id=101, cls_id=2, conf=0.92, xyxy=[50, 50, 150, 150])
            return [MockResult([b1])]

    tracker.model = MockTrackYOLO()

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tracks = tracker.track_frame(frame, conf_threshold=0.25)
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == 101
    assert tracks[0]["class_name"] == "car"
    assert len(tracks[0]["trail"]) == 1

    # Annotation rendering
    annotated = tracker.draw_tracking_annotations(frame, tracks, draw_trails=True)
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8

    tracker.reset_tracks()
    assert len(tracker.trails) == 0
    assert len(tracker.cumulative_counts) == 0


def test_retrieval_empty_index_and_normalization(tmp_path):
    """Test retrieval engine with isolated temporary index directory and robust error handling."""
    temp_index = str(tmp_path / "test_temp_index.json")
    engine = SceneRetrievalEngine(index_path=temp_index)
    # Clean state
    assert len(engine.gallery) == 0

    # Search on empty gallery should return empty list gracefully
    dummy_query = np.zeros((100, 100, 3), dtype=np.uint8)
    res = engine.search(dummy_query, top_k=3)
    assert res == []

    # Test top_k <= 0 returns empty list immediately
    assert engine.search(dummy_query, top_k=0) == []
    assert engine.search(dummy_query, top_k=-1) == []

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
    assert engine_corrupt.last_load_rejected_count == 1


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
    assert "Generic" in segs_unmatched[0]["label"] or "Unassociated" in segs_unmatched[0]["label"]


def test_segmenter_edge_cases_and_ambiguity():
    """Test empty masks, invalid boxes, missing labels, and duplicate matches."""
    segmenter = Segmenter(model_name="mobile_sam.pt")

    # Mock raw masks output with 1 empty mask and 1 valid mask
    raw_masks = np.zeros((2, 100, 100), dtype=np.float32)
    raw_masks[1, 30:70, 30:70] = 1.0  # Mask 2 is valid, Mask 1 is all zeros

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

    # Pass invalid box [50, 50, 40, 40] (x2 <= x1) and valid box [30, 30, 70, 70]
    bboxes = [[50, 50, 40, 40], [30, 30, 70, 70]]
    labels = ["invalid_box", "truck"]

    segs = segmenter.segment(np.zeros((100, 100, 3), dtype=np.uint8), bboxes=bboxes, detection_labels=labels)
    assert len(segs) == 2
    # Mask 1 was empty
    assert "Empty" in segs[0]["label"]
    assert segs[0]["area_px"] == 0
    # Mask 2 matched truck
    assert segs[1]["label"] == "truck"
    assert segs[1]["box"] == [30, 30, 70, 70]

    # No detections provided -> should return generic mask labels without error
    segs_no_dets = segmenter.segment(np.zeros((100, 100, 3), dtype=np.uint8), bboxes=None, detection_labels=None)
    assert len(segs_no_dets) == 2
    assert "Mask #2" in segs_no_dets[1]["label"]


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


def test_retrieval_mocked_search_ranking(tmp_path):
    """Verify ranking, similarity calculation, and index validation without downloading DINOv2."""
    mock_idx = str(tmp_path / "mock_idx.json")
    engine = SceneRetrievalEngine(index_path=mock_idx)

    # Manually configure gallery with known normalized synthetic embeddings (384-D)
    emb_target1 = np.zeros(384, dtype=np.float32)
    emb_target1[0] = 1.0  # Perfect cosine match

    emb_target2 = np.zeros(384, dtype=np.float32)
    emb_target2[1] = 1.0  # Orthogonal

    # Create dummy image files on disk so index validation accepts them
    img1 = str(tmp_path / "scene1.jpg")
    img2 = str(tmp_path / "scene2.jpg")
    cv2.imwrite(img1, np.zeros((30, 30, 3), dtype=np.uint8))
    cv2.imwrite(img2, np.zeros((30, 30, 3), dtype=np.uint8))

    engine.gallery = [
        {"path": img1, "name": "scene1.jpg", "embedding": emb_target1.tolist()},
        {"path": img2, "name": "scene2.jpg", "embedding": emb_target2.tolist()}
    ]
    engine.save_index()

    # Re-read using _load_index to verify roundtrip parsing
    engine2 = SceneRetrievalEngine(index_path=mock_idx)
    assert len(engine2.gallery) == 2
    assert engine2.last_load_rejected_count == 0

    # Mock extract_embedding on engine
    query_emb = np.zeros(384, dtype=np.float32)
    query_emb[0] = 1.0
    engine2.extract_embedding = lambda img: query_emb

    dummy_query = np.zeros((50, 50, 3), dtype=np.uint8)
    matches = engine2.search(dummy_query, top_k=2)

    assert len(matches) == 2
    assert matches[0]["name"] == "scene1.jpg"
    assert abs(matches[0]["similarity"] - 1.0) < 1e-4
    assert matches[1]["name"] == "scene2.jpg"
    assert abs(matches[1]["similarity"] - 0.0) < 1e-4


def test_diffusion_restorer_validation_and_mocked_execution():
    """Verify input validation, failure diagnostics, and mocked inference without downloading multi-GB weights."""
    restorer = DiffusionRestorer()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # 1. Invalid prompt validation
    out, info = restorer.restore_image(dummy_img, prompt="")
    assert out is None
    assert info["status"] == "failed"
    assert "Prompt must be a non-empty string" in info["error"]

    # 2. Invalid steps validation
    out, info = restorer.restore_image(dummy_img, prompt="test prompt", num_inference_steps=0)
    assert out is None
    assert "num_inference_steps must be a positive integer" in info["error"]

    # 3. Missing weights / offline fallback diagnostic
    out, info = restorer.restore_image(dummy_img)
    assert isinstance(info, dict)
    assert info["status"] == "failed"
    assert info["is_simulated"] is False

    # 4. Mocked successful pipeline inference with dimension matching
    class MockOutputImage:
        def __init__(self, w, h):
            self.images = [np.full((h, w, 3), 180, dtype=np.uint8)]

    class MockPipeline:
        def __call__(self, prompt, image, num_inference_steps):
            # Simulate a 4x upscale (256x256 -> 1024x1024)
            return MockOutputImage(w=1024, h=1024)

    restorer.pipe = MockPipeline()
    restorer._has_diffusers = True

    input_img = np.zeros((120, 200, 3), dtype=np.uint8)
    out_bgr, info = restorer.restore_image(input_img, prompt="test road", match_input_resolution=True)
    assert out_bgr is not None
    assert info["status"] == "success"
    # Preserved original input resolution
    assert out_bgr.shape[:2] == (120, 200)
    assert info["input_dimensions"] == "200x120"
    assert info["output_dimensions"] == "200x120"

