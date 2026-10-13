import os
import pytest
import numpy as np
import cv2
from vision.detection import ObjectDetector, DEFAULT_ROAD_CLASSES


def test_detector_initialization_offline():
    """Verify detector initialization and fallback class map without downloading weights."""
    detector = ObjectDetector(model_name="non_existent_weights.pt", lazy_load=True)
    # Lazy initialization should not crash or load immediately
    classes = detector.get_available_classes()
    assert isinstance(classes, dict)
    assert 0 in classes
    assert classes[0] == "person"
    assert classes[2] == "car"


def test_detector_mocked_inference_and_annotations():
    """Test detection formatting, relative area computation, and annotation rendering with mocked inference."""
    detector = ObjectDetector(model_name="yolov8n.pt", lazy_load=True)

    # Mock YOLO model result
    class MockBox:
        def __init__(self, cls_id, conf, xyxy):
            import torch
            self.cls = torch.tensor([cls_id])
            self.conf = torch.tensor([conf])
            self.xyxy = torch.tensor([xyxy])

    class MockResult:
        def __init__(self, boxes):
            self.boxes = boxes

    class MockYOLO:
        names = DEFAULT_ROAD_CLASSES
        def predict(self, source, conf=0.25, iou=0.45, verbose=False):
            # 2 mocked boxes:
            # Box 1: Car at lower viewport with large area [100, 600, 400, 900] in a 1000x1000 image
            # Box 2: Person at upper viewport with small area [50, 50, 80, 100]
            b1 = MockBox(2, 0.88, [100, 600, 400, 900])
            b2 = MockBox(0, 0.65, [50, 50, 80, 100])
            return [MockResult([b1, b2])]
        def __call__(self, *args, **kwargs):
            return self.predict(*args, **kwargs)

    detector.model = MockYOLO()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)

    dets = detector.detect(frame, conf_threshold=0.25)
    assert len(dets) == 2

    # Check first detection (Car)
    assert dets[0]["class_name"] == "car"
    assert dets[0]["confidence"] == 0.88
    assert dets[0]["box"] == [100, 600, 400, 900]
    assert dets[0]["relative_area_pct"] == 9.0  # 300 * 300 / 1000000 = 9%
    assert dets[0]["proximity_heuristic"] == "Immediate Proximity (Visual Heuristic)"

    # Check second detection (Person)
    assert dets[1]["class_name"] == "person"
    assert dets[1]["confidence"] == 0.65
    assert dets[1]["relative_area_pct"] == 0.15  # 30 * 50 / 1000000 = 0.15%
    assert dets[1]["proximity_heuristic"] == "Far / Distant (Visual Heuristic)"

    # Check annotation rendering
    annotated = detector.draw_annotations(frame, dets, show_proximity=True)
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8


@pytest.mark.integration
def test_detector_real_inference_integration():
    """Real YOLOv8n weight loading and inference (requires yolov8n.pt or internet access)."""
    if not os.path.exists("yolov8n.pt"):
        pytest.skip("yolov8n.pt weight file not present in workspace")

    detector = ObjectDetector(model_name="yolov8n.pt", lazy_load=False)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame, conf_threshold=0.25)
    assert isinstance(detections, list)

