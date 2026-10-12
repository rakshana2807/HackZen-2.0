import pytest
import numpy as np
import cv2
from vision.detection import ObjectDetector


def test_detector_initialization():
    detector = ObjectDetector(model_name="yolov8n.pt")
    assert detector.is_ready is True
    assert detector.error_message is None
    classes = detector.get_available_classes()
    assert len(classes) > 0
    assert 0 in classes  # 'person' in COCO


def test_detector_inference_on_synthetic_frame():
    detector = ObjectDetector(model_name="yolov8n.pt")
    # Clean blank frame
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame, conf_threshold=0.25)
    assert isinstance(detections, list)

    # Annotate function test
    annotated = detector.draw_annotations(frame, detections, show_proximity=True)
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
