"""Real-time object detection module using Ultralytics YOLO models.

Provides bounding box inference, class filtering, and experimental image-space
proximity heuristics without claiming calibrated physical metric distance.
"""

from typing import List, Dict, Any, Optional, Set
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Standard COCO road-relevant classes for ADAS
DEFAULT_ROAD_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign"
}

# Color palette (BGR) for distinct road entities in sleek dark dashboard theme
CLASS_COLORS = {
    "person": (235, 140, 52),     # Amber/Orange
    "bicycle": (50, 205, 50),     # Lime green
    "car": (245, 206, 66),        # Cyan/Yellow
    "motorcycle": (255, 105, 180),# Pink
    "bus": (255, 140, 0),         # Dark orange
    "truck": (218, 112, 214),     # Orchid
    "traffic light": (0, 215, 255),# Gold
    "stop sign": (0, 0, 255)      # Red
}


class ObjectDetector:
    """Wrapper around Ultralytics YOLO for ADAS detection and visualization."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        """Initialize YOLO model.

        Weights are loaded from local disk or automatically downloaded by Ultralytics.
        """
        self.model_name = model_name
        self.model: Optional[YOLO] = None
        self._load_error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = YOLO(self.model_name)
            self._load_error = None
        except Exception as e:
            self.model = None
            self._load_error = f"Failed to load model '{self.model_name}': {str(e)}"

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def error_message(self) -> Optional[str]:
        return self._load_error

    def get_available_classes(self) -> Dict[int, str]:
        """Return class ID to class name mapping."""
        if not self.is_ready or self.model.names is None:
            return DEFAULT_ROAD_CLASSES
        return self.model.names

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        allowed_classes: Optional[Set[int]] = None
    ) -> List[Dict[str, Any]]:
        """Run YOLO inference on image.

        Args:
            image: BGR uint8 numpy array.
            conf_threshold: Minimum confidence score [0.0, 1.0].
            iou_threshold: NMS IoU threshold.
            allowed_classes: Set of class IDs to retain. If None, retain all.

        Returns:
            List of detected object dictionaries:
            {
                "box": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "class_name": str,
                "relative_area_pct": float,
                "proximity_heuristic": str
            }
        """
        if not self.is_ready:
            raise RuntimeError(f"Detector is not initialized. Error: {self._load_error}")

        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        img_area = float(h * w) if (h * w) > 0 else 1.0

        # Run inference via Ultralytics (handles BGR internally)
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        boxes = first_res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if allowed_classes is not None and cls_id not in allowed_classes:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()

            # Relative bounding box area in image space
            box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            relative_area_pct = (box_area / img_area) * 100.0

            # Experimental Image-Space Proximity Heuristic (Explicitly NOT Physical Metres)
            # A larger bbox occupying a significant lower-half viewport implies visual proximity
            if relative_area_pct > 12.0 or (y2 > (h * 0.85) and relative_area_pct > 6.0):
                proximity = "Immediate Proximity (Visual Heuristic)"
            elif relative_area_pct > 3.0:
                proximity = "Mid-Field (Visual Heuristic)"
            else:
                proximity = "Far / Distant (Visual Heuristic)"

            cls_name = self.model.names.get(cls_id, f"Class {cls_id}")

            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "class_id": cls_id,
                "class_name": cls_name,
                "relative_area_pct": round(relative_area_pct, 2),
                "proximity_heuristic": proximity
            })

        return detections

    def draw_annotations(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        show_confidence: bool = True,
        show_proximity: bool = False
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and badges onto an image copy.

        Args:
            image: Original BGR image.
            detections: List of detection dictionaries.
            show_confidence: Include confidence in the label text.
            show_proximity: Include proximity label tag.

        Returns:
            Annotated BGR copy of the image.
        """
        canvas = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            prox = det["proximity_heuristic"]

            color = CLASS_COLORS.get(cls_name, (0, 220, 220))

            # Draw rounded or bordered rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Build label text
            label_parts = [cls_name.upper()]
            if show_confidence:
                label_parts.append(f"{conf:.0%}")
            label = " | ".join(label_parts)

            # Label banner
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            banner_y1 = max(0, y1 - th - 8)
            banner_y2 = y1
            banner_x2 = min(canvas.shape[1], x1 + tw + 10)

            # Draw label background
            cv2.rectangle(canvas, (x1, banner_y1), (banner_x2, banner_y2), color, -1)
            # Text in dark or white depending on contrast
            text_color = (15, 23, 42)  # Dark slate for contrast over vivid labels
            cv2.putText(
                canvas,
                label,
                (x1 + 5, banner_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA
            )

            if show_proximity:
                prox_label = f"[{prox.split()[0]}]"
                cv2.putText(
                    canvas,
                    prox_label,
                    (x1, min(canvas.shape[0] - 5, y2 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA
                )

        return canvas
