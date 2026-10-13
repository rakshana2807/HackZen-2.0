"""Tracking and traffic analysis module using YOLOv8 and ByteTrack.

Provides persistent multi-object tracking, movement trails, and traffic flow counts.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO

from .detection import CLASS_COLORS, DEFAULT_ROAD_CLASSES


class ObjectTracker:
    """Wrapper around YOLOv8 tracking with ByteTrack integration."""

    def __init__(self, model_name: str = "yolov8n.pt", max_trail_len: int = 25, lazy_load: bool = True):
        self.model_name = model_name
        self.model: Optional[YOLO] = None
        self._load_error: Optional[str] = None
        self.max_trail_len = max_trail_len
        # track_id -> deque of (x_center, y_center)
        self.trails: Dict[int, deque] = {}
        # Cumulative unique track IDs seen: class_name -> set(track_id)
        self.cumulative_counts: Dict[str, Set[int]] = {}
        if not lazy_load:
            self._load_model()

    def _load_model(self) -> bool:
        if self.model is not None:
            return True
        try:
            self.model = YOLO(self.model_name)
            self._load_error = None
            return True
        except Exception as e:
            self.model = None
            self._load_error = f"Failed to initialize tracker with {self.model_name}: {str(e)}"
            return False

    @property
    def is_ready(self) -> bool:
        if self.model is not None:
            return True
        return os.path.exists(self.model_name)

    @property
    def error_message(self) -> Optional[str]:
        return self._load_error

    def reset_tracks(self) -> None:
        """Reset historical trails and cumulative counts."""
        self.trails.clear()
        self.cumulative_counts.clear()

    def track_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        allowed_classes: Optional[Set[int]] = None
    ) -> List[Dict[str, Any]]:
        """Run ByteTrack persistence tracking on a single frame.

        Args:
            frame: BGR uint8 image array.
            conf_threshold: Minimum detection confidence.
            allowed_classes: Set of class IDs to track.

        Returns:
            List of track dictionaries:
            {
                "track_id": int,
                "box": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "class_name": str,
                "trail": List[Tuple[int, int]]
            }
        """
        if not self._load_model():
            raise RuntimeError(f"Tracker is not initialized. Error: {self._load_error}")

        if frame is None or frame.size == 0:
            return []

        # Run YOLO with persistent ByteTrack tracking
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            verbose=False
        )

        tracked_objects: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return tracked_objects

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return tracked_objects

        active_track_ids = set()

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if allowed_classes is not None and cls_id not in allowed_classes:
                continue

            # Check if tracker assigned a persistent ID
            if box.id is not None:
                track_id = int(box.id[0].item())
            else:
                track_id = -1  # Unassigned/untracked detection

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            cls_name = self.model.names.get(cls_id, f"Class {cls_id}")

            if track_id > 0:
                active_track_ids.add(track_id)
                if track_id not in self.trails:
                    self.trails[track_id] = deque(maxlen=self.max_trail_len)
                self.trails[track_id].append((center_x, center_y))

                if cls_name not in self.cumulative_counts:
                    self.cumulative_counts[cls_name] = set()
                self.cumulative_counts[cls_name].add(track_id)

                trail_pts = list(self.trails[track_id])
            else:
                trail_pts = [(center_x, center_y)]

            tracked_objects.append({
                "track_id": track_id,
                "box": [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "class_id": cls_id,
                "class_name": cls_name,
                "trail": trail_pts
            })

        # Evict old trails that haven't been seen for 60 frames to save memory
        if len(self.trails) > 200:
            for old_id in list(self.trails.keys()):
                if old_id not in active_track_ids:
                    del self.trails[old_id]

        return tracked_objects

    def draw_tracking_annotations(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        draw_trails: bool = True
    ) -> np.ndarray:
        """Render bounding boxes, persistent IDs, and motion trail overlays."""
        canvas = frame.copy()

        for t in tracks:
            x1, y1, x2, y2 = t["box"]
            cls_name = t["class_name"]
            conf = t["confidence"]
            track_id = t["track_id"]
            color = CLASS_COLORS.get(cls_name, (0, 220, 220))

            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label banner
            id_str = f"ID: {track_id} | " if track_id > 0 else ""
            label = f"{id_str}{cls_name.upper()} {conf:.0%}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            banner_y1 = max(0, y1 - th - 6)
            banner_y2 = y1
            banner_x2 = min(canvas.shape[1], x1 + tw + 8)

            cv2.rectangle(canvas, (x1, banner_y1), (banner_x2, banner_y2), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 4, banner_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (15, 23, 42),
                1,
                cv2.LINE_AA
            )

            # Draw motion trail
            if draw_trails and len(t["trail"]) > 1:
                pts = t["trail"]
                for i in range(1, len(pts)):
                    thickness = int(np.sqrt(float(i + 1)) * 1.5)
                    cv2.line(canvas, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        return canvas
