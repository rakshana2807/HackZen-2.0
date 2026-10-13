"""Object Segmentation & Analysis module using Segment Anything Model (SAM).

Provides instance mask generation, YOLO detection prompt integration,
pixel area computation, and overlay rendering.
"""

from typing import List, Dict, Any, Optional
import os
import cv2
import numpy as np
import torch
from ultralytics import SAM


# High-contrast color palette for segmentation masks
MASK_COLORS = [
    (56, 189, 248),   # Sky blue
    (52, 211, 153),   # Emerald green
    (251, 146, 60),   # Bright orange
    (244, 114, 182),  # Rose pink
    (167, 139, 250),  # Purple
    (250, 204, 21),   # Yellow
    (248, 113, 113),  # Red
    (45, 212, 191),   # Teal
]


class Segmenter:
    """Wrapper around Segment Anything Model for ADAS mask generation and analysis."""

    def __init__(self, model_name: str = "mobile_sam.pt"):
        """Initialize SAM.

        MobileSAM (~39MB) is lightweight and CPU-friendly, natively supported
        by Ultralytics without requiring massive GPU VRAM.
        """
        self.model_name = model_name
        self.model: Optional[SAM] = None
        self._load_error: Optional[str] = None

    def load_model(self) -> bool:
        """Lazily load MobileSAM model."""
        if self.model is not None:
            return True
        try:
            self.model = SAM(self.model_name)
            self._load_error = None
            return True
        except Exception as e:
            self.model = None
            self._load_error = f"Failed to load SAM ({self.model_name}): {str(e)}"
            return False

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def error_message(self) -> Optional[str]:
        return self._load_error

    def segment(
        self,
        image_bgr: np.ndarray,
        bboxes: Optional[List[List[int]]] = None,
        detection_labels: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate segmentation masks.

        Args:
            image_bgr: Input BGR image.
            bboxes: Optional bounding box prompts [x1, y1, x2, y2] from detector.
            detection_labels: Optional class names corresponding to bboxes.

        Returns:
            List of mask dictionaries:
            {
                "mask_id": int,
                "label": str,
                "mask": 2D boolean np.ndarray,
                "area_px": int,
                "area_pct": float,
                "box": Optional[List[int]]
            }
        """
        if not self.load_model():
            raise RuntimeError(f"SAM is not loaded. Error: {self._load_error}")

        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        total_pixels = float(h * w) if (h * w) > 0 else 1.0

        # Predict using SAM
        if bboxes and len(bboxes) > 0:
            # Box prompt mode (prompted by YOLO bounding boxes)
            results = self.model(image_bgr, bboxes=bboxes, verbose=False)
        else:
            # Automatic segmentation mode
            results = self.model(image_bgr, verbose=False)

        segmentations: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return segmentations

        res = results[0]
        if res.masks is None or len(res.masks) == 0:
            return segmentations

        raw_masks = res.masks.data.cpu().numpy()  # shape (N, H, W)

        for i, mask_2d in enumerate(raw_masks):
            # Resize binary mask to image shape if necessary
            if mask_2d.shape[:2] != (h, w):
                mask_bool = cv2.resize(mask_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask_bool = mask_2d > 0.5

            area_px = int(np.sum(mask_bool))
            area_pct = round((area_px / total_pixels) * 100.0, 2)

            # Associate label if provided
            if detection_labels and i < len(detection_labels):
                label = detection_labels[i]
            else:
                label = f"Mask #{i+1}"

            box = bboxes[i] if (bboxes and i < len(bboxes)) else None

            segmentations.append({
                "mask_id": i + 1,
                "label": label,
                "mask": mask_bool,
                "area_px": area_px,
                "area_pct": area_pct,
                "box": box
            })

        return segmentations

    def draw_mask_overlays(
        self,
        image_bgr: np.ndarray,
        segmentations: List[Dict[str, Any]],
        alpha: float = 0.45
    ) -> np.ndarray:
        """Render distinct translucent color masks and contour boundaries on the image."""
        overlay = image_bgr.copy()
        canvas = image_bgr.copy()

        for seg in segmentations:
            mask = seg["mask"]
            mask_id = seg["mask_id"]
            color = MASK_COLORS[(mask_id - 1) % len(MASK_COLORS)]

            # Fill mask area on overlay
            overlay[mask] = color

            # Draw outer contour for crisp edges
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)

        # Blend translucent overlay
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        # Draw labels and area badges at centroids
        for seg in segmentations:
            mask = seg["mask"]
            label = f"{seg['label']} ({seg['area_pct']}%)"
            y_indices, x_indices = np.where(mask)
            if len(x_indices) > 0 and len(y_indices) > 0:
                cx = int(np.mean(x_indices))
                cy = int(np.mean(y_indices))
                cv2.putText(
                    canvas,
                    label,
                    (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    canvas,
                    label,
                    (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (15, 23, 42),
                    1,
                    cv2.LINE_AA
                )

        return canvas
