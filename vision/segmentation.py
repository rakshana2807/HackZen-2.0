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
        if res.masks is None:
            return segmentations
        try:
            if len(res.masks) == 0:
                return segmentations
        except TypeError:
            pass

        raw_masks = res.masks.data.cpu().numpy()  # shape (N, H, W)
        if len(raw_masks) == 0:
            return segmentations

        # Extract binary masks and compute bounding boxes for each mask
        mask_entries = []
        for i, mask_2d in enumerate(raw_masks):
            if mask_2d.shape[:2] != (h, w):
                mask_bool = cv2.resize(mask_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask_bool = mask_2d > 0.5

            area_px = int(np.sum(mask_bool))
            area_pct = round((area_px / total_pixels) * 100.0, 2)

            y_idx, x_idx = np.where(mask_bool)
            if len(x_idx) > 0 and len(y_idx) > 0:
                mbound = [int(np.min(x_idx)), int(np.min(y_idx)), int(np.max(x_idx)), int(np.max(y_idx))]
            else:
                mbound = None

            mask_entries.append({
                "mask_id": i + 1,
                "mask": mask_bool,
                "area_px": area_px,
                "area_pct": area_pct,
                "mbound": mbound
            })

        # Calculate spatial overlap matrix between all masks and all detection boxes
        # Do NOT assume i-th mask corresponds to i-th detection
        matched_labels = {}
        matched_boxes = {}

        if bboxes and detection_labels and len(bboxes) > 0 and len(detection_labels) > 0:
            assigned_det_indices = set()

            for m_idx, m_ent in enumerate(mask_entries):
                mbound = m_ent["mbound"]
                if mbound is None or m_ent["area_px"] == 0:
                    continue

                mx1, my1, mx2, my2 = mbound
                mask_box_area = max(1, (mx2 - mx1) * (my2 - my1))

                best_iou = 0.0
                best_det_idx = -1

                for d_idx, pbox in enumerate(bboxes):
                    if d_idx in assigned_det_indices:
                        continue

                    px1, py1, px2, py2 = pbox
                    ix1 = max(mx1, px1)
                    iy1 = max(my1, py1)
                    ix2 = min(mx2, px2)
                    iy2 = min(my2, py2)

                    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter_area == 0:
                        continue

                    prompt_box_area = max(1, (px2 - px1) * (py2 - py1))
                    iou = inter_area / float(mask_box_area + prompt_box_area - inter_area)
                    overlap_ratio = inter_area / mask_box_area

                    # Find best match above documented threshold (IoU > 0.25 or overlap > 0.5)
                    score = max(iou, overlap_ratio * 0.8)
                    if score > best_iou:
                        best_iou = score
                        best_det_idx = d_idx

                if best_iou > 0.25 and best_det_idx >= 0 and best_det_idx < len(detection_labels):
                    matched_labels[m_idx] = detection_labels[best_det_idx]
                    matched_boxes[m_idx] = bboxes[best_det_idx]
                    assigned_det_indices.add(best_det_idx)

        segmentations: List[Dict[str, Any]] = []
        for m_idx, m_ent in enumerate(mask_entries):
            if m_ent["mbound"] is None or m_ent["area_px"] == 0:
                label = f"Mask #{m_ent['mask_id']} (Empty)"
            elif m_idx in matched_labels:
                label = matched_labels[m_idx]
            else:
                label = f"Mask #{m_ent['mask_id']}"

            segmentations.append({
                "mask_id": m_ent["mask_id"],
                "label": label,
                "mask": m_ent["mask"],
                "area_px": m_ent["area_px"],
                "area_pct": m_ent["area_pct"],
                "box": matched_boxes.get(m_idx, None)
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
