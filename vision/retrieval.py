"""Visual Scene Search module using DINOv2 self-supervised embeddings and Nearest-Neighbor Retrieval.

Extracts normalized visual embeddings and performs cosine/L2 nearest-neighbor search.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import json
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class SceneRetrievalEngine:
    """Extracts DINOv2 feature representations and conducts Nearest-Neighbor search."""

    def __init__(self, model_name: str = "dinov2_vits14", index_path: str = "samples/dino_index.json"):
        self.model_name = model_name
        self.index_path = index_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self._load_error: Optional[str] = None

        # Preprocessing transform required by DINOv2
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Local gallery index: list of { "path": str, "name": str, "embedding": List[float] }
        self.gallery: List[Dict[str, Any]] = []
        self._load_index()

    def load_model(self) -> bool:
        """Lazily load DINOv2 model via torch.hub."""
        if self.model is not None:
            return True
        try:
            # Load lightweight DINOv2 small (ViT-S/14, ~21M params, ~80MB)
            self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._load_error = None
            return True
        except Exception as e:
            self.model = None
            self._load_error = f"Failed to load DINOv2 ({self.model_name}): {str(e)}"
            return False

    @property
    def is_model_loaded(self) -> bool:
        return self.model is not None

    @property
    def error_message(self) -> Optional[str]:
        return self._load_error

    def extract_embedding(self, image_bgr: np.ndarray) -> np.ndarray:
        """Extract a 384-dimensional L2-normalized embedding vector.

        Args:
            image_bgr: BGR uint8 numpy array.

        Returns:
            Normalized 1D numpy vector (float32).
        """
        if not self.load_model():
            raise RuntimeError(f"DINOv2 model unavailable: {self._load_error}")

        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image is empty.")

        # Convert BGR to RGB PIL
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(input_tensor)
            # L2 Normalize
            features = torch.nn.functional.normalize(features, dim=-1)
            emb = features.cpu().numpy().flatten().astype(np.float32)

        return emb

    def _load_index(self) -> None:
        """Load pre-indexed embeddings from JSON if available and validate entries."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                valid_entries = []
                if isinstance(data, list):
                    for item in data:
                        # Validate schema, embedding dimensions, and file existence
                        if (
                            isinstance(item, dict)
                            and "path" in item
                            and "embedding" in item
                            and isinstance(item["embedding"], list)
                            and len(item["embedding"]) == 384
                            and os.path.exists(item["path"])
                        ):
                            valid_entries.append(item)
                self.gallery = valid_entries
            except Exception:
                self.gallery = []
        else:
            self.gallery = []

    def save_index(self) -> None:
        """Save indexed embeddings to disk."""
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.gallery, f, indent=2)

    def rebuild_index(self, directory: str = "samples") -> int:
        """Clear existing index and completely rebuild from target directory."""
        self.gallery = []
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
            except OSError:
                pass
        return self.build_index_from_directory(directory)

    def add_image_to_index(self, image_path: str) -> bool:
        """Extract embedding and add image to local gallery."""
        if not os.path.exists(image_path):
            return False
        img = cv2.imread(image_path)
        if img is None:
            return False

        # Avoid duplicate paths
        for entry in self.gallery:
            if os.path.abspath(entry["path"]) == os.path.abspath(image_path):
                return True

        emb = self.extract_embedding(img)
        # Validate embedding shape
        if emb.shape[0] != 384 or not np.isfinite(emb).all():
            return False

        self.gallery.append({
            "path": image_path,
            "name": os.path.basename(image_path),
            "embedding": emb.tolist()
        })
        self.save_index()
        return True

    def build_index_from_directory(self, directory: str = "samples") -> int:
        """Scan a folder and index all valid image files."""
        if not os.path.exists(directory):
            return 0
        valid_exts = {".jpg", ".jpeg", ".png"}
        indexed_count = 0
        for fname in sorted(os.listdir(directory)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                fpath = os.path.join(directory, fname)
                if self.add_image_to_index(fpath):
                    indexed_count += 1
        return indexed_count

    def search(self, query_bgr: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find the top-k visually similar scenes using Cosine Similarity.

        Returns:
            List of { "path": str, "name": str, "similarity": float }
        """
        if not self.gallery:
            return []

        if query_bgr is None or not isinstance(query_bgr, np.ndarray) or query_bgr.size == 0:
            raise ValueError("Query image must be a non-empty numpy array.")

        query_emb = self.extract_embedding(query_bgr)
        # Ensure query embedding is valid and finite
        if not np.isfinite(query_emb).all():
            return []

        q_norm = np.linalg.norm(query_emb)
        if q_norm > 0:
            query_emb = query_emb / q_norm

        results = []
        for item in self.gallery:
            try:
                target_emb = np.array(item["embedding"], dtype=np.float32)
                if target_emb.shape[0] != 384 or not np.isfinite(target_emb).all():
                    continue

                t_norm = np.linalg.norm(target_emb)
                if t_norm > 0:
                    target_emb = target_emb / t_norm

                sim = float(np.dot(query_emb, target_emb))
                if np.isnan(sim) or np.isinf(sim):
                    continue

                results.append({
                    "path": item["path"],
                    "name": item["name"],
                    "similarity": round(float(np.clip(sim, -1.0, 1.0)), 4)
                })
            except Exception:
                continue

        # Sort descending by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
