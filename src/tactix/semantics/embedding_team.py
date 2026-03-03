"""
Project: Tactix
File Created: 2026-03-01
Author: Xingnan Zhu
File Name: embedding_team.py
Description:
    Alternative team classifier using SigLIP vision embeddings + UMAP + KMeans.
    More robust than raw color KMeans for similar jersey colors, shadows, and
    lighting variation. Requires optional dependencies: transformers, umap-learn, torch.

    Usage:
        uv sync --extra embedding
        # Then set USE_EMBEDDING_CLASSIFIER = True in config.py
"""

from typing import List, Optional, Tuple

import numpy as np
import cv2
from sklearn.cluster import KMeans

from tactix.core.types import Player


class EmbeddingTeamClassifier:
    """
    Team classifier that uses SigLIP vision transformer embeddings.
    Crops player bounding boxes → SigLIP embedding → UMAP 3D → KMeans(n=2).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.kmeans: Optional[KMeans] = None
        self.team_colors = {}
        self._model = None
        self._processor = None
        self._reducer = None
        self._fitted = False

        self._load_model()

    def _load_model(self) -> None:
        """Lazily load SigLIP model and processor."""
        try:
            import torch
            from transformers import SiglipImageProcessor, SiglipVisionModel

            model_name = "google/siglip-base-patch16-224"
            print(f"🧠 Loading SigLIP model: {model_name}...")
            # Use SiglipImageProcessor (not SiglipProcessor) — image-only, avoids tokenizer bug in transformers 5.x
            self._processor = SiglipImageProcessor.from_pretrained(model_name)
            self._model = SiglipVisionModel.from_pretrained(model_name)

            # Move to device
            if self.device == "mps" and torch.backends.mps.is_available():
                self._model = self._model.to("mps")
            elif self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            else:
                self._model = self._model.to("cpu")

            self._model.eval()
            print("🧠 SigLIP model loaded successfully")
        except ImportError:
            raise ImportError(
                "EmbeddingTeamClassifier requires: transformers, torch. "
                "Install with: uv sync --extra embedding"
            )

    def _extract_crop(self, frame: np.ndarray, rect: Tuple[float, float, float, float], mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Extract player crop for SigLIP input. When mask is provided, background pixels are zeroed."""
        x1, y1, x2, y2 = map(int, rect)
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        if mask is not None:
            # Slice the full-frame mask to the bbox region and zero out background pixels.
            mask_2d = mask.squeeze() if mask.ndim == 3 else mask
            mask_crop = mask_2d[y1:y2, x1:x2]
            if mask_crop.shape[:2] == crop.shape[:2]:
                crop = crop * np.expand_dims(mask_crop.astype(np.uint8), axis=-1)
        # Convert BGR to RGB for SigLIP
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop_rgb

    def _extract_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract SigLIP embeddings from a batch of RGB crops."""
        import torch
        from PIL import Image

        images = [Image.fromarray(c) for c in crops]
        inputs = self._processor(images=images, return_tensors="pt")

        # Move inputs to model device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # SiglipVisionModel returns last_hidden_state; mean-pool across patches
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return embeddings

    def fit_from_crops(self, crops: List[np.ndarray]) -> bool:
        """
        Train the classifier from pre-collected player crops.
        Extracts embeddings → UMAP → KMeans.
        Returns True if training succeeded.
        """
        if len(crops) < 4:
            return False

        try:
            import umap
        except ImportError:
            raise ImportError(
                "EmbeddingTeamClassifier requires umap-learn. "
                "Install with: uv sync --extra embedding"
            )

        print(f"🧠 Extracting SigLIP embeddings for {len(crops)} crops...")

        # Batch extract embeddings
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i + batch_size]
            emb = self._extract_embeddings(batch)
            all_embeddings.append(emb)
        embeddings = np.vstack(all_embeddings)

        # UMAP reduce to 3D
        print("🧠 Running UMAP dimensionality reduction...")
        self._reducer = umap.UMAP(n_components=3, random_state=42)
        reduced = self._reducer.fit_transform(embeddings)

        # KMeans cluster
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        self.kmeans.fit(reduced)
        self._train_labels = self.kmeans.labels_.copy()

        self._fitted = True
        print(f"🧠 Embedding classifier trained ({len(crops)} samples)")
        return True

    def get_team_color_means(
        self, colors: List[np.ndarray]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return mean shirt color per SigLIP cluster.
        Uses the cluster labels from fit_from_crops — requires that `colors` is
        the same length and same order as the `crops` passed to fit_from_crops.
        Returns (color_cluster_0, color_cluster_1), or None on failure.
        """
        if not self._fitted or self._train_labels is None:
            return None
        if len(colors) != len(self._train_labels):
            return None

        colors_arr = np.array(colors)
        mask_0 = self._train_labels == 0
        mask_1 = self._train_labels == 1
        if not np.any(mask_0) or not np.any(mask_1):
            return None

        return colors_arr[mask_0].mean(axis=0), colors_arr[mask_1].mean(axis=0)

    def unload(self) -> None:
        """Release model weights from memory after prescan training is complete."""
        self._model = None
        self._processor = None
