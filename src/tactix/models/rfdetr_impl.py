"""
Project: Tactix
File Created: 2026-03-03
File Name: rfdetr_impl.py
Description:
    RF-DETR detector implementation for Tactix.
    Fine-tuned RF-DETR checkpoint is loaded via the rfdetr package.

    IMPORTANT — rfdetr requires transformers<5.0, which conflicts with the main
    project's transformers>=5.0 dependency (needed by SigLIP/SAM3).
    To run with RF-DETR, disable USE_EMBEDDING_CLASSIFIER in config.py and
    run via the isolated overlay:

        uv run --with rfdetr --with "transformers<5.0" python run.py

    Alternatively, if you only need RF-DETR inference (no SigLIP prescan),
    you can install rfdetr into the main env at the cost of downgrading
    transformers:

        uv add rfdetr
"""

import numpy as np
from PIL import Image

from tactix.core.types import Player, Ball, FrameData, TeamID
from tactix.models.interface import BaseDetector

# Class IDs emitted by our fine-tuned RF-DETR model.
# Must match the CLASS_NAMES order used in train_rfdetr.py:
#   ["ball", "goalkeeper", "player", "referee"]
_CLASS_MAP = {
    0: "ball",
    1: "goalkeeper",
    2: "player",
    3: "referee",
}


class RFDETRDetector(BaseDetector):
    """
    RF-DETR inference wrapper that mirrors the interface of YOLODetector.
    Applies the same ball sanity-filtering (area + aspect-ratio + double-standard
    confidence threshold) so downstream logic is unaffected.
    """

    supports_masks = False

    def __init__(
        self,
        model_weights: str,
        model_size: str = "large",
        device: str = "mps",
        conf_threshold: float = 0.3,
    ) -> None:
        try:
            from rfdetr import RFDETRBase, RFDETRLarge
        except ImportError as exc:
            raise ImportError(
                "rfdetr is not installed in this environment.\n"
                "Run with:\n"
                "  uv run --with rfdetr --with 'transformers<5.0' python run.py\n"
                "Or install rfdetr (will downgrade transformers):\n"
                "  uv add rfdetr"
            ) from exc

        print(f"🔍 Loading RF-DETR ({model_size}) from {model_weights} on {device}...")
        ModelClass = RFDETRLarge if model_size == "large" else RFDETRBase
        self.model = ModelClass(pretrain_weights=model_weights)
        self.device = device
        self.conf_threshold = conf_threshold

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, frame_index: int) -> FrameData:
        """Run RF-DETR inference on a BGR frame and return FrameData."""
        # RF-DETR expects an RGB PIL Image.
        img_rgb = Image.fromarray(frame[:, :, ::-1])

        # predict() returns a supervision-compatible Detections object.
        detections = self.model.predict(img_rgb, threshold=self.conf_threshold)

        frame_data = FrameData(frame_index=frame_index, image_shape=frame.shape[:2])
        ball_candidates: list = []
        player_boxes: list = []

        # --- First pass: classify every detection ---
        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            confidence = float(detections.confidence[i])
            class_id = int(detections.class_id[i])
            class_name = _CLASS_MAP.get(class_id, "unknown")
            rect = tuple(xyxy.tolist())

            x1, y1, x2, y2 = xyxy
            width = x2 - x1
            height = y2 - y1
            area = width * height
            ratio = width / height if height > 0 else 0.0

            # Correction: tiny square object misclassified → force ball
            if class_name != "ball" and area < 900 and ratio > 0.7:
                class_name = "ball"

            # Sanity check: objects too large or too elongated cannot be ball
            if class_name == "ball":
                if area > 900 or ratio < 0.6 or ratio > 1.5:
                    continue

            if class_name == "ball":
                ball_candidates.append((rect, confidence))
            elif class_name in ("player", "goalkeeper", "referee"):
                player = Player(
                    id=-1,
                    rect=rect,
                    class_id=class_id,
                    confidence=confidence,
                    team=TeamID.UNKNOWN,
                )
                if class_name == "referee":
                    player.team = TeamID.REFEREE
                elif class_name == "goalkeeper":
                    player.team = TeamID.GOALKEEPER
                frame_data.players.append(player)
                player_boxes.append(xyxy)

        # --- Second pass: "double-standard" ball filtering ---
        best_ball: Ball | None = None
        best_score = -1.0

        for rect, score in ball_candidates:
            ball_x = (rect[0] + rect[2]) / 2
            ball_y = (rect[1] + rect[3]) / 2
            is_touching = any(
                p[0] < ball_x < p[2] and p[1] < ball_y < p[3]
                for p in player_boxes
            )
            # At player's feet → stricter threshold (avoid false positives)
            threshold = 0.6 if is_touching else 0.1
            if score > threshold and score > best_score:
                best_score = score
                best_ball = Ball(rect=rect, score=score)

        if best_ball:
            frame_data.ball = best_ball

        return frame_data

    def warmup(self) -> None:
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.detect(dummy, 0)
