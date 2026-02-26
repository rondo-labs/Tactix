"""
Project: Tactix
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: sam3_impl.py
Description:
    SAM 3-based mask refiner for Tactix. Requires sam3.pt weights and Ultralytics >=8.3.237.
"""

import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
from tactix.core.types import Player

class SAM3Refiner:
    def __init__(self, model_path: str, device: str = 'mps', conf: float = 0.25, half: bool = True):
        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=model_path,
            half=half,
            save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.device = device
        self.conf = conf
        self.half = half
        self.model_path = model_path
        self._features = None

    def set_image(self, frame: np.ndarray):
        self.predictor.set_image(frame)
        self._features = self.predictor.features
        self._src_shape = frame.shape[:2]

    def refine(self, players: list[Player]) -> None:
        """
        Refine YOLO-detected players with SAM 3 masks using bbox prompts.
        Updates each Player.mask and Player.mask_score in-place.
        """
        if self._features is None:
            raise RuntimeError("Call set_image(frame) before refine().")
        bboxes = [p.rect for p in players]
        if not bboxes:
            return
        masks, boxes = self.predictor.inference_features(
            self._features,
            src_shape=self._src_shape,
            bboxes=bboxes
        )
        if masks is not None:
            masks_np = masks.cpu().numpy()
            for p, mask in zip(players, masks_np):
                p.mask = mask
                p.mask_score = 1.0  # Placeholder, SAM 3 returns binary mask; can add score if available

    def segment_by_concept(self, text_prompts: list[str]) -> tuple:
        """
        Segment all instances matching text prompts. Returns masks and boxes.
        """
        if self._features is None:
            raise RuntimeError("Call set_image(frame) before segment_by_concept().")
        masks, boxes = self.predictor.inference_features(
            self._features,
            src_shape=self._src_shape,
            text=text_prompts
        )
        return masks, boxes

    def clear(self):
        self._features = None
        self._src_shape = None
