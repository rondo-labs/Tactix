"""
Project: Tactix
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: hybrid_detector.py
Description:
    Hybrid detector combining any BaseDetector backend (YOLO or RF-DETR)
    with SAM3 for single-frame mask refinement.
"""

import numpy as np
from tactix.models.interface import BaseDetector
from tactix.models.sam3_impl import SAM3Refiner
from tactix.core.types import FrameData


class HybridDetector(BaseDetector):
    """
    Wraps any BaseDetector (YOLODetector or RFDETRDetector) and runs SAM3
    mask refinement on every detected player after detection.

    Usage:
        base = YOLODetector(...)         # or RFDETRDetector(...)
        detector = HybridDetector(base, sam3_weights=..., device=...)
    """

    supports_masks = True

    def __init__(
        self,
        detector: BaseDetector,
        sam3_weights: str,
        device: str = "mps",
        conf_sam3: float = 0.25,
        sam3_half: bool = True,
    ) -> None:
        self._detector = detector
        self.sam3 = SAM3Refiner(sam3_weights, device=device, conf=conf_sam3, half=sam3_half)
        self.device = device

    def detect(self, frame: np.ndarray, frame_index: int) -> FrameData:
        # Step 1: base detection (YOLO or RF-DETR)
        frame_data = self._detector.detect(frame, frame_index)
        # Step 2: SAM3 single-frame mask refinement
        self.sam3.set_image(frame)
        self.sam3.refine(frame_data.players)
        return frame_data

    def warmup(self) -> None:
        self._detector.warmup()
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.sam3.set_image(dummy)
