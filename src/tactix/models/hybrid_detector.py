"""
Project: Tactix
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: hybrid_detector.py
Description:
    Hybrid detector combining YOLO and SAM 3 for mask refinement.
"""

import numpy as np
from tactix.models.interface import BaseDetector
from tactix.models.yolo_impl import YOLODetector
from tactix.models.sam3_impl import SAM3Refiner
from tactix.core.types import FrameData

class HybridDetector(BaseDetector):
    supports_masks = True

    def __init__(self, yolo_weights: str, sam3_weights: str, device: str = 'mps', conf_yolo: float = 0.3, conf_sam3: float = 0.25, sam3_half: bool = True):
        self.yolo = YOLODetector(yolo_weights, device=device, conf_threshold=conf_yolo)
        self.sam3 = SAM3Refiner(sam3_weights, device=device, conf=conf_sam3, half=sam3_half)
        self.device = device

    def detect(self, frame: np.ndarray, frame_index: int) -> FrameData:
        # Step 1: YOLO detection
        frame_data = self.yolo.detect(frame, frame_index)
        # Step 2: SAM3 mask refinement
        self.sam3.set_image(frame)
        self.sam3.refine(frame_data.players)
        # Optionally: refine ball mask if needed
        # (Ball mask refinement can be added here)
        return frame_data

    def warmup(self) -> None:
        self.yolo.warmup()
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.sam3.set_image(dummy)
