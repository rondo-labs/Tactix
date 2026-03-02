"""
Project: Tactix
File Created: 2026-02-05 17:42:12
Author: Xingnan Zhu
File Name: ai_estimator.py
Description:
    Implements the pitch keypoint estimation using a YOLO-Pose model.
    It detects 26 standard keypoints on the football pitch to establish
    the correspondence between the video frame and the 2D tactical board.
"""

from ultralytics import YOLO
import numpy as np
from typing import Tuple, Optional, List
from tactix.vision.calibration.base import BasePitchEstimator

class AIPitchEstimator(BasePitchEstimator):
    def __init__(self, model_path: str, device: str = 'mps'):
        print(f"🏟️ Loading AI Pitch Model: {model_path}...")
        self.model = YOLO(model_path)
        self.device = device

    def predict(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Multi-box keypoint fusion: for each keypoint index, takes the
        highest-confidence prediction across all detected boxes.

        Returns: (keypoints_xy, confidences)
        xy shape: (26, 2)
        conf shape: (26,)
        """
        results = self.model(frame, device=self.device, verbose=False)[0]

        if results.keypoints is None or len(results.keypoints.data) == 0:
            return None, None

        all_kpts = results.keypoints.data.cpu().numpy()  # (N_boxes, 26, 3)
        n_kpts = all_kpts.shape[1]

        # Per-index fusion: pick the box with the highest confidence for each keypoint
        best_box_per_kpt = all_kpts[:, :, 2].argmax(axis=0)  # (26,)
        best_xy = np.zeros((n_kpts, 2), dtype=np.float32)
        best_conf = np.zeros(n_kpts, dtype=np.float32)
        for i in range(n_kpts):
            b = best_box_per_kpt[i]
            best_xy[i] = all_kpts[b, i, :2]
            best_conf[i] = all_kpts[b, i, 2]

        # Zero-coordinate cleanup: (0,0) is a placeholder for undetected keypoints,
        # but it coincides with TL_CORNER (index 0). Suppress non-TL_CORNER zeros.
        for i in range(1, n_kpts):
            if best_xy[i, 0] == 0 and best_xy[i, 1] == 0:
                best_conf[i] = 0.0

        return best_xy, best_conf
    
class MockPitchEstimator(BasePitchEstimator):
    """
    A dummy estimator that returns fixed coordinates. Useful for testing without loading heavy models.
    """
    def __init__(self, mock_points: List[Tuple[int, int, int]]):
        print(f"⚠️ Warning: Using Mock Pitch Estimator (Fixed Coordinates)")
        self.mock_points = mock_points
        
        self.dummy_xy = np.zeros((26, 2), dtype=float)
        self.dummy_conf = np.zeros(26, dtype=float)

        for x, y, idx in mock_points:
            if idx < 26:
                self.dummy_xy[idx] = [x, y]
                self.dummy_conf[idx] = 1.0 # Full confidence

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Return the 4 points regardless of the input frame
        return self.dummy_xy, self.dummy_conf