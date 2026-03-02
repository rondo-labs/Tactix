"""
Project: Tactix
File Created: 2026-02-02 12:13:36
Author: Xingnan Zhu
File Name: camera.py
Description:
    Implements camera motion tracking using Optical Flow (Lucas-Kanade).
    It stabilizes the pitch detection by tracking keypoints across frames,
    reducing jitter and handling frames where the detection model might fail.
    Includes a smoothing mechanism (Moving Average) for robust coordinate output.
"""

from typing import Optional, List, Tuple, Union
import cv2
import numpy as np
from collections import deque

class CameraSmoother:
    """
    Smooths coordinate points using Moving Average.
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        # Queue to store historical coordinate datasets
        self.history = deque(maxlen=window_size)

    def update(self, current_points: np.ndarray) -> np.ndarray:
        """
        Adds new datasets and returns the smoothed result.
        :param current_points: Coordinates of the current frame (N, 2)
        :return: Smoothed coordinates (N, 2)
        """
        self.history.append(current_points)
        
        # Calculate average
        # stack converts list of arrays into (window_size, N, 2)
        # mean(axis=0) averages over the time dimension -> (N, 2)
        smoothed = np.mean(np.stack(self.history), axis=0)
        return smoothed


class CameraTracker:
    def __init__(self, initial_keypoints: np.ndarray = None, smoothing_window: int = 5,
                 max_drift_frames: int = 20, blend_alpha: float = 0.7):
        """
        Initializes the camera tracker.
        :param initial_keypoints: Initial keypoint coordinates (N, 2), if None, waits for first update to initialize
        :param smoothing_window: Size of the smoothing window
        :param max_drift_frames: Max consecutive optical flow frames before forced recalibration
        :param blend_alpha: Weight for new YOLO detection in soft_reset blend (0-1)
        """
        self.current_keypoints: Optional[np.ndarray] = None
        self._num_points: int = 0
        # Maps each tracked point to its original YOLO keypoint index (0–25)
        self._kpt_indices: Optional[np.ndarray] = None
        if initial_keypoints is not None:
            self.current_keypoints = initial_keypoints.astype(np.float32).reshape(-1, 1, 2)
            self._num_points = len(initial_keypoints)

        self.prev_gray: Optional[np.ndarray] = None

        # Optical Flow parameters (LK Optical Flow)
        self.lk_params = dict(
            winSize=(20, 20),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Initialize smoother
        self.smoother = CameraSmoother(window_size=smoothing_window)

        # Drift detection: count consecutive frames running on optical flow only
        self._consecutive_flow_frames: int = 0
        self._max_drift_frames: int = max_drift_frames
        self._blend_alpha: float = blend_alpha

    def reset(self, keypoints: np.ndarray, frame: np.ndarray):
        """
        Force resets tracking points (usually called when YOLO detects high-confidence keypoints).
        """
        self.current_keypoints = keypoints.astype(np.float32).reshape(-1, 1, 2)
        self._num_points = len(keypoints)
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._kpt_indices = np.arange(self._num_points)
        self._consecutive_flow_frames = 0
        # Reset smoother history to avoid old datasets dragging down new position accuracy
        self.smoother = CameraSmoother(window_size=self.smoother.window_size)
        self.smoother.update(keypoints) # Immediately push current value

    def soft_reset(self, keypoints: np.ndarray, frame: np.ndarray,
                   confs: Optional[np.ndarray] = None, conf_threshold: float = 0.3):
        """
        Gently updates tracking points from YOLO without destroying smoother history.

        Accepts the full (26, 2) keypoint array from YOLO. Only tracks points that
        have valid coordinates (non-zero) and confidence above threshold. Maintains
        _kpt_indices so downstream code can reconstruct the 26-point indexed array.
        """
        # Determine which keypoints are valid for tracking
        if confs is not None:
            valid = confs >= conf_threshold
        else:
            # Fallback: treat non-zero coordinates as valid
            valid = ~((keypoints[:, 0] == 0) & (keypoints[:, 1] == 0))
        valid_indices = np.where(valid)[0]

        if len(valid_indices) < 4:
            # Too few valid points — skip soft_reset entirely
            return

        valid_pts = keypoints[valid_indices].astype(np.float32).reshape(-1, 1, 2)

        # Weighted blend if index set matches the currently tracked set
        if (self.current_keypoints is not None
                and self._kpt_indices is not None
                and np.array_equal(valid_indices, self._kpt_indices)):
            self.current_keypoints = (
                self._blend_alpha * valid_pts + (1.0 - self._blend_alpha) * self.current_keypoints
            )
        else:
            # Index set changed — hard reset tracking state
            self.current_keypoints = valid_pts
            self._num_points = len(valid_indices)
            self.smoother = CameraSmoother(window_size=self.smoother.window_size)
            self.smoother.update(keypoints[valid_indices])

        self._kpt_indices = valid_indices
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._consecutive_flow_frames = 0

    def update(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Takes the current frame, calculates new positions using Optical Flow.
        Returns (smoothed_points, kpt_indices) or None if tracking fails.
        kpt_indices maps each returned point to its original YOLO keypoint index.
        """
        if self.current_keypoints is None or self._kpt_indices is None:
            return None

        # Drift guard: if optical flow has been running too long without YOLO reset,
        # accumulated error is too high — force recalibration
        self._consecutive_flow_frames += 1
        if self._consecutive_flow_frames > self._max_drift_frames:
            print(f"⚠️ Optical flow drift limit reached ({self._max_drift_frames} frames), requesting recalibration")
            return None

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If running update for the first time and no prev_gray (but points given in init)
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            raw_points = self.current_keypoints.reshape(-1, 2)
            return self.smoother.update(raw_points), self._kpt_indices.copy()

        # === Core: Optical Flow Tracking ===
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray, self.current_keypoints, None, **self.lk_params
        )

        # Check tracking status — allow partial survival (≥ 4 points)
        if status is not None:
            alive = status.flatten().astype(bool)
            n_alive = int(np.sum(alive))

            if n_alive >= 4:
                # Keep only the surviving points and their index mapping
                if not np.all(alive):
                    new_points = new_points[alive]
                    self.current_keypoints = new_points
                    self._kpt_indices = self._kpt_indices[alive]
                    # Rebuild smoother since point count changed
                    self.smoother = CameraSmoother(window_size=self.smoother.window_size)
                else:
                    self.current_keypoints = new_points

                raw_points = self.current_keypoints.reshape(-1, 2)
                smoothed_points = self.smoother.update(raw_points)
                self.prev_gray = frame_gray
                return smoothed_points, self._kpt_indices.copy()

        # Fewer than 4 points survived — request recalibration
        print("⚠️ Camera Tracker lost too many keypoints!")
        return None
