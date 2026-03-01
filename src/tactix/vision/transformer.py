"""
Project: Tactix
File Created: 2026-02-02 23:22:57
Author: Xingnan Zhu
File Name: transformer.py
Description:
    Handles the perspective transformation (Homography) between the video frame
    and the 2D tactical board. It maps detected keypoints to their real-world
    counterparts to compute the transformation matrix.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from tactix.core.types import PitchConfig, Player
from tactix.core.keypoints import YOLO_INDEX_MAP 
from tactix.core.geometry import WORLD_POINTS
from tactix.vision.filters import OneEuroFilter

class ViewTransformer:
    def __init__(
        self,
        smooth_enabled: bool = True,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        fps: float = 30.0,
        max_jump: float = 0.4,
        ransac_threshold: float = 3.0,
    ):
        self.homography_matrix = None
        self.scale_x = PitchConfig.PIXEL_WIDTH / PitchConfig.LENGTH
        self.scale_y = PitchConfig.PIXEL_HEIGHT / PitchConfig.WIDTH

        # Homography smoothing via OneEuroFilter (9-D: flattened 3×3 matrix)
        self._smooth_enabled = smooth_enabled
        self._h_filter: Optional[OneEuroFilter] = None
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._fps = fps

        # Robustness parameters
        self._max_jump = max_jump
        self._ransac_threshold = ransac_threshold
        self._prev_raw_h: Optional[np.ndarray] = None

    def set_fps(self, fps: float) -> None:
        """Update the FPS used by the OneEuroFilter. Call before processing starts."""
        self._fps = fps
        # Reset filter so it reinitializes with the correct rate
        self._h_filter = None

    def update(self, keypoints: np.ndarray, confs: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Attempts to update the homography matrix.
        Returns: bool (Whether a valid matrix is available, new or old)
        """
        if keypoints is None: 
            # If no points, check if we have an old matrix to fallback on
            return self.homography_matrix is not None

        src_pts = [] 
        dst_pts = [] 

        for i, (x, y) in enumerate(keypoints):
            if confs[i] < threshold: continue
            
            name = YOLO_INDEX_MAP.get(i)
            if name and name in WORLD_POINTS:
                src_pts.append([x, y])
                world_x, world_y = WORLD_POINTS[name]
                target_x = int(world_x * self.scale_x)
                target_y = int(world_y * self.scale_y)
                dst_pts.append([target_x, target_y])

        # 🔥 Core modification: If not enough points, don't crash, don't clear, just use the old matrix
        if len(src_pts) < 4:
            return self.homography_matrix is not None

        src_arr = np.array(src_pts).reshape(-1, 1, 2)
        dst_arr = np.array(dst_pts).reshape(-1, 1, 2)

        # RANSAC Calculation
        n_src = len(src_pts)
        h, mask = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, self._ransac_threshold)

        if h is not None:
            inliers = int(np.sum(mask))
            min_inliers = max(6, int(n_src * 0.5))

            # Validate: enough inliers
            if inliers < min_inliers:
                return self.homography_matrix is not None

            # Validate: condition number (reject degenerate matrices)
            cond = np.linalg.cond(h)
            if cond > 1e5:
                return self.homography_matrix is not None

            # Validate: frame-to-frame consistency (reject sudden jumps)
            if self._prev_raw_h is not None:
                prev_norm = np.linalg.norm(self._prev_raw_h)
                if prev_norm > 0:
                    relative_diff = np.linalg.norm(h - self._prev_raw_h) / prev_norm
                    if relative_diff > self._max_jump:
                        return self.homography_matrix is not None

            self._prev_raw_h = h.copy()
            if self._smooth_enabled:
                self.homography_matrix = self._smooth_homography(h)
            else:
                self.homography_matrix = h
            return True
        
        # If new calculation is bad, continue using old one
        return self.homography_matrix is not None

    def _smooth_homography(self, raw_h: np.ndarray) -> np.ndarray:
        """
        Apply OneEuroFilter to the 3×3 homography matrix (flattened to 9-D).
        Initializes the filter lazily on the first valid matrix.
        """
        flat = raw_h.flatten()  # (9,)

        if self._h_filter is None:
            # First valid matrix — initialize the filter
            self._h_filter = OneEuroFilter(
                ndim=9,
                rate=self._fps,
                min_cutoff=self._min_cutoff,
                beta=self._beta,
            )

        smoothed = self._h_filter.filter(flat)
        return smoothed.reshape(3, 3)

    def transform_point(self, xy: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        # As long as there is a matrix (even an old one), calculate it!
        if self.homography_matrix is None: return None
        
        point_arr = np.array([[[xy[0], xy[1]]]], dtype=np.float32)
        try:
            transformed = cv2.perspectiveTransform(point_arr, self.homography_matrix)[0][0]
            
            # 🔥 Extra protection: Check if coordinates flew off the earth
            # If calculated coordinates are negative or huge, matrix is bad, return None to avoid drawing errors
            tx, ty = int(transformed[0]), int(transformed[1])
            if -500 < tx < 3000 and -500 < ty < 2000: # Tolerant boundaries
                return tx, ty
        except cv2.error as e:
            print(f"OpenCV Transformation Error: {e}")
        except Exception as e:
            print(f"Unexpected Transformation Error: {e}")
            
        return None

    def transform_players(self, players: List[Player]):
        for p in players:
            # Use bottom_center (feet) for more accurate transformation, otherwise use center
            # Assuming Player.rect is [x1, y1, x2, y2]
            # anchor_x = (x1 + x2) / 2
            # anchor_y = y2 (feet)
            result = self.transform_point(p.anchor)
            
            if result:
                # Assignment depends on Point definition in types.py
                # If p.pitch_position is Point type:
                from tactix.core.types import Point
                p.pitch_position = Point(x=result[0], y=result[1])
            else:
                p.pitch_position = None