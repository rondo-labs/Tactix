"""
Project: Tactix
File Created: 2026-02-02 23:35:48
Author: Xingnan Zhu
File Name: keypoints.py
Description: Defines standard football pitch dimensions and key coordinate points.
    It maps physical locations to pixel coordinates and provides a mapping for
    YOLO model outputs to specific pitch landmarks.
"""


# tactix/core/keypoints.py
import numpy as np
from tactix.core.types import PitchConfig

# ==========================================
# ⚙️ Basic Configuration
# ==========================================
# Read configuration from types.py to avoid hardcoding
W = PitchConfig.PIXEL_WIDTH   # 1559
H = PitchConfig.PIXEL_HEIGHT  # 1010

LENGTH_M = PitchConfig.LENGTH # 105.0
WIDTH_M = PitchConfig.WIDTH   # 68.0

# Scale (Pixels/Meter)
X_PER_M = W / LENGTH_M
Y_PER_M = H / WIDTH_M

def to_px(x_m, y_m):
    """Convert physical coordinates (meters) to tactical board pixel coordinates"""
    return [int(x_m * X_PER_M), int(y_m * Y_PER_M)]

# ==========================================
# 📍 Full Pitch Keypoint Dictionary
# ==========================================
# Origin (0,0) = Top-Left Corner Flag
# X-axis -> Right (0 to 105)
# Y-axis -> Down (0 to 68)

KEY_POINTS = {
    # ----------------------------------------
    # 1. Four Corners
    # ----------------------------------------
    "TL_CORNER": to_px(0, 0),              # Top-Left
    "BL_CORNER": to_px(0, WIDTH_M),        # Bottom-Left
    "TR_CORNER": to_px(LENGTH_M, 0),       # Top-Right
    "BR_CORNER": to_px(LENGTH_M, WIDTH_M), # Bottom-Right

    # ----------------------------------------
    # 2. Midfield Area
    # ----------------------------------------
    "MID_TOP":     to_px(LENGTH_M/2, 0),         # Midline Top
    "MID_BOTTOM":  to_px(LENGTH_M/2, WIDTH_M),   # Midline Bottom
    "CENTER_SPOT": to_px(LENGTH_M/2, WIDTH_M/2), # Center Spot

    # Center Circle: midline intersection (top) + left/right widest points
    "CIRCLE_TOP":   to_px(LENGTH_M/2, WIDTH_M/2 - 9.15),
    "CIRCLE_LEFT":  to_px(LENGTH_M/2 - 9.15, WIDTH_M/2),
    "CIRCLE_RIGHT": to_px(LENGTH_M/2 + 9.15, WIDTH_M/2),

    # ----------------------------------------
    # 3. Left Penalty Area - X = 0 ~ 16.5
    # ----------------------------------------
    "L_PA_TOP_LINE":      to_px(0, 13.84),        # Goal line / PA top
    "L_PA_TOP_CORNER":    to_px(16.5, 13.84),     # PA top inner corner
    "L_PA_BOTTOM_CORNER": to_px(16.5, 68-13.84),  # PA bottom inner corner
    "L_PA_BOTTOM_LINE":   to_px(0, 68-13.84),     # Goal line / PA bottom

    # Left Goal Area (6-yard box)
    "L_GA_TOP_LINE":      to_px(0, 24.84),        # Goal line / GA top
    "L_GA_TOP_CORNER":    to_px(5.5, 24.84),      # GA top inner corner
    "L_GA_BOTTOM_CORNER": to_px(5.5, 68-24.84),   # GA bottom inner corner
    "L_GA_BOTTOM_LINE":   to_px(0, 68-24.84),     # Goal line / GA bottom

    # ----------------------------------------
    # 4. Right Penalty Area - X = 88.5 ~ 105
    # ----------------------------------------
    "R_PA_TOP_LINE":      to_px(LENGTH_M, 13.84),      # Goal line / PA top
    "R_PA_TOP_CORNER":    to_px(105-16.5, 13.84),      # PA top inner corner
    "R_PA_BOTTOM_CORNER": to_px(105-16.5, 68-13.84),   # PA bottom inner corner
    "R_PA_BOTTOM_LINE":   to_px(LENGTH_M, 68-13.84),   # Goal line / PA bottom

    # Right Goal Area (6-yard box)
    "R_GA_TOP_LINE":      to_px(LENGTH_M, 24.84),      # Goal line / GA top
    "R_GA_TOP_CORNER":    to_px(105-5.5, 24.84),       # GA top inner corner
    "R_GA_BOTTOM_CORNER": to_px(105-5.5, 68-24.84),    # GA bottom inner corner
    "R_GA_BOTTOM_LINE":   to_px(LENGTH_M, 68-24.84),   # Goal line / GA bottom
}

def get_target_points(keys):
    """
    Get coordinate array by key names
    """
    points = []
    for k in keys:
        if k not in KEY_POINTS:
            # Fault tolerance: Allow reverse lookup (e.g., mapping PA_TOP_LEFT to L_PA_TOP_CORNER)
            # For simplicity, raise error if not found
            available = list(KEY_POINTS.keys())
            raise ValueError(f"❌ Unknown point '{k}'. Please choose from:\n{available}")
        points.append(KEY_POINTS[k])
    return np.array(points)


# ==========================================
# 5. YOLO Model Output Mapping (26-point unified scheme)
# ==========================================
# This order must match pitch_kpt_26.yaml used during training.
# Indices 0-25 correspond to the 26 keypoints detected by the model.
YOLO_INDEX_MAP = {
    # Corners
    0:  "TL_CORNER",
    1:  "MID_TOP",
    2:  "TR_CORNER",

    # Left goal line intersections (x = 0)
    3:  "L_PA_TOP_LINE",
    4:  "L_GA_TOP_LINE",
    5:  "L_GA_BOTTOM_LINE",
    6:  "L_PA_BOTTOM_LINE",

    7:  "BL_CORNER",
    8:  "MID_BOTTOM",
    9:  "BR_CORNER",

    # Right goal line intersections (x = 105)
    10: "R_PA_BOTTOM_LINE",
    11: "R_GA_BOTTOM_LINE",
    12: "R_GA_TOP_LINE",
    13: "R_PA_TOP_LINE",

    # Penalty area inner corners
    14: "R_PA_TOP_CORNER",
    15: "R_PA_BOTTOM_CORNER",
    16: "L_PA_TOP_CORNER",
    17: "L_PA_BOTTOM_CORNER",

    # Goal area inner corners
    18: "L_GA_TOP_CORNER",
    19: "L_GA_BOTTOM_CORNER",
    20: "R_GA_TOP_CORNER",
    21: "R_GA_BOTTOM_CORNER",

    # Center circle
    22: "CIRCLE_TOP",
    23: "CIRCLE_LEFT",
    24: "CENTER_SPOT",
    25: "CIRCLE_RIGHT",
}
