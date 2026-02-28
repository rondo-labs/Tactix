"""
Project: Tactix
File Created: 2026-02-02 16:16:28
Author: Xingnan Zhu
File Name: geometry.py
Description:
    Defines the physical geometry of the football pitch. It maps standard
    pitch landmarks to their real-world coordinates (in meters), serving as
    the ground truth for perspective transformation and tactical analysis.
"""

from tactix.core.types import PitchConfig

# Real pitch dimensions (meters)
L = PitchConfig.LENGTH # 105
W = PitchConfig.WIDTH  # 68

# Define keypoints in "Real World" coordinates (Unit: meters)
# Coordinate System: Origin (0,0) at Top-Left Corner Flag
# X-axis: Right, Y-axis: Down
WORLD_POINTS = {
    # Corners
    "TL_CORNER": [0, 0],
    "BL_CORNER": [0, W],
    "TR_CORNER": [L, 0],
    "BR_CORNER": [L, W],

    # Midfield
    "MID_TOP":     [L/2, 0],
    "MID_BOTTOM":  [L/2, W],
    "CENTER_SPOT": [L/2, W/2],

    # Center circle
    "CIRCLE_TOP":   [L/2, W/2 - 9.15],
    "CIRCLE_LEFT":  [L/2 - 9.15, W/2],
    "CIRCLE_RIGHT": [L/2 + 9.15, W/2],

    # Left Penalty Area
    "L_PA_TOP_LINE":      [0, 13.84],
    "L_PA_TOP_CORNER":    [16.5, 13.84],
    "L_PA_BOTTOM_CORNER": [16.5, W-13.84],
    "L_PA_BOTTOM_LINE":   [0, W-13.84],

    # Left Goal Area (6-yard box)
    "L_GA_TOP_LINE":      [0, 24.84],
    "L_GA_TOP_CORNER":    [5.5, 24.84],
    "L_GA_BOTTOM_CORNER": [5.5, W-24.84],
    "L_GA_BOTTOM_LINE":   [0, W-24.84],

    # Right Penalty Area
    "R_PA_TOP_LINE":      [L, 13.84],
    "R_PA_TOP_CORNER":    [L-16.5, 13.84],
    "R_PA_BOTTOM_CORNER": [L-16.5, W-13.84],
    "R_PA_BOTTOM_LINE":   [L, W-13.84],

    # Right Goal Area (6-yard box)
    "R_GA_TOP_LINE":      [L, 24.84],
    "R_GA_TOP_CORNER":    [L-5.5, 24.84],
    "R_GA_BOTTOM_CORNER": [L-5.5, W-24.84],
    "R_GA_BOTTOM_LINE":   [L, W-24.84],
}