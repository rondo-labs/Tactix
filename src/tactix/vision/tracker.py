"""
Project: Tactix
File Created: 2026-02-02 12:13:21
Author: Xingnan Zhu
File Name: tracker.py
Description:
    Implements object tracking using BotSORT (boxmot library).
    BotSORT uses both motion (Kalman filter) and appearance (ReID CNN) features
    for robust ID persistence across occlusions, re-entries, and camera movements.
    A scene cut detector resets tracker state on hard camera cuts to prevent
    stale Kalman predictions from corrupting re-association.
"""

import cv2
import numpy as np
from collections import deque
from pathlib import Path

from tactix.core.types import FrameData, Point


class Tracker:
    def __init__(
        self,
        fps: float = 30.0,
        reid_model_path: str = "assets/weights/osnet_x0_25_msmt17.pt",
        device: str = "mps",
        scene_cut_threshold: float = 0.5,
    ):
        self._fps = fps
        self._dt = 1.0 / fps
        self._device = device
        self._reid_model_path = reid_model_path
        self._scene_cut_threshold = scene_cut_threshold

        # Lazy-initialized after set_fps() is called from the main pipeline.
        self._tracker = None

        # Scene cut detection: stores the previous frame's grayscale histogram.
        self._prev_hist: np.ndarray | None = None

        # Velocity calculation cache: {player_id: deque([(frame_idx, x, y), ...])}
        self.position_history: dict = {}

    def _init_tracker(self) -> None:
        import torch
        from boxmot import BotSort

        reid_path = Path(self._reid_model_path)
        self._tracker = BotSort(
            reid_weights=reid_path,
            device=torch.device(self._device),
            half=False,
            frame_rate=int(self._fps),
            match_thresh=0.8,
        )

    def set_fps(self, fps: float) -> None:
        """Update FPS used for velocity and track_buffer. Call before processing starts."""
        self._fps = fps
        self._dt = 1.0 / fps

    def detect_scene_cut(self, frame: np.ndarray) -> bool:
        """
        Detect hard camera cuts via grayscale histogram Bhattacharyya distance.
        Returns True if the current frame is a new scene (camera cut detected).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        hist_flat = hist.flatten()

        is_cut = False
        if self._prev_hist is not None:
            dist = cv2.compareHist(self._prev_hist, hist_flat, cv2.HISTCMP_BHATTACHARYYA)
            is_cut = dist > self._scene_cut_threshold

        self._prev_hist = hist_flat
        return is_cut

    def update(self, frame_data: FrameData, frame: np.ndarray) -> None:
        """
        Assign persistent track IDs to frame_data.players using BotSORT.
        Resets tracker state on detected scene cuts.
        """
        if self._tracker is None:
            self._init_tracker()

        # Hard cut: discard stale Kalman predictions and appearance memory.
        if self.detect_scene_cut(frame):
            self._tracker.reset()
            self.position_history.clear()

        if not frame_data.players:
            return

        # Build boxmot input: [N, 6] with [x1, y1, x2, y2, conf, cls_id]
        xyxy = np.array([p.rect for p in frame_data.players], dtype=np.float32)
        confs = np.array([p.confidence for p in frame_data.players], dtype=np.float32).reshape(-1, 1)
        cls_ids = np.array([p.class_id for p in frame_data.players], dtype=np.float32).reshape(-1, 1)
        dets = np.hstack([xyxy, confs, cls_ids])

        # BotSORT uses the frame image for ReID feature extraction.
        tracks = self._tracker.update(dets, frame)

        if tracks is None or len(tracks) == 0:
            return

        # tracks: [M, 8] → [x1, y1, x2, y2, track_id, conf, cls_id, det_ind]
        # det_ind (col 7) directly indexes the input dets array → O(1) assignment.
        # Fall back to IoU matching when det_ind is out of range (lost-track predictions).
        original_xyxy = np.array([p.rect for p in frame_data.players])

        for track in tracks:
            t_id = int(track[4])
            det_idx = int(track[7])
            if 0 <= det_idx < len(frame_data.players):
                frame_data.players[det_idx].id = t_id
            else:
                # Lost-track prediction: match by IoU against current detections.
                t_box = track[:4]
                ious = self._box_iou_batch(t_box, original_xyxy)
                best_idx = int(np.argmax(ious))
                if ious[best_idx] > 0.5:
                    frame_data.players[best_idx].id = t_id

    def update_velocity(self, frame_data: FrameData) -> None:
        """
        Call this method after Transformer calculates pitch_position to compute velocity.
        """
        current_frame = frame_data.frame_index
        dt = self._dt

        for p in frame_data.players:
            if p.id == -1 or p.pitch_position is None:
                continue

            pid = p.id
            cx, cy = p.pitch_position.x, p.pitch_position.y

            if pid not in self.position_history:
                self.position_history[pid] = deque(maxlen=5)

            history = self.position_history[pid]
            history.append((current_frame, cx, cy))

            if len(history) >= 3:
                start_frame, sx, sy = history[0]
                end_frame, ex, ey = history[-1]

                # Guard against large frame gaps caused by tracking loss + re-detection.
                # If history spans more than 10 frames, the position jump is not real
                # motion — clear the stale history and start fresh this frame.
                frame_gap = end_frame - start_frame
                if frame_gap > 10:
                    self.position_history[pid].clear()
                    self.position_history[pid].append((current_frame, cx, cy))
                    continue

                time_diff = (end_frame - start_frame) * dt
                if time_diff > 0:
                    vx = (ex - sx) / time_diff
                    vy = (ey - sy) / time_diff

                    speed = np.sqrt(vx**2 + vy**2)
                    if speed < 15.0:
                        p.velocity = Point(x=vx, y=vy)
                        p.speed = speed
                    else:
                        # Outlier, likely a position jump — zero out velocity.
                        p.velocity = Point(0, 0)
                        p.speed = 0

    @staticmethod
    def _box_iou_batch(box_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Calculate IoU between one box (box_a) and a set of boxes (boxes_b)."""
        x_a = np.maximum(box_a[0], boxes_b[:, 0])
        y_a = np.maximum(box_a[1], boxes_b[:, 1])
        x_b = np.minimum(box_a[2], boxes_b[:, 2])
        y_b = np.minimum(box_a[3], boxes_b[:, 3])

        inter_area = np.maximum(0, x_b - x_a) * np.maximum(0, y_b - y_a)

        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        boxes_b_area = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

        return inter_area / (box_a_area + boxes_b_area - inter_area)
