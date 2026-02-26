"""
Project: Tactix
File Created: 2026-02-26
Author: Xingnan Zhu
File Name: yolo_impl.py
Description:
	YOLO-based detector implementation for Tactix, refactored from vision/detector.py.
"""

import numpy as np
import supervision as sv
from ultralytics import YOLO

from tactix.core.types import Player, Ball, FrameData, TeamID
from tactix.models.interface import BaseDetector

class YOLODetector(BaseDetector):
	supports_masks = False

	def __init__(
		self,
		model_weights: str,
		device: str = 'mps',
		conf_threshold: float = 0.3,
		iou_threshold: float = 0.7
	):
		self.model = YOLO(model_weights)
		self.device = device
		self.conf_threshold = conf_threshold
		self.iou_threshold = iou_threshold
		self.CLASS_MAP = {
			0: 'ball',
			1: 'goalkeeper',
			2: 'player',
			3: 'referee'
		}

	def detect(self, frame: np.ndarray, frame_index: int) -> FrameData:
		results = self.model(
			frame,
			device=self.device,
			verbose=False,
			conf=self.conf_threshold,
			iou=self.iou_threshold,
			imgsz=1280,
		)[0]
		detections = sv.Detections.from_ultralytics(results)
		frame_data = FrameData(frame_index=frame_index, image_shape=frame.shape[:2])
		ball_candidates = []
		player_boxes = []
		for i, class_id in enumerate(detections.class_id):
			xyxy = detections.xyxy[i]
			rect = tuple(xyxy.tolist())
			confidence = float(detections.confidence[i])
			class_name = self.CLASS_MAP.get(class_id, 'unknown')
			x1, y1, x2, y2 = xyxy
			width, height = x2 - x1, y2 - y1
			area = width * height
			ratio = width / height if height > 0 else 0
			if class_name != 'ball' and area < 900 and ratio > 0.7:
				class_name = 'ball'
			if class_name == 'ball':
				if area > 900 or ratio < 0.6 or ratio > 1.5:
					continue
			if class_name == 'ball':
				ball_candidates.append((rect, confidence))
			elif class_name in ['player', 'goalkeeper', 'referee']:
				player = Player(
					id=-1,
					rect=rect,
					class_id=class_id,
					confidence=confidence,
					team=TeamID.UNKNOWN
				)
				if class_name == 'referee':
					player.team = TeamID.REFEREE
				elif class_name == 'goalkeeper':
					player.team = TeamID.GOALKEEPER
				frame_data.players.append(player)
				player_boxes.append(xyxy)
		best_ball = None
		best_score = -1.0
		for rect, score in ball_candidates:
			is_touching_player = False
			ball_x = (rect[0] + rect[2]) / 2
			ball_y = (rect[1] + rect[3]) / 2
			for p_box in player_boxes:
				px1, py1, px2, py2 = p_box
				if px1 < ball_x < px2 and py1 < ball_y < py2:
					is_touching_player = True
					break
			threshold = 0.6 if is_touching_player else 0.1
			if score > threshold:
				if score > best_score:
					best_score = score
					best_ball = Ball(rect=rect, score=score)
		if best_ball:
			frame_data.ball = best_ball
		return frame_data

	def warmup(self) -> None:
		dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
		self.detect(dummy, 0)
