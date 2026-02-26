"""
Project: Tactix
File Created: 2026-02-07 16:39:59
Author: Xingnan Zhu
File Name: interface.py
Description: 
	Defines the abstract interface for pluggable detection backends (YOLO, SAM, etc.)
"""

from abc import ABC, abstractmethod
import numpy as np
from tactix.core.types import FrameData

class BaseDetector(ABC):
	"""
	Abstract base class for detection models.
	"""
	device: str
	supports_masks: bool = False

	@abstractmethod
	def detect(self, frame: np.ndarray, frame_index: int) -> FrameData:
		"""
		Run detection on a frame and return FrameData.
		"""
		pass

	def warmup(self) -> None:
		"""
		Optional: Preload model weights, run dummy inference, etc.
		"""
		pass
