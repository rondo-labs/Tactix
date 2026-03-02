"""
Project: Tactix
File Created: 2026-03-02
Author: Xingnan Zhu
File Name: json_exporter.py
Description:
    Exports per-frame tracking data to a viewer-compatible JSON format.
    Output is consumed by viewer/index.html for browser-based 2D visualization.

    Output JSON structure:
        {
          "meta": { "fps": 25, "total_frames": N, "canvas": {"width": W, "height": H} },
          "frames": [
            {
              "i": 0,
              "players": [{"id":1,"team":"A","x":780,"y":505,"vx":0.1,"vy":0.05,
                           "speed":2.1,"jersey":"7","pressure":0.2}],
              "ball": {"x": 750, "y": 490}   // null if not visible
            }, ...
          ]
        }

    Coordinates: minimap pixel space (same as pitch_position in FrameData).
    canvas.width/height tells the viewer the reference canvas size so it can
    scale positions to whatever size it draws at.
"""

import json
import os
from typing import List, Optional, TYPE_CHECKING

from tactix.export.base import BaseExporter
from tactix.core.types import FrameData, TeamID, PitchConfig

if TYPE_CHECKING:
    from tactix.config import Config


# Map TeamID enum to short strings for compact JSON
_TEAM_LABELS = {
    TeamID.A:          "A",
    TeamID.B:          "B",
    TeamID.REFEREE:    "REF",
    TeamID.GOALKEEPER: "GK",
    TeamID.UNKNOWN:    "?",
}


class ViewerJsonExporter(BaseExporter):
    """
    Buffers per-frame tracking data and writes a single JSON file on save().
    Designed to be consumed by viewer/index.html.
    """

    def __init__(self, output_path: str, cfg: "Config", fps: float = 25.0) -> None:
        self.output_path = output_path
        self.cfg = cfg
        self.fps = fps
        self._frames: List[dict] = []

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ------------------------------------------------------------------
    # BaseExporter interface
    # ------------------------------------------------------------------

    def add_frame(self, frame_data: FrameData) -> None:
        """Buffer one frame of tracking data."""
        players = []
        for p in frame_data.players:
            if not p.pitch_position:
                continue

            entry: dict = {
                "id":    p.id,
                "team":  _TEAM_LABELS.get(p.team, "?"),
                "x":     round(p.pitch_position.x, 1),
                "y":     round(p.pitch_position.y, 1),
            }

            if p.velocity:
                entry["vx"] = round(p.velocity.x, 3)
                entry["vy"] = round(p.velocity.y, 3)

            if p.speed:
                entry["speed"] = round(p.speed, 2)

            if p.jersey_number:
                entry["jersey"] = p.jersey_number

            if p.pressure > 0.01:
                entry["pressure"] = round(p.pressure, 3)

            players.append(entry)

        ball = None
        if frame_data.ball and frame_data.ball.pitch_position:
            ball = {
                "x": round(frame_data.ball.pitch_position.x, 1),
                "y": round(frame_data.ball.pitch_position.y, 1),
            }

        self._frames.append({
            "i":       frame_data.frame_index,
            "players": players,
            "ball":    ball,
        })

    def save(self) -> None:
        """Write buffered frames to JSON file."""
        payload = {
            "meta": {
                "fps":          self.fps,
                "total_frames": len(self._frames),
                "canvas": {
                    "width":  PitchConfig.PIXEL_WIDTH,
                    "height": PitchConfig.PIXEL_HEIGHT,
                },
                # Physical pitch dimensions (meters) — used by viewer for scale
                "pitch": {
                    "length": PitchConfig.LENGTH,
                    "width":  PitchConfig.WIDTH,
                },
            },
            "frames": self._frames,
        }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

        size_kb = os.path.getsize(self.output_path) / 1024
        print(f"   🌐 Viewer JSON: {self.output_path} ({len(self._frames)} frames, {size_kb:.0f} KB)")
