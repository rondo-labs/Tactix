"""
Project: Tactix
File Created: 2026-02-04 16:14:41
Author: Xingnan Zhu
File Name: run.py
Description:
    The entry point for the Tactix application.
    It handles initialization, optional interactive calibration, visualization settings,
    and starts the main engine loop.
"""


from tactix.engine.system import TactixEngine
from tactix.config import Config
from tactix.ui.visualization_menu import VisualizationMenu

if __name__ == "__main__":
    # 1. Initialize Config
    cfg = Config()

    # 2. Visualization Menu (Optional but recommended)
    # Allows user to toggle layers before starting
    viz_menu = VisualizationMenu(cfg)
    viz_menu.run()

    # 3. Start the Engine
    engine = TactixEngine(cfg=cfg)
    engine.run()