"""
Project: Tactix
File Created: 2026-02-15
Author: Xingnan Zhu
File Name: cli.py
Description:
    CLI entry point for the Tactix package.
    Installed as the `tactix` console command via pyproject.toml.
"""

from __future__ import annotations


def main() -> None:
    """Entry point for the ``tactix`` console command."""
    from tactix.config import Config
    from tactix.engine.system import TactixEngine
    from tactix.ui.visualization_menu import VisualizationMenu

    cfg = Config()

    # Visualization Menu
    viz_menu = VisualizationMenu(cfg)
    viz_menu.run()

    # Start the Engine
    engine = TactixEngine(cfg=cfg)
    engine.run()


if __name__ == "__main__":
    main()
