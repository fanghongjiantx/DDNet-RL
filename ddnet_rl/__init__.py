"""
DDNet RL - Gymnasium environment for DDNet platformer levels.

Uses a Rust backend (twgame physics) via PyO3 for fast, deterministic simulation.
"""

from .env import DDNetEnv

__all__ = ["DDNetEnv"]
