"""
OpenAI Gymnasium environment for DDNet RL.

Wraps the Rust LiteEnv (twgame physics) with a standard Gym interface:
- Observation: dict with position, velocity, tile grid, progress info
- Action: discrete movement + continuous aim angle
- Reward: progress-based with bonuses for checkpoints and finish
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ddnet_rl_rust import LiteEnv

# Tile type constants (matching Rust side)
TILE_AIR = 0
TILE_COLLISION = 1
TILE_KILL = 2
TILE_UNHOOKABLE = 3
TILE_FREEZE = 9
TILE_START = 33
TILE_FINISH = 34

# Action encoding
# Discrete: 0=nothing, 1=left, 2=right, 3=left+jump, 4=right+jump, 5=jump
#           6=hook, 7=left+hook, 8=right+hook, 9=hook+jump, 10=left+hook+jump, 11=right+hook+jump
ACTION_TABLE = [
    # (direction, jump, hook)
    (0, False, False),   # 0: nothing
    (-1, False, False),  # 1: left
    (1, False, False),   # 2: right
    (-1, True, False),   # 3: left+jump
    (1, True, False),    # 4: right+jump
    (0, True, False),    # 5: jump
    (0, False, True),    # 6: hook
    (-1, False, True),   # 7: left+hook
    (1, False, True),    # 8: right+hook
    (0, True, True),     # 9: jump+hook
    (-1, True, True),    # 10: left+jump+hook
    (1, True, True),     # 11: right+jump+hook
]


class DDNetEnv(gym.Env):
    """Gymnasium environment for DDNet RL training.

    Observation space (Box):
        - pos (2,): player position (x, y) in world coordinates
        - vel (2,): player velocity (vx, vy)
        - grounded (1,): whether player is on ground
        - jumps (1,): jumps remaining
        - hook_state (1,): hook state (0=idle, 1=flying, 2=grabbed)
        - tile_grid (15*15,): tile types around player (7×7 grid)

    Action space (Discrete):
        0-11: combined movement/jump/hook actions
    """

    metadata = {"render_modes": []}

    def __init__(self, map_path: str, tile_radius: int = 7, max_steps: int = 10000):
        super().__init__()
        self._env = LiteEnv(map_path)
        self._tile_radius = tile_radius
        self._max_steps = max_steps
        self._step_count = 0
        self._start_pos = self._env.spawn_pos()
        self._prev_dist_to_finish: float | None = None

        # Track for computing progress reward
        self._last_pos = np.array(self._env.pos, dtype=np.float32)

        grid_size = (2 * tile_radius + 1) ** 2

        # Observation space: Box with all features concatenated
        obs_dim = 2 + 2 + 1 + 1 + 1 + grid_size  # pos + vel + grounded + jumps + hook + tiles
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: 12 discrete actions
        self.action_space = spaces.Discrete(len(ACTION_TABLE))

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        pos = np.array(self._env.pos, dtype=np.float32)
        vel = np.array(self._env.vel, dtype=np.float32)
        grounded = np.array([1.0 if self._env.is_grounded_at(pos[0], pos[1]) else 0.0], dtype=np.float32)
        jumps = np.array([self._env.jumps_remaining], dtype=np.float32) / 2.0
        hook_state = np.array([self._env.hook_state], dtype=np.float32) / 2.0

        tile_data = self._env.tile_grid(self._tile_radius)
        tile_grid = np.frombuffer(tile_data, dtype=np.uint8).astype(np.float32) / 255.0

        return np.concatenate([pos / 1000.0, vel / 100.0, grounded, jumps, hook_state, tile_grid]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset()
        self._step_count = 0
        self._last_pos = np.array(self._env.pos, dtype=np.float32)
        self._prev_dist_to_finish = None
        return self._get_obs(), {}

    def step(self, action: int):
        direction, jump, hook = ACTION_TABLE[action]

        # Default aim: in movement direction for hook
        aim_x = float(direction) if direction != 0 else 0.0
        aim_y = -0.5  # slight upward angle

        pos_x, pos_y, vel_x, vel_y, grounded, dead, finished = self._env.step(
            direction, jump, hook, aim_x, aim_y
        )

        self._step_count += 1
        current_pos = np.array([pos_x, pos_y], dtype=np.float32)

        # Reward calculation
        reward = 0.0

        # Progress reward (distance moved toward finish approximated by movement)
        progress = float(np.linalg.norm(current_pos - self._last_pos))
        reward += progress * 0.01

        # Small survival reward
        reward += 0.0001

        self._last_pos = current_pos

        # Death penalty
        terminated = dead
        if dead:
            reward -= 5.0

        # Finish bonus
        if finished:
            reward += 100.0
            terminated = True

        # Timeout
        truncated = self._step_count >= self._max_steps

        obs = self._get_obs()
        info = {
            "pos": (pos_x, pos_y),
            "vel": (vel_x, vel_y),
            "grounded": grounded,
            "dead": dead,
            "finished": finished,
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info
