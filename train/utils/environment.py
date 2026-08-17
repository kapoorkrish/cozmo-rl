import gymnasium as gym
from gymnasium import spaces

import numpy as np

from sim.simulation import CozmoSim
from train.tasks.task import Task
from utils import HZ, STATE_MIN, STATE_MAX, VISION_DIM


class CozmoEnv(gym.Env):
    """Structures the Cozmo simulation as a Gym environment for training."""

    def __init__(self, sim: CozmoSim, task: Task):
        super().__init__()
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": HZ}
        self.render_mode = "rgb_array"

        self.sim = sim
        self.task = task

        self.action_space = spaces.Box(-1.0, 1.0, (4,))
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(STATE_MIN, STATE_MAX),
                "vision": spaces.Box(0, 255, VISION_DIM, dtype=np.uint8)
            }
        )

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "state": self.sim.get_state(),
            "vision": self.sim.get_frames()
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.sim.reset(seed)

        obs = self._get_obs()
        self.task.reset(self.sim)

        return obs, {}

    def step(self, action: np.ndarray):
        self.sim.apply(self.task.map_action(action))
        self.sim.step_sim()

        obs = self._get_obs()
        reward = self.task.reward(self.sim, action)
        terminated = self.task.do_terminate(self.sim)
        truncated = self.sim.step_count >= 500
        info = {"is_success": terminated}

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.get_video_frame()
