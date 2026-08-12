import gymnasium as gym
from gymnasium import spaces

from collections import deque
import numpy as np

from sim.simulation import CozmoSim
from train.tasks.task import Task
from constants import HZ, STATE_MIN, STATE_MAX, VISION_DIM


class CozmoEnv(gym.Env):
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
                "vision": spaces.Box(0, 255, VISION_DIM, dtype=np.uint8),
            }
        )

        self.frames = deque(maxlen=VISION_DIM[0])

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "state": np.clip(self.sim.get_state(), STATE_MIN, STATE_MAX),
            "vision": np.stack(self.frames, axis=0)
        }

    def _push_frame(self) -> None:
        # Downsample by strides of 4
        frame = self.sim.get_frame()[::4, ::4]
        self.frames.append(frame.astype(np.uint8))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.sim.reset()
        self.frames.clear()
        self._push_frame()

        # Push frame copies until stack is full
        while len(self.frames) < VISION_DIM[0]:
            self.frames.append(self.frames[-1])

        self.task.reset(self._get_obs())

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.sim.apply(self.task.map_action(action))
        self.sim.step_sim()
        self._push_frame()

        reward = self.task.reward(self._get_obs(), action)
        terminated = self.task.do_terminate(self._get_obs())
        truncated = self.sim.step_count >= 500

        info = {"is_success": terminated}

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        return self.sim.get_video_frame()
