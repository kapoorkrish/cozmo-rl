from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

import numpy as np
import re

from utils import ACTION_MIN, ACTION_MAX

if TYPE_CHECKING:
    from sim.simulation import CozmoSim


class Task(ABC):
    """Defines a task to learn in simulation."""

    def get_fixed_actions(self) -> dict[int, float]:
        """Defines actions to keep constant for the task. \n
        {action_id: action_value}"""
        return {}

    @final
    def __str__(self) -> str:
        """Class name in snake_case used for paths and model names."""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", type(self).__name__).lower()

    @final
    def map_action(self, action: np.ndarray) -> np.ndarray:
        """Maps actions from [-1,1] to task's action space"""
        action = np.clip(action, -1.0, 1.0)
        scaled = ACTION_MIN + (action + 1.0) * 0.5 * (ACTION_MAX - ACTION_MIN)

        # Apply the task's fixed actions
        for k, v in self.get_fixed_actions().items():
            scaled[k] = v

        return scaled

    # These methods should only be used during sim training
    @abstractmethod
    def reset(self, sim: CozmoSim) -> None:
        """Defines logic to reset task state for new episode."""

    @abstractmethod
    def reward(self, sim: CozmoSim, action: np.ndarray) -> float:
        """Defines reward function for the task."""

    @abstractmethod
    def do_terminate(self, sim: CozmoSim) -> bool:
        """Defines condition to terminate the episode (success or failure)."""
