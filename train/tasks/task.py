from abc import ABC, abstractmethod
from typing import final

import numpy as np

from constants import ACTION_MIN, ACTION_MAX


class Task(ABC):
    """Defines a task to learn in simulation."""

    def get_fixed_actions(self) -> dict[int, float]:
        """Defines actions to keep constant for the task. \n
        {action_id: action_value}"""
        return {}

    @final
    def map_action(self, action: np.ndarray) -> np.ndarray:
        """Maps actions from [-1,1] to task's action space"""
        action = np.clip(action, -1.0, 1.0)
        scaled = ACTION_MIN + (action + 1.0) * 0.5 * (ACTION_MAX - ACTION_MIN)

        # Apply the task's fixed actions
        for k, v in self.get_fixed_actions().items():
            scaled[k] = v

        return scaled

    @abstractmethod
    def reset(self) -> None:
        """Defines logic to reset task state for new episode."""
        pass

    @abstractmethod
    def reward(self, action: np.ndarray) -> float:
        """Defines reward function for the task."""
        pass

    @abstractmethod
    def do_terminate(self) -> bool:
        """Defines condition to terminate the episode (success or failure)."""
        pass
