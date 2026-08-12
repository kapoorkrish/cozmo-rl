import numpy as np
import math
from typing import override

from train.tasks.task import Task
from constants import HZ

def _wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class DriveStraight(Task):
    """Cozmo should drive straight forward."""

    # Max for one sec / frame rate = max for one step
    max_distance = 200.0 / HZ
    max_turn = 8.33 / HZ

    # Weights for reward function
    distance_mult = 1.0
    turn_penalty = 3.0
    action_penalty = 0.01

    @override
    def get_fixed_actions(self):
        # Keep head and lift positions constant
        return {2: -0.20, 3: 0.0}

    @override
    def reset(self, obs):
        state = obs["state"]

        self.pos_start = state[[0, 1]].copy()
        self.pos_prev = self.pos_start.copy()
        self.pose_angle_prev = float(state[2])

    @override
    def reward(self, obs, action):
        state = obs["state"]

        pos = np.array([state[0], state[1]])
        pose_angle = float(state[2])

        # Reward metrics
        pos_diff = pos - self.pos_prev

        distance = float(pos_diff @ np.array([math.cos(self.pose_angle_prev), math.sin(self.pose_angle_prev)]))
        turn = abs(_wrap_angle(pose_angle - self.pose_angle_prev))

        self.pos_prev, self.pose_angle_prev = pos, pose_angle

        # Reward function
        return ((distance / self.max_distance)  * self.distance_mult
                - (turn / self.max_turn)        * self.turn_penalty
                - float(np.sum(action ** 2))    * self.action_penalty)

    @override
    def do_terminate(self, obs):
        state = obs["state"]

        return float(np.linalg.norm(state[[0, 1]] - self.pos_start)) >= 750.0
