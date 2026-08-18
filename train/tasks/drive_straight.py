import numpy as np
import math
from typing import override

from train.tasks.task import Task
from utils import HZ

def _wrap_angle(angle: float) -> float:
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
    def reset(self, sim):
        state = sim.get_raw_state()

        self.pos_start = np.array([state["pose_x"], state["pose_y"]])
        self.pos_prev = self.pos_start.copy()
        self.angle_prev = state["pose_angle"]

    @override
    def reward(self, sim, action):
        state = sim.get_raw_state()

        pos = np.array([state["pose_x"], state["pose_y"]])
        pose_angle = state["pose_angle"]

        pos_diff = pos - self.pos_prev
        pos_angle_diff = pose_angle - self.pose_angle_prev

        # Distance traveled along heading, and turning deviation
        distance = float(pos_diff @ np.array([math.cos(self.pose_angle_prev), math.sin(self.pose_angle_prev)]))
        turn = abs(_wrap_angle(pos_angle_diff))

        self.pos_prev, self.pose_angle_prev = pos, pose_angle

        # Reward function
        return ((distance / self.max_distance)  * self.distance_mult
                - (turn / self.max_turn)        * self.turn_penalty
                - float(np.sum(action ** 2))    * self.action_penalty)

    @override
    def do_terminate(self, sim):
        state = sim.get_raw_state()
        displacement = np.array([state["pose_x"], state["pose_y"]]) - self.pos_start

        return float(np.linalg.norm(displacement)) >= 750.0
