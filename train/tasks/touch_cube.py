import numpy as np
from typing import override

from train.tasks.task import Task
from utils import HZ


class TouchCube(Task):
    """Cozmo should align with a cube and drive into it."""

    # Max for one sec / frame rate = max for one step
    max_dist = 200.0 / HZ

    # Cube accel deviation to count as a touch
    accel_threshold = 0.2

    # Weights for reward function
    distance_prog_mult = 0.2
    align_prog_mult = 2
    success_bonus = 50
    time_penalty = 0.01

    def _distance_to_cube(self, state: dict[str, float]) -> float:
        return float(np.hypot(state["cube_x"] - state["pose_x"], state["cube_y"] - state["pose_y"]))

    def _cube_alignment(self, state: dict[str, float]) -> float:
        # Dot product of cozmo heading with cozmo to cube vector
        to_cube = np.array([state["cube_x"] - state["pose_x"], state["cube_y"] - state["pose_y"]])
        heading = np.array([np.cos(state["pose_angle"]), np.sin(state["pose_angle"])])

        return float(heading @ to_cube / (np.linalg.norm(to_cube) + 1e-6))

    def _delta_accel(self, state: dict[str, float]) -> float:
        accel = np.array([state["accel_x"], state["accel_y"], state["accel_z"]])

        return float(np.linalg.norm(accel - self.start_accel))

    def _cube_touched(self, state: dict[str, float]) -> bool:
        return self._delta_accel(state) > self.accel_threshold

    @override
    def get_fixed_actions(self) -> dict[int, float]:
        # Keep lift down and head constant
        return {2: -0.20, 3: -0.15}

    @override
    def reset(self, sim):
        state = sim.get_raw_state()

        self.start_accel = np.array([state["accel_x"], state["accel_y"], state["accel_z"]])

        self.prev_distance = self._distance_to_cube(state)
        self.prev_align = self._cube_alignment(state)

    @override
    def reward(self, sim, action):
        state = sim.get_raw_state()

        # Distance to cube progress
        distance = self._distance_to_cube(state)
        distance_prog = self.prev_distance - distance
        self.prev_distance = distance

        # Cube alignment progress
        align = self._cube_alignment(state)
        align_prog = align - self.prev_align
        self.prev_align = align

        reward = ((distance_prog / self.max_dist) * self.distance_prog_mult
                + align_prog                      * self.align_prog_mult
                - self.time_penalty)

        # Big bonus for reaching objective
        if self._cube_touched(state):
            reward += self.success_bonus

        return reward

    @override
    def do_terminate(self, sim):
        return self._cube_touched(sim.get_raw_state())