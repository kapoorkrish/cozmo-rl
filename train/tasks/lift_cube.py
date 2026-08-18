import numpy as np
from typing import override

from train.tasks.task import Task
from utils import HZ


class LiftCube(Task):
    """Cozmo should align with a cube, drive to it, and lift it off the ground."""

    # Max for one sec / frame rate = max for one step
    max_dist = 200.0 / HZ
    max_lift = 45.0

    # Weights for reward function
    distance_prog_mult = 0.2
    align_prog_mult = 2
    lift_prog_mult = 5
    success_bonus = 50
    time_penalty = 0.01

    def _distance_to_cube(self, state: dict[str, float]) -> float:
        return float(np.hypot(state["cube_x"] - state["pose_x"], state["cube_y"] - state["pose_y"]))

    def _cube_alignment(self, state: dict[str, float]) -> float:
        # Dot product of cozmo heading with cozmo to cube vector
        to_cube = np.array([state["cube_x"] - state["pose_x"], state["cube_y"] - state["pose_y"]])
        heading = np.array([np.cos(state["pose_angle"]), np.sin(state["pose_angle"])])

        return float(heading @ to_cube / (np.linalg.norm(to_cube) + 1e-6))

    def _cube_lift_height(self, state: dict[str, float]) -> float:
        return state["cube_z"] - self.start_cube_z

    @override
    def reset(self, sim):
        state = sim.get_raw_state()

        self.start_cube_z = state["cube_z"]

        self.prev_distance = self._distance_to_cube(state)
        self.prev_align = self._cube_alignment(state)
        self.prev_lift = 0.0

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

        # Cube lift progress
        lift = self._cube_lift_height(state)
        lift_prog = lift - self.prev_lift
        self.prev_lift = lift

        reward = ((distance_prog / self.max_dist) * self.distance_prog_mult
                + align_prog                      * self.align_prog_mult
                + (lift_prog / self.max_lift)     * self.lift_prog_mult
                - self.time_penalty)

        if lift > self.max_lift:
            reward += self.success_bonus

        return reward

    @override
    def do_terminate(self, sim):
        state = sim.get_raw_state()

        return self._cube_lift_height(state) > self.max_lift
