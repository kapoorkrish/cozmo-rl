import mujoco
import numpy as np
from typing import override

from train.tasks.task import Task
from utils import HZ


class TouchCube(Task):
    """Cozmo should align with a cube and drive into it."""

    # Max for one sec / frame rate = max for one step
    max_dist = 200.0 / HZ

    # Weights for reward function
    distance_prog_mult = 0.2
    align_prog_mult = 3
    success_bonus = 50
    time_penalty = 0.01

    def _geom_ids(self, model, body: str) -> set[int]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        return set(np.flatnonzero(model.geom_bodyid == bid).tolist())

    def _distance_to_cube(self, state: dict[str, float]) -> float:
        return float(np.hypot(state["cube_x"] - state["pose_x"], state["cube_y"] - state["pose_y"]))

    def _cube_alignment(self, state: dict[str, float]) -> float:
        # Dot product of cozmo heading with cozmo to cube vector
        to_cube = np.array([state["cube_x"] - state["pose_x"], state["cube_y"] - state["pose_y"]])
        heading = np.array([np.cos(state["pose_angle"]), np.sin(state["pose_angle"])])

        return float(heading @ to_cube / (np.linalg.norm(to_cube) + 1e-6))

    def _cube_touched(self, sim) -> bool:
        data = sim.data

        # True if fork geom and cube geoms are colliding
        for i in range(data.ncon):
            collision = {data.contact[i].geom1, data.contact[i].geom2}
            if collision & self.fork_ids and collision & self.cube_ids:
                return True

        return False

    @override
    def get_fixed_actions(self) -> dict[int, float]:
        # Keep lift down and head constant
        return {2: -0.20, 3: -0.15}

    @override
    def reset(self, sim):
        state = sim.get_raw_state()

        self.fork_ids = self._geom_ids(sim.model, "fork")
        self.cube_ids = self._geom_ids(sim.model, "c1_cube")

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
        if self._cube_touched(sim):
            reward += self.success_bonus

        return reward

    @override
    def do_terminate(self, sim):
        return self._cube_touched(sim)