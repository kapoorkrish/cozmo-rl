import mujoco
import numpy as np
from typing import override

from train.tasks.task import Task
from utils import HZ

class RollCube(Task):
    """Cozmo should raise the lift, drive to a cube, and move lift down while reversing to roll it over."""
    
    # Max for one sec / frame rate = max for one step
    max_dist = 200.0 / HZ

    # Acceleration threshold as axis shifts to trigger success, with settle time
    tilt_threshold = 0.15
    settle_steps = 5

    # Weights for reward function
    distance_prog_mult = 0.2
    align_prog_mult = 2
    tilt_prog_mult = 10
    fork_contact_bonus = 7.5
    success_bonus = 50
    time_penalty = 0.01

    def _prefix_geom_ids(self, model, prefixes: tuple[str, ...]) -> set[int]:
        ids = set()

        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name and name.startswith(prefixes):
                ids.add(g)

        return ids

    def _distance_to_cube(self, state: dict[str, float]) -> float:
        return float(np.hypot(state["cube_x"] - state["pose_x"], state["cube_y"] - state["pose_y"]))
    
    def _cube_alignment(self, state: dict[str, float]) -> float:
        # Dot product of cozmo heading with cozmo to cube vector
        to_cube = np.array([state["cube_x"] - state["pose_x"], state["cube_y"] - state["pose_y"]])
        heading = np.array([np.cos(state["pose_angle"]), np.sin(state["pose_angle"])])

        return float(heading @ to_cube / (np.linalg.norm(to_cube) + 1e-6))

    def _cube_tilt(self, state: dict[str, float]) -> float:
        print(state["accel_z"] / 9.81)
        return np.clip(state["accel_z"] / 9.81, -1.0, 1.0)
    
    def _fork_on_cube(self, sim) -> bool:
        data = sim.data

        for i in range(data.ncon):
            collision = {data.contact[i].geom1, data.contact[i].geom2}
            if collision & self.fork_ids and collision & self.cube_top_ids:
                return True

        return False

    @override
    def get_fixed_actions(self):
        # Keep head position constant
        return {3: -0.15}

    @override
    def reset(self, sim):
        state = sim.get_raw_state()

        self.fork_ids = self._prefix_geom_ids(sim.model, ("fork_bottom"))
        self.cube_top_ids = self._prefix_geom_ids(sim.model, ("c1_cc_cap_"))

        self.fork_contacted = False
        self.tipped_steps = 0
        self.prev_distance = self._distance_to_cube(state)
        self.prev_align = self._cube_alignment(state)
        self.prev_tilt = self._cube_tilt(state)

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

        # Cube tilt progress, measured by z axis acceleration due to gravity
        tilt = self._cube_tilt(state)
        tilt_prog = self.prev_tilt - tilt
        self.prev_tilt = tilt

        reward = ((distance_prog / self.max_dist) * self.distance_prog_mult
                + align_prog                      * self.align_prog_mult
                + tilt_prog                       * self.tilt_prog_mult
                - self.time_penalty)

        # Bonus for fork bottom touching top of cube
        if not self.fork_contacted and self._fork_on_cube(sim):
            reward += self.fork_contact_bonus
            self.fork_contacted = True

        # Record number of steps cube is thought to be tipped
        self.tipped_steps = self.tipped_steps + 1 if tilt < self.tilt_threshold else 0

        # Bonus for cube tipped in a settled state
        if self.tipped_steps == self.settle_steps:
            reward += self.success_bonus

        return reward

    @override
    def do_terminate(self, sim):
        return self.tipped_steps >= self.settle_steps