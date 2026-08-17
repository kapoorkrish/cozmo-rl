import mujoco

import math
import numpy as np
from collections import deque
from skimage.color import rgb2gray

from sim.utils.domain_rand import DomainRandomizer
from sim.utils.world import build_model
from utils import HZ, VISION_DIM, normalize_state

ACTUATORS = ("left_front_motor", "right_front_motor", "lift_motor", "head_motor")


class CozmoSim:
    """Serves as an API to interact with mujoco simulation."""

    def __init__(self, num_cubes=1, target=1, seed=None):
        self.model = build_model(num_cubes=num_cubes)
        self.data = mujoco.MjData(self.model)

        # Hide Cozmo head from obstructing its camera
        self.cam_option = mujoco.MjvOption()
        self.cam_option.geomgroup[1] = 0

        self.renderer = mujoco.Renderer(self.model, VISION_DIM[1], VISION_DIM[2])
        self.video_renderer = None

        self.randomizer = DomainRandomizer(self.model, self.data,
                                           [(self.renderer._gl_context, self.renderer._mjr_context)])
        self.seed(seed)

        self.target = f"c{target}_"

        self.cube_joints = list(self.randomizer.cube_joints)
        self.target_joint = next(j for j in self.cube_joints if j.startswith(self.target))

        self.act_ids = np.array([self.model.actuator(n).id for n in ACTUATORS])

        self.substeps = max(1, round(1.0 / (HZ * self.model.opt.timestep)))
        self.step_count = 0

        self.frames = deque(maxlen=VISION_DIM[0])

    def _push_frame(self) -> None:
        """Render Cozmo's camera into the frame stack."""
        self.renderer.update_scene(self.data, camera="cozmo_cam", scene_option=self.cam_option)
        grayscale = (rgb2gray(self.renderer.render()) * 255).astype(np.uint8)
        self.frames.append(grayscale)

        # Push frame copies until stack is full
        while len(self.frames) < self.frames.maxlen:
            self.frames.append(self.frames[-1])

    def add_context(self, gl_context, mjr_context) -> None:
        """Register a context to receive randomized textures (used for teleop & video rendering)."""
        self.randomizer.contexts.append((gl_context, mjr_context))

    def seed(self, seed=None) -> None:
        """Reseed the sim and the randomizer with independent streams."""
        sim_seed, rand_seed = np.random.SeedSequence(seed).spawn(2)
        self.rng = np.random.default_rng(sim_seed)
        self.randomizer.rng = np.random.default_rng(rand_seed)

    def reset(self, seed=None) -> None:
        """Reset mujoco sim with new randomization. A seed makes the episode reproducible."""
        if seed is not None:
            self.seed(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.randomizer.randomize()

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0

        self.frames.clear()
        self._push_frame()

    def apply(self, action: np.ndarray):
        """Apply action vector to sim."""
        self.data.ctrl[self.act_ids] = action

    def step_sim(self):
        """Take substeps according to refresh rate, equating to one timestep."""
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        self._push_frame()

    def get_raw_state(self) -> dict[str, float]:
        """State in physical units: mm, rad, g. \n
        This state includes privileged info present in simulation but not reality,
        which may only be used for rewards and termination.
        """
        d = self.data
        pose = d.sensor("pose").data
        xaxis = d.sensor("pose_xaxis").data
        accel = d.sensor(self.target + "cube_accel").data / 9.81
        cube = d.joint(self.target_joint).qpos

        return {
            "pose_x": pose[0] * 1000.0,
            "pose_y": pose[1] * 1000.0,
            "pose_angle": math.atan2(xaxis[1], xaxis[0]),
            "lwheel": d.sensor("lwheel_speed").data[0],
            "rwheel": d.sensor("rwheel_speed").data[0],
            "lift": d.sensor("lift_angle").data[0],
            "head": d.sensor("head_angle").data[0],
            "accel_x": accel[0],
            "accel_y": accel[1],
            "accel_z": accel[2],
            # Privileged sim info
            "cube_x": cube[0] * 1000.0,
            "cube_y": cube[1] * 1000.0,
            "cube_z": cube[2] * 1000.0,
            "step": self.step_count,
        }

    def get_state(self) -> np.ndarray:
        """Observation vector normalized to [-1, 1],
        with only sensor data available in real CozmoObserver."""
        state = self.get_raw_state()

        return normalize_state(
            np.array(
                [
                    state["pose_x"],
                    state["pose_y"],
                    state["pose_angle"],
                    state["lwheel"],
                    state["rwheel"],
                    state["lift"],
                    state["head"],
                    state["accel_x"],
                    state["accel_y"],
                    state["accel_z"],
                ],
                dtype=np.float32
            )
        )

    def get_frames(self) -> np.ndarray:
        """Stacked grayscale vision observation, oldest to newest."""
        return np.stack(self.frames)

    def get_video_frame(self) -> np.ndarray:
        """Get RGB 3rd person image for rendering video"""
        if self.video_renderer is None:
            self.video_renderer = mujoco.Renderer(self.model, 240, 320)
            self.add_context(self.video_renderer._gl_context, self.video_renderer._mjr_context)
        
        self.video_renderer.update_scene(self.data, camera="cozmo_chase")
        return self.video_renderer.render()