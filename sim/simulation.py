import mujoco

import math
import numpy as np
from collections import deque
from skimage.color import rgb2gray

from sim.utils.domain_rand import DomainRandomizer
from sim.utils.world import build_model
from constants import HZ, VISION_DIM

ACTUATORS = ("left_front_motor", "right_front_motor", "lift_motor", "head_motor")
STATE_FIELDS = ("pose_x", "pose_y", "pose_angle",
                "lwheel", "rwheel", "lift", "head",
                "accel_x", "accel_y", "accel_z")


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
                                           [(self.renderer._gl_context, self.renderer._mjr_context)],
                                           seed=seed)
        self.rng = np.random.default_rng(seed)

        self.target = f"c{target}_"
        self.cube_joints = [f"c{i + 1}_cube_free" for i in range(num_cubes)]

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

    def reset(self) -> None:
        """Reset mujoco sim with new randomization."""
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

    def get_state(self) -> np.ndarray:
        """State observation vector: \n
        [pose x, pose y, pose angle,
        left speed, right speed, lift angle, head angle,
        cube accel x, cube accel y, cube accel z]
        """
        d = self.data
        pose = d.sensor("pose").data
        xaxis = d.sensor("pose_xaxis").data
        ax, ay, az = d.sensor(self.target + "cube_accel").data / 9.81

        return np.array(
            [
                pose[0] * 1000.0,
                pose[1] * 1000.0,
                math.atan2(xaxis[1], xaxis[0]),
                d.sensor("lwheel_speed").data[0],
                d.sensor("rwheel_speed").data[0],
                d.sensor("lift_angle").data[0],
                d.sensor("head_angle").data[0],
                ax,
                ay,
                az,
            ],
            dtype=np.float32
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
