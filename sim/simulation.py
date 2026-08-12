import mujoco

import math
import numpy as np
from skimage.color import rgb2gray

from sim.utils.domain_rand import DomainRandomizer
from sim.utils.world import build_model, get_spawns
from constants import HZ

ACTUATORS = ("left_front_motor", "right_front_motor", "lift_motor", "head_motor")
STATE_FIELDS = ("pose_x", "pose_y", "pose_angle",
                "lwheel", "rwheel", "lift", "head",
                "accel_x", "accel_y", "accel_z")


class CozmoSim:
    def __init__(self, num_cubes=1, target=1, seed=None, width=320, height=240):
        self.model = build_model(num_cubes=num_cubes, seed=seed)
        self.data = mujoco.MjData(self.model)
        self.randomizer = DomainRandomizer(self.model, seed=seed)
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.rng = np.random.default_rng(seed)

        self.target = f"c{target}_"
        self.cube_joints = [f"c{i + 1}_cube_free" for i in range(num_cubes)]

        self.act_ids = np.array([self.model.actuator(n).id for n in ACTUATORS])
        self.wall_geoms = {
            g for g in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("wall_")
        }

        self.substeps = max(1, round(1.0 / (HZ * self.model.opt.timestep)))
        self.step_count = 0

    def reset(self) -> None:
        self.renderer._gl_context.make_current()
        self.randomizer.randomize(self.renderer._mjr_context)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize spawns and orientations of cubes and cozmo
        cozmo_radius = (0.0, 0.20)
        spawns = get_spawns(self.rng, 1, cozmo_radius) + get_spawns(self.rng, len(self.cube_joints))

        for name, pos in zip(("root", *self.cube_joints), spawns):
            yaw = self.rng.uniform(0, 2 * np.pi)
            qpos = self.data.joint(name).qpos
            qpos[0:2] = pos
            qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0

    def apply(self, action: np.ndarray):
        """Apply action vector to sim."""
        self.data.ctrl[self.act_ids] = action

    def step_sim(self):
        """Take substeps according to refresh rate, equating to one timestep."""
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

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

    def get_frame(self) -> np.ndarray:
        """Get grayscale image observation from Cozmo's camera"""
        self.renderer.update_scene(self.data, camera="cozmo_cam")
        rgb = self.renderer.render()
        return (rgb2gray(rgb) * 255).astype(np.uint8)

    def get_video_frame(self) -> np.ndarray:
        """Get RGB 3rd person image for rendering video"""
        self.renderer.update_scene(self.data, camera="cozmo_chase")
        return self.renderer.render()

    def did_hit_wall(self) -> bool:
        """Whether any collision occurred with the wall."""
        return any(
            self.data.contact[i].geom1 in self.wall_geoms
            or self.data.contact[i].geom2 in self.wall_geoms
            for i in range(self.data.ncon)
        )