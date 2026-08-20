import glob
import os

import numpy as np
import mujoco
from mujoco import mj_name2id, mj_id2name
from PIL import Image, ImageOps

from utils import COZMO_SPAWN_RADIUS, CUBE_SPAWN_RADIUS, SPAWN_GAP, VISION_DIM

TEXTURE_DIR = "./sim/assets/textures"
SURFACES = ("floor", "wall")
REPEAT = {"floor": (4.0, 12.0), "wall": (6.0, 14.0)}
TINT = 0.08

LIGHT_ON_PROB = 0.6
MIN_LIGHTS_ON = 1
AIM_SPREAD = 0.5
BRIGHTNESS = (0.4, 0.9)
WARMTH = 0.10
SPECULAR = (0.0, 0.4)
AMBIENT = (0.0, 0.05)
SHADOW_PROB = 0.85
HEADLIGHT_AMBIENT = (0.03, 0.12)
HEADLIGHT_DIFFUSE = (0.05, 0.20)

# Camera setting randomness per episode
CAM_GAIN = (0.75, 1.25)
CAM_BIAS = (-25.0, 25.0)
CAM_GAMMA = (0.8, 1.3)
CAM_VIGNETTE = (0.0, 0.4)
CAM_FPN = (0.0, 4.0)
CAM_NOISE = (2.0, 10.0)
# Camera setting randomness per frame
CAM_FLICKER = 3.0

WHEELS = ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")
FRICTION = (1.1, 1.35)
WHEEL_SCALE = (0.97, 1.03)
WHEEL_BIAS = 0.02
ARMATURE_SCALE = (0.9, 1.1)


class DomainRandomizer:
    """Randomize the scene setting to mitigate overfitting in training."""

    def __init__(self,
                 model: mujoco.MjModel, data: mujoco.MjData,
                 contexts: tuple = (),
                 seed: int | None = None):

        self.model = model
        self.data = data
        self.rng = np.random.default_rng(seed)
        self.contexts = list(contexts)

        joint_names = [mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        self.cube_joints = sorted(n for n in joint_names if n and n.endswith("_cube_joint"))

        self.textures = {k: mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, f"{k}_texture") for k in SURFACES}
        self.materials = {k: mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, f"{k}_material") for k in SURFACES}
        self.surface_files = {k: sorted(glob.glob(os.path.join(TEXTURE_DIR, k, "*.png"))) for k in SURFACES}

        floor = mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        ceil = mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ceiling")

        self.half = float(model.geom_size[floor, 0])
        self.height = float(model.geom_pos[ceil, 2]) if ceil >= 0 else 1.5

        self.floor_geom = floor
        self.wheel_geoms = [
            g for g in range(model.ngeom)
            if mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) in WHEELS
        ]

        # Nominal drivetrain values, captured once so randomization never compounds
        self.left_geoms = {
            g for g in self.wheel_geoms
            if "_left_" in mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        }
        self.wheel_radii = {g: float(model.geom_size[g, 0]) for g in self.wheel_geoms}
        self.wheel_armature = {}

        for w in WHEELS:
            dof = int(model.joint(f"{w}_joint").dofadr[0])
            self.wheel_armature[dof] = float(model.dof_armature[dof])

        self.cube_geoms = [mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"c{i + 1}_cube_visual")
                           for i in range(len(self.cube_joints))]
        self.cube_mats = [[mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, f"c{i + 1}_cube{j + 1}_material")
                           for j in range(3)] for i in range(len(self.cube_joints))]

    # Textures
    def _set_texture(self, tid: int, path: str) -> None:
        """Load image into model texture and push to all contexts."""
        model = self.model
        h, w = int(model.tex_height[tid]), int(model.tex_width[tid])
        nc = int(model.tex_nchannel[tid])
        nface = 6 if model.tex_type[tid] != mujoco.mjtTexture.mjTEXTURE_2D else 1
        img = ImageOps.fit(Image.open(path).convert("RGB"), (w, h // nface), Image.LANCZOS)
        img = np.asarray(img, dtype=np.uint8)

        if nface > 1:
            img = np.tile(img, (nface, 1, 1))
        if nc == 4:
            img = np.dstack([img, np.full(img.shape[:2] + (1,), 255, np.uint8)])

        adr = int(model.tex_adr[tid])
        model.tex_data[adr:adr + h * w * nc] = img.reshape(-1)

        # Upload texture to all contexts
        for gl, mjr in self.contexts:
            gl.make_current()
            mujoco.mjr_uploadTexture(model, mjr, tid)

    def randomize_textures(self) -> None:
        """Pick a random texture per surface and distinct design per cube."""
        model, rng = self.model, self.rng

        # Floor and walls
        for k in SURFACES:
            files = self.surface_files[k]
            if files:
                self._set_texture(self.textures[k], files[rng.integers(len(files))])

            mid = self.materials[k]
            model.mat_texrepeat[mid] = rng.uniform(*REPEAT[k])
            model.mat_rgba[mid, :3] = np.clip(1 + rng.uniform(-TINT, TINT, 3), 0, 1)

        # Cubes
        picks = rng.choice(3, len(self.cube_geoms), replace=False)
        for gid, mats, j in zip(self.cube_geoms, self.cube_mats, picks):
            model.geom_matid[gid] = mats[j]

    # Lighting
    def randomize_lights(self) -> None:
        """Move, recolor, and toggle every light."""
        model, rng = self.model, self.rng
        L, H = self.half, self.height

        on = rng.random(model.nlight) < LIGHT_ON_PROB

        need = min(MIN_LIGHTS_ON, model.nlight)
        if on.sum() < need:
            off = np.flatnonzero(~on)
            on[rng.choice(off, need - on.sum(), replace=False)] = True

        for i in range(model.nlight):
            model.light_active[i] = bool(on[i])
            if not on[i]:
                continue

            pos = np.array([rng.uniform(-L, L), rng.uniform(-L, L), rng.uniform(0.5 * H, 0.9 * H)])
            target = np.array([rng.uniform(-L, L) * AIM_SPREAD, rng.uniform(-L, L) * AIM_SPREAD, 0.0])
            aim = target - pos

            model.light_pos[i] = pos
            model.light_dir[i] = aim / np.linalg.norm(aim)

            w = rng.uniform(-WARMTH, WARMTH)

            model.light_diffuse[i] = np.clip(rng.uniform(*BRIGHTNESS) * np.array([1 + w, 1.0, 1 - w]), 0, 1)
            model.light_specular[i] = rng.uniform(*SPECULAR)
            model.light_ambient[i] = rng.uniform(*AMBIENT)
            model.light_castshadow[i] = bool(rng.random() < SHADOW_PROB)

        model.vis.headlight.ambient[:] = rng.uniform(*HEADLIGHT_AMBIENT)
        model.vis.headlight.diffuse[:] = rng.uniform(*HEADLIGHT_DIFFUSE)

    # Camera
    def randomize_camera(self) -> None:
        """Resample camera response, held fixed for the episode."""
        rng = self.rng

        ramp = np.arange(256, dtype=np.float32) / 255.0
        self.cam_lut = np.clip(ramp ** rng.uniform(*CAM_GAMMA) * 255.0 * rng.uniform(*CAM_GAIN)
                               + rng.uniform(*CAM_BIAS), 0, 255).astype(np.uint8)

        h, w = VISION_DIM[1], VISION_DIM[2]
        ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]

        self.cam_gain_map = 1.0 - rng.uniform(*CAM_VIGNETTE) * (ys ** 2 + xs ** 2) / 2.0
        self.cam_fpn = rng.normal(0.0, rng.uniform(*CAM_FPN), (h, w)).astype(np.float32)
        self.cam_noise = rng.uniform(*CAM_NOISE)

    def augment(self, frame: np.ndarray) -> np.ndarray:
        """Apply camera response and per-frame sensor noise to a rendered frame."""
        rng = self.rng

        out = self.cam_lut[frame] * self.cam_gain_map + self.cam_fpn
        out += rng.normal(0.0, self.cam_noise, out.shape) + rng.normal(0.0, CAM_FLICKER)

        return np.clip(out, 0, 255).astype(np.uint8)

    # Friction
    def randomize_friction(self) -> None:
        """Randomize friction between floor and wheels."""
        mu = self.rng.uniform(*FRICTION)

        for g in [self.floor_geom] + self.wheel_geoms:
            self.model.geom_friction[g, 0] = mu

    # Drivetrain
    def randomize_drivetrain(self) -> None:
        """Randomize wheel size, drive inertia, and left/right imbalance."""
        model, rng = self.model, self.rng

        scale, bias = rng.uniform(*WHEEL_SCALE), rng.uniform(-WHEEL_BIAS, WHEEL_BIAS)
        for g, radius in self.wheel_radii.items():
            side = 1.0 + bias if g in self.left_geoms else 1.0 - bias
            model.geom_size[g, 0] = radius * scale * side

        inertia = rng.uniform(*ARMATURE_SCALE)
        for dof, nominal in self.wheel_armature.items():
            model.dof_armature[dof] = nominal * inertia

    # Spawn locations and orientations
    def _get_spawns(self, n: int, radius: tuple[float, float]) -> list[np.ndarray]:
        """Get spawns for n entities within a given radius range."""
        out = []
        while len(out) < n:
            r, th = self.rng.uniform(*radius), self.rng.uniform(0, 2 * np.pi)
            p = np.array([r * np.cos(th), r * np.sin(th)])

            if all(np.linalg.norm(p - q) > SPAWN_GAP for q in out):
                out.append(p)

        return out

    def randomize_spawns(self) -> None:
        """Randomize positions and orientations of cozmo and cubes."""
        rng = self.rng
        spawns = (self._get_spawns(1, COZMO_SPAWN_RADIUS)
                  + self._get_spawns(len(self.cube_joints), CUBE_SPAWN_RADIUS))

        for name, pos in zip(("root", *self.cube_joints), spawns):
            yaw = rng.uniform(0, 2 * np.pi)
            qpos = self.data.joint(name).qpos
            qpos[0:2] = pos
            qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

    # Randomize environment
    def randomize(self) -> None:
        """Randomize various aspects of scene."""
        self.randomize_textures()
        self.randomize_lights()
        self.randomize_camera()
        self.randomize_friction()
        self.randomize_drivetrain()
        self.randomize_spawns()