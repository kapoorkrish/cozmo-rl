import glob
import os
from pathlib import Path

import numpy as np
import mujoco
from mujoco import mj_name2id, mj_id2name
from PIL import Image, ImageOps

TEXTURE_DIR = str(Path(__file__).resolve().parents[1] / "assets" / "textures")
SURFACES = ("floor", "wall")
REPEAT = {"floor": (4.0, 12.0), "wall": (6.0, 14.0)}
TINT = 0.08

LIGHT_ON_PROB = 0.8
BRIGHTNESS = (0.4, 0.9)
WARMTH = 0.10
SPECULAR = (0.0, 0.4)
AMBIENT = (0.0, 0.08)
SHADOW_PROB = 0.6
HEADLIGHT_AMBIENT = (0.02, 0.20)
HEADLIGHT_DIFFUSE = (0.0, 0.25)

WHEELS = ("front_left_wheel", "front_right_wheel",
                "rear_left_wheel", "rear_right_wheel")
FRICTION = (0.8, 1.3)


class DomainRandomizer:
    def __init__(self, m, seed=None):
        self.m = m
        self.rng = np.random.default_rng(seed)

        self.tex = {k: mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, f"tex_{k}") for k in SURFACES}
        self.mat = {k: mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"{k}_mat") for k in SURFACES}
        self.files = {k: sorted(glob.glob(os.path.join(TEXTURE_DIR, k, "*.png"))) for k in SURFACES}

        floor = mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        ceil = mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ceiling")

        self.half = float(m.geom_size[floor, 0])
        self.height = float(m.geom_pos[ceil, 2]) if ceil >= 0 else 1.5

        self.floor_geom = floor
        self.wheel_geoms = [
            g for g in range(m.ngeom)
            if mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) in WHEELS
        ]

    # Textures
    def _write_texture(self, tid, path):
        m = self.m
        h, w = int(m.tex_height[tid]), int(m.tex_width[tid])
        nc = int(m.tex_nchannel[tid])
        nface = 6 if m.tex_type[tid] != mujoco.mjtTexture.mjTEXTURE_2D else 1
        img = ImageOps.fit(Image.open(path).convert("RGB"), (w, h // nface), Image.LANCZOS)
        img = np.asarray(img, dtype=np.uint8)

        if nface > 1:
            img = np.tile(img, (nface, 1, 1))
        if nc == 4:
            img = np.dstack([img, np.full(img.shape[:2] + (1,), 255, np.uint8)])

        adr = int(m.tex_adr[tid])
        m.tex_data[adr:adr + h * w * nc] = img.reshape(-1)

    def randomize_textures(self, ctx=None):
        """Pick a random texture per surface"""
        m, rng = self.m, self.rng

        for k in SURFACES:
            files = self.files[k]
            if files:
                self._write_texture(self.tex[k], files[rng.integers(len(files))])
                if ctx is not None:
                    mujoco.mjr_uploadTexture(m, ctx, self.tex[k])

            mid = self.mat[k]
            m.mat_texrepeat[mid] = rng.uniform(*REPEAT[k])
            m.mat_rgba[mid, :3] = np.clip(1 + rng.uniform(-TINT, TINT, 3), 0, 1)

    # Lighting
    def randomize_lights(self):
        """Move, recolor, and toggle every light"""
        m, rng = self.m, self.rng
        L, H = self.half, self.height

        on = rng.random(m.nlight) < LIGHT_ON_PROB
        if not on.any():
            on[rng.integers(m.nlight)] = True

        for i in range(m.nlight):
            m.light_active[i] = bool(on[i])
            if not on[i]:
                continue

            pos = np.array([rng.uniform(-L, L), rng.uniform(-L, L), rng.uniform(0.5 * H, 0.9 * H)])
            aim = np.array([rng.uniform(-L, L), rng.uniform(-L, L), 0.0]) - pos
            m.light_pos[i] = pos
            m.light_dir[i] = aim / np.linalg.norm(aim)

            w = rng.uniform(-WARMTH, WARMTH)

            m.light_diffuse[i] = np.clip(rng.uniform(*BRIGHTNESS) * np.array([1 + w, 1.0, 1 - w]), 0, 1)
            m.light_specular[i] = rng.uniform(*SPECULAR)
            m.light_ambient[i] = rng.uniform(*AMBIENT)
            m.light_castshadow[i] = bool(rng.random() < SHADOW_PROB)

        m.vis.headlight.ambient[:] = rng.uniform(*HEADLIGHT_AMBIENT)
        m.vis.headlight.diffuse[:] = rng.uniform(*HEADLIGHT_DIFFUSE)

    # Friction
    def randomize_friction(self):
        """Randomize friction between floor and wheels"""
        m, rng = self.m, self.rng
        mu = rng.uniform(*FRICTION)

        for g in [self.floor_geom] + self.wheel_geoms:
            m.geom_friction[g, 0] = mu

    # Randomize environment
    def randomize(self, ctx=None):
        self.randomize_textures(ctx)
        self.randomize_lights()
        self.randomize_friction()