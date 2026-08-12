"""Build a randomized world with the specified number of cubes"""

from pathlib import Path

import numpy as np
import mujoco

MODELS = Path(__file__).resolve().parents[1] / "models"
SCENE = str(MODELS / "scene.xml")
CUBE = str(MODELS / "cube.xml")
CUBE_TEXTURES = ("cube/cube1.png", "cube/cube2.png", "cube/cube3.png")

CUBE_SPAWN_RADIUS = (0.15, 0.40)
SPAWN_GAP = 0.09


def get_spawns(rng, n, radius=CUBE_SPAWN_RADIUS):
    out = []
    while len(out) < n:
        r, th = rng.uniform(*radius), rng.uniform(0, 2 * np.pi)
        p = np.array([r * np.cos(th), r * np.sin(th)])
        
        if all(np.linalg.norm(p - q) > SPAWN_GAP for q in out):
            out.append(p)

    return out

def build_model(num_cubes=None, seed=None):
    """Randomly draw n cubes"""
    rng = np.random.default_rng(seed)
    n = len(CUBE_TEXTURES) if num_cubes is None else min(num_cubes, len(CUBE_TEXTURES))

    spec = mujoco.MjSpec.from_file(SCENE)
    picks = rng.choice(len(CUBE_TEXTURES), n, replace=False)

    for i, (k, pos) in enumerate(zip(picks, get_spawns(rng, n, CUBE_SPAWN_RADIUS))):
        cube = mujoco.MjSpec.from_file(CUBE)
        cube.textures[0].file = CUBE_TEXTURES[k]

        yaw = rng.uniform(0, 2 * np.pi)
        frame = spec.worldbody.add_frame(
            pos=[pos[0], pos[1], 0],
            quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        
        spec.attach(cube, prefix=f"c{i + 1}_", frame=frame)

    return spec.compile()