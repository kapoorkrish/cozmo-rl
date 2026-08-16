"""Build the scene with the specified number of cubes."""

import mujoco
from pathlib import Path

MODELS = Path("./sim/models").resolve()
SCENE = str(MODELS / "scene.xml")
CUBE = str(MODELS / "cube.xml")


def build_model(num_cubes=None):
    """Compile scene with n cubes attached."""
    n = 3 if num_cubes is None else num_cubes
    spec = mujoco.MjSpec.from_file(SCENE)

    for i in range(n):
        cube = mujoco.MjSpec.from_file(CUBE)
        spec.attach(cube, prefix=f"c{i + 1}_", frame=spec.worldbody.add_frame())

    return spec.compile()