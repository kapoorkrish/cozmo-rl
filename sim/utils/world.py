"""Build the scene with the specified number of cubes."""

import mujoco
from pathlib import Path

MODELS = Path("./sim/models").resolve()
SCENE = str(MODELS / "scene.xml")
CUBE = str(MODELS / "cube.xml")


def build_model(num_cubes: int) -> mujoco.MjModel:
    """Compile scene with n cubes attached."""
    spec = mujoco.MjSpec.from_file(SCENE)

    for i in range(num_cubes):
        cube = mujoco.MjSpec.from_file(CUBE)
        spec.attach(cube, prefix=f"c{i + 1}_", frame=spec.worldbody.add_frame())

    return spec.compile()