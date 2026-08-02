import time
from pathlib import Path

import mujoco
import mujoco.viewer

from domain_rand import DomainRandomizer

MODEL_PATH = str(Path(__file__).parent / "models" / "scene.xml")

m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

DomainRandomizer(m).randomize()

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(m, d)
        viewer.sync()

        # Real-time pacing
        elapsed = time.time() - step_start
        if elapsed < m.opt.timestep:
            time.sleep(m.opt.timestep - elapsed)