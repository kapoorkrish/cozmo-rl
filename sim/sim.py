import time

import mujoco
import mujoco.viewer

from utils.domain_rand import DomainRandomizer
from utils.world import build_model

m = build_model()
d = mujoco.MjData(m)

DomainRandomizer(m).randomize()

with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer.cam.distance = 0.9

    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(m, d)
        viewer.sync()

        # Real-time pacing
        elapsed = time.time() - step_start
        if elapsed < m.opt.timestep:
            time.sleep(m.opt.timestep - elapsed)