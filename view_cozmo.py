import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "assets/cozmo.xml"

m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # Drive in a slow arc
        # [left_front, left_rear, right_front, right_rear]
        d.ctrl[:] = [0.25, 0.25, 0.8, 0.8]

        mujoco.mj_step(m, d)
        viewer.sync()

        # Real-time pacing
        elapsed = time.time() - step_start
        if elapsed < m.opt.timestep:
            time.sleep(m.opt.timestep - elapsed)
