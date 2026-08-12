import time
import mujoco.viewer

from sim.simulation import CozmoSim
from constants import HZ

sim = CozmoSim()
sim.reset()

with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
    viewer.cam.distance = 0.9

    while viewer.is_running():
        step_start = time.time()

        sim.step_sim()
        viewer.sync()

        elapsed = time.time() - step_start
        
        if elapsed < 1 / HZ:
            time.sleep(1 / HZ - elapsed)