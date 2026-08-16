"""
  W / S         Drive forward / back
  A / D         Turn left / right
  Shift         Precision mode
  Up / Down     Move head
  Left / Right  Move lift
  [ / ]         Cycle camera (chase / cozmo_cam / tracking)
  Space         Pause / resume
  R             Reset
  Esc           Quit
"""

import time

from sim.simulation import CozmoSim
from sim.utils.teleop.control import Control
from sim.utils.teleop.window import Window

from constants import HZ


sim = CozmoSim()
control = Control(sim.model, sim.data)

def reset():
    control.reset()
    sim.reset()

window = Window(sim.model, sim.data, reset)
sim.add_context(window, window.mjr_context)
reset()

print(__doc__)
next_wall = time.perf_counter()

while not window.should_close():
    if not window.paused:
        control.apply(window.held, 1 / HZ)
        sim.step_sim()

    window.render(control.status_lines())

    next_wall += (1 / HZ)
    slack = next_wall - time.perf_counter()
    if slack > 0:
        time.sleep(slack)
    else:
        next_wall = time.perf_counter()

window.close()