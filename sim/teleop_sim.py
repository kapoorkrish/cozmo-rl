"""
  W / S         Drive forward / back
  A / D         Turn left / right
  Shift         Precision mode
  Up / Down     Move head
  Left / Right  Move lift
  [ / ]         Cycle camera (chase / cozmo_cam / tracking)
  Space         Start demo / save demo
  Backspace     Discard demo
  R             Reset
  Q             Quit
"""

import time
import cv2
import numpy as np

from sim.simulation import CozmoSim
from sim.utils.teleop.control import Control
from sim.utils.teleop.recorder import Recorder
from sim.utils.teleop.window import Window

from utils import HZ

SHOW_FRAMES = True
SCALE = 6


sim = CozmoSim()
control = Control(sim.model, sim.data)

def reset(seed=None):
    recorder.discard()
    control.reset()
    sim.reset(seed)

recorder = Recorder(sim, reset)
window = Window(sim.model, sim.data, recorder, reset)
sim.add_context(window, window.mjr_context)
reset()
state = sim.get_raw_state()
start_accel = np.linalg.norm([state["accel_x"], state["accel_y"], state["accel_z"]])

if SHOW_FRAMES:
    cv2.namedWindow("obs")

print(__doc__)
next_wall = time.perf_counter()

while not window.should_close():
    control.apply(window.held, 1 / HZ)
    recorder.capture()
    sim.step_sim()

    state = sim.get_raw_state()
    print(state["accel_x"])
    print(state["accel_y"])
    print(state["accel_z"])
    print(f"Norm diff: {np.linalg.norm([state["accel_x"], state["accel_y"], state["accel_z"]]) - start_accel}")

    window.render(control.status_lines() + recorder.status_lines())

    if SHOW_FRAMES:
        strip = np.hstack(sim.get_frames())
        cv2.imshow("obs", cv2.resize(strip, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(1)

    next_wall += (1 / HZ)
    slack = next_wall - time.perf_counter()
    if slack > 0:
        time.sleep(slack)
    else:
        next_wall = time.perf_counter()

window.close()
cv2.destroyAllWindows()