"""
  W / S         Drive forward / back
  A / D         Turn left / right
  Shift         Precision mode
  Up / Down     Move head
  Left / Right  Move lift
  [ / ]         Cycle camera (chase / cozmo_cam / tracking)
  T             New floor / wall textures
  Space         Pause / resume
  R             Reset
  Esc           Quit
"""

import time
from pathlib import Path

import mujoco

from teleop.control import Control, DT_CTRL
from teleop.window import Window
from domain_rand import DomainRandomizer

MODEL_PATH = str(Path(__file__).parent / "models" / "scene.xml")


def main():
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)

    control = Control(m, d)
    window = Window(m, d)
    rand = DomainRandomizer(m)

    window.on_reset = control.reset
    window.on_randomize = lambda: rand.randomize(window.ctx)
    rand.randomize(window.ctx)

    print(__doc__)
    next_wall = time.perf_counter()

    while not window.should_close():
        if not window.paused:
            control.apply(window.held, DT_CTRL)
            for _ in range(control.n_substeps):
                mujoco.mj_step(m, d)

        window.render(control.status_lines())

        next_wall += DT_CTRL
        slack = next_wall - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        else:
            next_wall = time.perf_counter()

    window.close()


if __name__ == "__main__":
    main()