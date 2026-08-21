"""Live side-by-side of the real and simulated camera, with a cube in view.

Esc to quit, R to resample the sim's lighting and camera response.
"""

import math
import time

import cv2
import mujoco
import numpy as np
import pycozmo

from hardware.utils.observer import CozmoObserver
from sim.simulation import CozmoSim
from utils import HZ

SCALE = 6
CUBE_DISTANCE = 0.15
CUBE_HEIGHT = 0.022
HEAD_ANGLE = 0.0


def place_cube(sim: CozmoSim) -> None:
    """Put the target cube straight ahead of Cozmo at a fixed distance."""
    x, y, th = sim.origin
    qpos = sim.data.joint(sim.target_joint).qpos

    qpos[0:3] = [x + CUBE_DISTANCE * math.cos(th), y + CUBE_DISTANCE * math.sin(th), CUBE_HEIGHT]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    mujoco.mj_forward(sim.model, sim.data)


def reset_sim(sim: CozmoSim) -> None:
    """New randomization draw with the cube in view and the head level."""
    sim.reset()
    place_cube(sim)
    sim.data.ctrl[sim.model.actuator("head_motor").id] = HEAD_ANGLE

    for _ in range(sim.frames.maxlen):
        sim.step_sim()


def stats(frames: np.ndarray) -> str:
    """Mean level, contrast, and edge energy of a frame stack."""
    f = frames.astype(np.float32)

    return f"mean {f.mean():5.1f}  std {f.std():5.1f}  edges {np.abs(np.diff(f, axis=-1)).mean():5.2f}"


def main():
    sim = CozmoSim()
    observer = CozmoObserver()

    with pycozmo.connect() as cli:  # type: ignore
        cli.add_handler(pycozmo.event.EvtNewRawCameraImage, observer._on_camera)
        cli.add_handler(pycozmo.protocol_encoder.RobotState, observer._on_robot_state)
        cli.enable_camera(enable=True, color=False)
        cli.set_head_light(True)
        cli.set_head_angle(HEAD_ANGLE)

        print("Waiting for stream...")
        while observer.get_obs() is None:
            time.sleep(1 / HZ)

        reset_sim(sim)
        print(f"Place a cube ~{CUBE_DISTANCE * 100:.0f} cm in front of Cozmo.")
        print("Esc to quit, R to resample sim.\n")

        while True:
            real = observer.get_obs()["vision"]
            sim.step_sim()
            sim_frames = sim.get_frames()

            strip = np.vstack([np.hstack(real), np.hstack(sim_frames)])
            cv2.imshow("real (top) vs sim (bottom)",
                       cv2.resize(strip, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST))

            print(f"real  {stats(real)}   |   sim  {stats(sim_frames)}", end="\r")

            key = cv2.waitKey(1)
            if key == 27:
                break
            if key == ord("r"):
                reset_sim(sim)

            time.sleep(1 / HZ)

        cli.stop_all_motors()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()