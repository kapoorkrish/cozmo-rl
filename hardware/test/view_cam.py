"""Show the frame stack the policy actually receives. Esc to quit."""

import time

import cv2
import numpy as np
import pycozmo

from hardware.utils.observer import CozmoObserver
from utils import HZ

SCALE = 6


observer = CozmoObserver()

with pycozmo.connect() as cli:  # type: ignore
    cli.add_handler(pycozmo.protocol_encoder.RobotState, observer._on_robot_state)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, observer._on_camera)
    cli.enable_camera(enable=True, color=False)
    cli.set_head_light(True)

    print("Waiting for streams...")
    while (obs := observer.get_obs()) is None:
        time.sleep(1 / HZ)

    while True:
        obs = observer.get_obs()

        strip = np.hstack(obs["vision"])
        cv2.imshow("cozmo_cam", cv2.resize(strip, None, fx=SCALE, fy=SCALE,
                                           interpolation=cv2.INTER_NEAREST))

        robot = observer.robot_state
        print(f"head {robot.head_angle_rad:+.2f} rad   lift {robot.lift_height_mm:5.1f} rad   "
              f"mean pixel {obs['vision'][-1].mean():5.1f}", end="\r")

        if cv2.waitKey(1) == 27:
            break
        
        time.sleep(1 / HZ)

cv2.destroyAllWindows()