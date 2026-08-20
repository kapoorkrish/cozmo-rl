import pycozmo
import time

from hardware.utils.observer import CozmoObserver
from utils import HZ

observer = CozmoObserver()

with pycozmo.connect() as cli: # type: ignore
    cli.add_handler(pycozmo.protocol_encoder.RobotState, observer._on_robot_state)
    cli.add_handler(pycozmo.protocol_encoder.ObjectAccel, observer._on_object_accel)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, observer._on_camera)
    cli.enable_camera(enable=True, color=False)
    cli.set_head_light(True)

    observer.connect_cube(cli, pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE1)

    while True:
        obs = observer.get_obs()

        if obs is not None:
            print(obs["state"])
        
        time.sleep(1 / HZ)