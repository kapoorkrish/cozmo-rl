import pycozmo
import time

from hardware.utils.observer import CozmoObserver
from utils import HZ

observer = CozmoObserver()

with pycozmo.connect() as cli: # type: ignore
    cli.add_handler(pycozmo.protocol_encoder.RobotState, observer._on_robot_state)
    cli.add_handler(pycozmo.protocol_encoder.ObjectAccel, observer._on_object_accel)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, observer._on_camera)
    # cli.enable_camera(enable=True, color=False)

    observer.connect_cube(cli, pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE1)

    while True:
        time.sleep(1.5)

        obs = observer.get_obs()
        if obs is not None:
            print(obs["state"][5])
        
        cli.set_lift_height(height=92)
        time.sleep(1.5)

        obs = observer.get_obs()
        if obs is not None:
            print(obs["state"][5])
    
        cli.set_lift_height(height=35)
        time.sleep(1.5)
        
        time.sleep(1 / HZ)