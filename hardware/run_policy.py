import time
import numpy as np
import pycozmo
from stable_baselines3 import PPO

from hardware.utils.observer import CozmoObserver

from config import TASK
from utils import HZ, WHEEL_RADIUS

MODEL = f"./models/bc/{TASK!s}"
MAX_STEPS = 500

def _wheel_rad_to_mm(rad):
    return rad * WHEEL_RADIUS

def _lift_rad_to_mm(rad):
    return 45.0 + 66.0 * np.sin(rad)

def _apply_action(cli, action) -> None:
    lwheel, rwheel, lift, head = TASK.map_action(action)

    cli.drive_wheels(lwheel_speed=_wheel_rad_to_mm(lwheel), rwheel_speed=_wheel_rad_to_mm(rwheel))
    cli.set_lift_height(height=_lift_rad_to_mm(lift))
    cli.set_head_angle(angle=head)


model = PPO.load(MODEL)
observer = CozmoObserver()

with pycozmo.connect() as cli:  # type: ignore
    cli.add_handler(pycozmo.protocol_encoder.RobotState, observer._on_robot_state)
    cli.add_handler(pycozmo.protocol_encoder.ObjectAccel, observer._on_object_accel)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, observer._on_camera)
    cli.enable_camera(enable=True, color=False)

    observer.connect_cube(cli, pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE1)

    print("Waiting for streams...")
    while (obs := observer.get_obs()) is None:
        time.sleep(1 / HZ)

    input("Press Enter to run policy.")

    try:
        for step in range(MAX_STEPS):
            start = time.perf_counter()

            obs = observer.get_obs()
            action, _ = model.predict(obs, deterministic=True)
            _apply_action(cli, action)

            print(f"{step:4d} {np.array2string(action, precision=2)}")
            time.sleep(max(0.0, (1 / HZ) - (time.perf_counter() - start)))
        
    except KeyboardInterrupt:
        pass

    finally:
        cli.stop_all_motors()
