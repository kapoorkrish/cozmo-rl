import pycozmo

import numpy as np
import time
from collections import deque
import threading

from constants import VISION_DIM, STATE_MIN, STATE_MAX

CUBE_1G = 31

def _lift_mm_to_rad(mm):
    pivot_h = 45.0
    lift_arm_len = 66.0
    lift_h_clipped = np.clip(mm, 32.0, 92.0)
    
    return np.arcsin((lift_h_clipped - pivot_h) / lift_arm_len)


class CozmoObserver:
    def __init__(self):
        self.lock = threading.Lock()
        self.robot_state = None
        self.cube_accel = (0.0, 0.0, 0.0)
        self.frames = deque(maxlen=VISION_DIM[0])
        self.cube_id = None

    def _on_robot_state(self, cli, pkt) -> None:
        with self.lock:
            self.robot_state = pkt

    def _on_object_accel(self, cli, pkt) -> None:
        with self.lock:
            if pkt.object_id == self.cube_id:
                self.cube_accel = (pkt.accel_x, pkt.accel_y, pkt.accel_z)

    def _on_camera(self, cli, image) -> None:
        # Downsample by strides of 4
        frame = np.asarray(image.convert("L"))[::4, ::4].astype(np.uint8)
        with self.lock:
            self.frames.append(frame)

    def _find_cube(self, cli, cube: pycozmo.protocol_encoder.ObjectType, timeout: float) -> int:
        deadline = time.time() + timeout

        while True:
            for fid, obj in dict(cli.available_objects).items():
                if obj.object_type is cube:
                    return fid

            if time.time() > deadline:
                raise TimeoutError(f"Failed to find {cube.name}.")

            time.sleep(1.0)

    def _connected_id(self, cli, cube: pycozmo.protocol_encoder.ObjectType, timeout: float) -> int:
        deadline = time.time() + timeout

        while True:
            for obj_id, obj in dict(cli.connected_objects).items():
                if obj["object_type"] is cube:
                    return obj_id

            if time.time() > deadline:
                raise TimeoutError(f"{cube.name} connection failed.")

            time.sleep(0.1)

    def connect_cube(self, cli, cube: pycozmo.protocol_encoder.ObjectType, timeout: float = 10.0) -> None:
        factory_id = self._find_cube(cli, cube, timeout)
        cli.conn.send(pycozmo.protocol_encoder.ObjectConnect(factory_id=factory_id, connect=True))

        self.cube_id = self._connected_id(cli, cube, timeout)
        cli.conn.send(pycozmo.protocol_encoder.StreamObjectAccel(object_id=self.cube_id, enable=True))

    def get_obs(self) -> dict[str, np.ndarray]:
        """Returns state and vision observation vectors or None until all streams are live."""
        with self.lock:
            if self.robot_state is None or len(self.frames) < VISION_DIM[0]:
                return None
            
            robot = self.robot_state
            cube_accel = [a / CUBE_1G for a in self.cube_accel]
            images = np.stack(self.frames, axis=0)

        state = np.array(
            [
                robot.pose_x,
                robot.pose_y,
                robot.pose_angle_rad,
                robot.lwheel_speed_mmps,
                robot.rwheel_speed_mmps,
                _lift_mm_to_rad(robot.lift_height_mm),
                robot.head_angle_rad,
                *cube_accel,
            ],
            dtype=np.float32
        )
        
        return {
            "state": np.clip(state, STATE_MIN, STATE_MAX),
            "vision": images
        }