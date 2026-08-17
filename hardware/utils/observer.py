import pycozmo

import numpy as np
from PIL import Image
import time
from collections import deque
import threading

from utils import VISION_DIM, WHEEL_RADIUS, normalize_state

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
        self.latest_frame = None
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
        """Keep only the newest frame. The stack is sampled on the control tick."""
        small = image.convert("L").resize((VISION_DIM[2], VISION_DIM[1]), Image.BOX)

        with self.lock:
            self.latest_frame = np.asarray(small, dtype=np.uint8)

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
        with self.lock:
            if self.robot_state is None or self.latest_frame is None:
                return None

            # One frame push per step, append copies until full
            self.frames.append(self.latest_frame)
            while len(self.frames) < self.frames.maxlen:
                self.frames.append(self.frames[-1])

            robot = self.robot_state
            cube_accel = [a / CUBE_1G for a in self.cube_accel]
            images = np.stack(self.frames, axis=0)

        state = np.array(
            [
                robot.pose_x,
                robot.pose_y,
                robot.pose_angle_rad,
                robot.lwheel_speed_mmps / WHEEL_RADIUS,
                robot.rwheel_speed_mmps / WHEEL_RADIUS,
                _lift_mm_to_rad(robot.lift_height_mm),
                robot.head_angle_rad,
                *cube_accel,
            ],
            dtype=np.float32
        )
        
        return {
            "state": normalize_state(state),
            "vision": images
        }