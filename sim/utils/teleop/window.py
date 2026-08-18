from collections.abc import Callable, Sequence

import glfw
import mujoco

from sim.utils.teleop.recorder import Recorder

WIDTH, HEIGHT = 1200, 900
TITLE = "Cozmo Teleop"

TRACK_DISTANCE = 0.45
TRACK_ELEVATION = -25.0
TRACK_AZIMUTH = 135.0

DEFAULT_CAM = "cozmo_cam"


class Window:
    """GLFW window, cameras, rendering, and keyboard state."""

    def __init__(self,
                 model: mujoco.MjModel, data: mujoco.MjData,
                 recorder: Recorder,
                 on_reset: Callable[[], None] = lambda: None):
        
        self.model, self.data = model, data
        self.on_reset = on_reset
        self.recorder = recorder
        
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.default_window_hints()
        self.window = glfw.create_window(WIDTH, HEIGHT, TITLE, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        self.cam = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.option)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)

        # Simplify graphics for teleop window performance
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.mjr_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self.track_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cozmo")

        # Tracking and all defined <camera>
        self.cam_list = ["track"] + [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"cam{i}"
            for i in range(model.ncam)
        ]
        self.cam_idx = 0
        self.set_cam(self.cam_list.index(DEFAULT_CAM) if DEFAULT_CAM in self.cam_list else 0)

        self.held = set()

        glfw.set_key_callback(self.window, self._on_key)

    def set_cam(self, i: int) -> None:
        """Select a camera by index, and wrap if out of bounds."""
        self.cam_idx = i % len(self.cam_list)

        if self.cam_idx == 0:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = self.track_body
            self.cam.distance = TRACK_DISTANCE
            self.cam.elevation = TRACK_ELEVATION
            self.cam.azimuth = TRACK_AZIMUTH
        else:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = self.cam_idx - 1

        # Hide Cozmo head from obstructing its camera
        self.option.geomgroup[1] = self.cam_list[self.cam_idx] != "cozmo_cam"

    def _on_key(self, win, key: int, scancode: int, action: int, mods: int):
        """Track held keys and execute one-time actions based on key."""
        if action == glfw.PRESS:
            self.held.add(key)
            
            if key == glfw.KEY_Q:
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_R:
                self.on_reset()
            elif key == glfw.KEY_RIGHT_BRACKET:
                self.set_cam(self.cam_idx + 1)
            elif key == glfw.KEY_LEFT_BRACKET:
                self.set_cam(self.cam_idx - 1)
            elif key == glfw.KEY_SPACE:
                self.recorder.toggle()
            elif key == glfw.KEY_BACKSPACE:
                self.recorder.discard()

        elif action == glfw.RELEASE:
            self.held.discard(key)

    def should_close(self) -> bool:
        """Reports whether window close flag is on."""
        return glfw.window_should_close(self.window)

    def render(self, status_lines: Sequence[str] = ()) -> None:
        """Render one frame with a status overlay."""
        glfw.make_context_current(self.window)
        fw, fh = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, fw, fh)
        mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.mjr_context)

        lines = list(status_lines) + [f"Camera  {self.cam_list[self.cam_idx]}"]
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, "\n".join(lines), "", self.mjr_context)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self) -> None:
        """Close window."""
        glfw.terminate()