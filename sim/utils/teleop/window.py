"""GLFW window, cameras, rendering, and keyboard state."""

import glfw
import mujoco

WIDTH, HEIGHT = 1200, 900
TITLE = "Cozmo Teleop"

TRACK_DISTANCE = 0.45
TRACK_ELEVATION = -25.0
TRACK_AZIMUTH = 135.0

DEFAULT_CAM = "cozmo_chase"
CAMERA_HIDE_GROUP = 1


class Window:
    def __init__(self, m, d):
        self.m, self.d = m, d

        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        self.win = glfw.create_window(WIDTH, HEIGHT, TITLE, None, None)
        glfw.make_context_current(self.win)
        glfw.swap_interval(0)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.scn = mujoco.MjvScene(m, maxgeom=10000)
        self.ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)

        self.track_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cozmo")

        # Tracking and all defined <camera>
        self.cam_list = ["track"] + [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"cam{i}"
            for i in range(m.ncam)
        ]
        self.cam_idx = 0
        self.set_cam(self.cam_list.index(DEFAULT_CAM) if DEFAULT_CAM in self.cam_list else 0)

        self.held = set()
        self.paused = False
        self.on_reset = None
        self.on_randomize = None

        glfw.set_key_callback(self.win, self._on_key)

    def set_cam(self, i):
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

        self.opt.geomgroup[CAMERA_HIDE_GROUP] = self.cam_list[self.cam_idx] != "cozmo_cam"

    def _on_key(self, win, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.held.add(key)
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_SPACE:
                self.paused = not self.paused
            elif key == glfw.KEY_R:
                if self.on_reset:
                    self.on_reset()
            elif key == glfw.KEY_T:
                if self.on_randomize:
                    self.on_randomize()
            elif key == glfw.KEY_RIGHT_BRACKET:
                self.set_cam(self.cam_idx + 1)
            elif key == glfw.KEY_LEFT_BRACKET:
                self.set_cam(self.cam_idx - 1)
        elif action == glfw.RELEASE:
            self.held.discard(key)

    def should_close(self):
        return glfw.window_should_close(self.win)

    def render(self, status_lines=()):
        fw, fh = glfw.get_framebuffer_size(self.win)
        viewport = mujoco.MjrRect(0, 0, fw, fh)
        mujoco.mjv_updateScene(self.m, self.d, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        mujoco.mjr_render(viewport, self.scn, self.ctx)

        lines = list(status_lines) + [f"Camera  {self.cam_list[self.cam_idx]}"]
        if self.paused:
            lines.append("PAUSED")
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, "\n".join(lines), "", self.ctx)

        glfw.swap_buffers(self.win)
        glfw.poll_events()

    def close(self):
        glfw.terminate()