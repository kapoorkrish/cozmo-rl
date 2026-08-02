"""Drive, lift and head control."""

import numpy as np
import glfw
import mujoco

CONTROL_HZ = 30.0
DT_CTRL = 1.0 / CONTROL_HZ

WHEEL_R_MM = 13.14
MAX_SPEED_MMPS = 200.0
MAX_TURN_MMPS = 120.0
ACCEL_MMPS2 = 600.0
DECEL_MMPS2 = 900.0
PRECISION_SCALE = 0.3

HEAD_RATE = 1.0
LIFT_RATE = 1.2

def _actuator_id(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def _ramp(cur, tgt, dt, scale):
    rate = ACCEL_MMPS2 if abs(tgt) > abs(cur) else DECEL_MMPS2
    step = rate * scale * dt
    
    return cur + float(np.clip(tgt - cur, -step, step))

class Control:
    def __init__(self, m, d):
        self.m, self.d = m, d

        self.a_lwheel = _actuator_id(m, "left_front_motor")
        self.a_rwheel = _actuator_id(m, "right_front_motor")
        self.a_head = _actuator_id(m, "head_motor")
        self.a_lift = _actuator_id(m, "lift_motor")

        self.head_lo, self.head_hi = m.actuator_ctrlrange[self.a_head]
        self.lift_lo, self.lift_hi = m.actuator_ctrlrange[self.a_lift]

        self.n_substeps = max(1, round(DT_CTRL / m.opt.timestep))
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)

        self.v_cur = self.w_cur = 0.0
        self.head_tgt = self.lift_tgt = 0.0
        self.left = self.right = 0.0
        self.d.ctrl[:] = 0.0

    def apply(self, held, dt):
        d = self.d
        scale = PRECISION_SCALE if glfw.KEY_LEFT_SHIFT in held else 1.0

        v_target = ((glfw.KEY_W in held) - (glfw.KEY_S in held)) * MAX_SPEED_MMPS * scale
        w_target = ((glfw.KEY_D in held) - (glfw.KEY_A in held)) * MAX_TURN_MMPS * scale

        self.v_cur = _ramp(self.v_cur, v_target, dt, scale)
        self.w_cur = _ramp(self.w_cur, w_target, dt, scale)

        left, right = self.v_cur + self.w_cur, self.v_cur - self.w_cur
        peak = max(abs(left), abs(right), MAX_SPEED_MMPS)
        if peak > MAX_SPEED_MMPS:
            k = MAX_SPEED_MMPS / peak
            left, right = left * k, right * k
        
        self.left, self.right = left, right

        d.ctrl[self.a_lwheel] = left / WHEEL_R_MM
        d.ctrl[self.a_rwheel] = right / WHEEL_R_MM

        dh = ((glfw.KEY_UP in held) - (glfw.KEY_DOWN in held)) * HEAD_RATE * scale * dt
        dl = ((glfw.KEY_RIGHT in held) - (glfw.KEY_LEFT in held)) * LIFT_RATE * scale * dt

        self.head_tgt = float(np.clip(self.head_tgt + dh, self.head_lo, self.head_hi))
        self.lift_tgt = float(np.clip(self.lift_tgt + dl, self.lift_lo, self.lift_hi))

        d.ctrl[self.a_head] = self.head_tgt
        d.ctrl[self.a_lift] = self.lift_tgt

    def status_lines(self):
        return [
            f"L {self.left:7.1f} mm/s",
            f"R {self.right:7.1f} mm/s",
            f"Lift {self.lift_tgt: .3f} rad",
            f"Head {self.head_tgt: .3f} rad",
        ]