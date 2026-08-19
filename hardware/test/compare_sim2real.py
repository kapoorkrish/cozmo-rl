"""Drive an identical action sequence on real Cozmo hardware and in the MuJoCo sim,
and measure how far the resulting observations diverge.
"""

import time

import numpy as np
import pycozmo

from hardware.utils.observer import CozmoObserver
from sim.simulation import CozmoSim
from config import TASK
from utils import HZ, STATE_MIN, STATE_MAX, lift_rad_to_mm, wheel_rad_to_mm, denormalize_state

STATE_LABELS = [
    "pose_x", "pose_y", "pose_angle",
    "lwheel", "rwheel", "lift", "head",
    "accel_x", "accel_y", "accel_z",
]
ANGLE_IDX = STATE_LABELS.index("pose_angle")

def _apply_action(cli, scaled: np.ndarray) -> None:
    """Send actions to cozmo hardware."""
    lwheel, rwheel, lift, head = scaled

    cli.drive_wheels(lwheel_speed=wheel_rad_to_mm(lwheel), rwheel_speed=wheel_rad_to_mm(rwheel))
    cli.set_lift_height(height=lift_rad_to_mm(lift))
    cli.set_head_angle(angle=head)

def build_scripted_actions() -> np.ndarray:
    """Fixed maneuver exercising wheels, lift, and head independently."""
    def hold(action, seconds) -> np.ndarray:
        return np.tile(np.array(action, dtype=np.float32), (round(seconds * HZ), 1))

    n_sweep = round(2.0 * HZ)
    t = np.linspace(0.0, 2 * np.pi, n_sweep, dtype=np.float32)
    head_sweep = np.stack(
        [np.zeros(n_sweep, dtype=np.float32), np.zeros(n_sweep, dtype=np.float32),
         np.full(n_sweep, -1.0, dtype=np.float32), np.sin(t).astype(np.float32)],
        axis=1,
    )

    return np.concatenate([
        hold([0.5, 0.5, -1.0, 0.0], 2.0),    # drive forward
        hold([0.4, -0.4, -1.0, 0.0], 1.5),   # turn in place
        hold([0.0, 0.0, 1.0, 0.0], 1.0),     # lift up
        hold([0.0, 0.0, -1.0, 0.0], 1.0),    # lift down
        head_sweep,                          # head sweep
        hold([0.0, 0.0, -1.0, 0.0], 0.5),    # stop
    ], axis=0)


def main() -> None:
    scripted = build_scripted_actions()
    observer = CozmoObserver()
    sim = CozmoSim()

    real_log, sim_log = [], []

    with pycozmo.connect() as cli:  # type: ignore
        cli.add_handler(pycozmo.protocol_encoder.RobotState, observer._on_robot_state)
        cli.add_handler(pycozmo.protocol_encoder.ObjectAccel, observer._on_object_accel)
        cli.add_handler(pycozmo.event.EvtNewRawCameraImage, observer._on_camera)
        cli.enable_camera(enable=True, color=False)

        observer.connect_cube(cli, pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE1)

        print("Waiting for streams...")
        while observer.get_obs() is None:
            time.sleep(1 / HZ)

        sim.reset()
        input("Press Enter to start.")

        try:
            for action in scripted:
                start = time.perf_counter()

                scaled = TASK.map_action(action)
                _apply_action(cli, scaled)
                sim.apply(scaled)
                sim.step_sim()

                time.sleep(max(0.0, (1 / HZ) - (time.perf_counter() - start)))

                real_log.append(observer.get_obs()["state"])
                sim_log.append(sim.get_state())
        
        except KeyboardInterrupt:
            pass

        finally:
            cli.stop_all_motors()

    real_norm = np.stack(real_log)
    sim_norm = np.stack(sim_log)

    norm_diff = real_norm - sim_norm
    diff = denormalize_state(real_norm) - denormalize_state(sim_norm)

    # Rewrap heading so a crossing at +/-pi doesn't read as a 2pi error.
    diff[:, ANGLE_IDX] = np.arctan2(np.sin(diff[:, ANGLE_IDX]), np.cos(diff[:, ANGLE_IDX]))
    norm_diff[:, ANGLE_IDX] = diff[:, ANGLE_IDX] * 2.0 / (STATE_MAX[ANGLE_IDX] - STATE_MIN[ANGLE_IDX])

    print("\n--- Sim-to-real gap ---")
    print(f"{'':12s}  {'Physical units':>24s}   {'Normalized':>24s}")
    for label, bias, spread, nbias, nspread in zip(
        STATE_LABELS, diff.mean(axis=0), diff.std(axis=0),
        norm_diff.mean(axis=0), norm_diff.std(axis=0),
    ):
        print(f"{label:12s}  mean={bias: 9.3f}  std={spread:8.3f}   "
              f"mean={nbias: 9.3f}  std={nspread:8.3f}")


if __name__ == "__main__":
    main()