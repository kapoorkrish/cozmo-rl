"""Drive a scripted action sequence on real Cozmo hardware, replay it in the MuJoCo sim,
and measure how far the resulting observations diverge.
"""

import time

import numpy as np
import pycozmo

from hardware.utils.observer import CozmoObserver
from sim.simulation import CozmoSim
from utils import HZ, denormalize_action, denormalize_state, lift_rad_to_mm, wheel_rad_to_mm

NOMINAL_FRICTION = 1.2

STATE_LABELS = [
    "pose_x", "pose_y", "pose_sin", "pose_cos",
    "lwheel", "rwheel", "lift", "head",
    "accel_x", "accel_y", "accel_z",
]

# name, normalized action, seconds
PHASES = [
    ("settle",    [0.0,  0.0, -1.0,  0.0], 0.5),
    ("forward",   [0.5,  0.5, -1.0,  0.0], 3.0),
    ("pause",     [0.0,  0.0, -1.0,  0.0], 0.5),
    ("turn",      [0.4, -0.4, -1.0,  0.0], 2.0),
    ("pause2",    [0.0,  0.0, -1.0,  0.0], 0.5),
    ("lift_up",   [0.0,  0.0,  1.0,  0.0], 1.0),
    ("lift_down", [0.0,  0.0, -1.0,  0.0], 1.0),
    ("head_up",   [0.0,  0.0, -1.0,  1.0], 1.0),
    ("head_down", [0.0,  0.0, -1.0, -1.0], 1.0),
]


def _apply_action(cli, scaled: np.ndarray) -> None:
    """Send actions to cozmo hardware."""
    lwheel, rwheel, lift, head = scaled

    cli.drive_wheels(lwheel_speed=wheel_rad_to_mm(lwheel), rwheel_speed=wheel_rad_to_mm(rwheel))
    cli.set_lift_height(height=lift_rad_to_mm(lift))
    cli.set_head_angle(angle=head)


def build_scripted_actions() -> tuple[np.ndarray, dict[str, slice]]:
    """Scripted maneuver and the index span of each phase."""
    actions, spans, cursor = [], {}, 0

    for name, action, duration in PHASES:
        steps = round(duration * HZ)
        actions.append(np.tile(np.array(action, dtype=np.float32), (steps, 1)))
        spans[name] = slice(cursor, cursor + steps)
        cursor += steps

    return np.concatenate(actions, axis=0), spans


def heading_deg(norm: np.ndarray) -> np.ndarray:
    """Normalized state log -> continuous heading in degrees."""
    phys = denormalize_state(norm)

    return np.degrees(np.unwrap(np.arctan2(phys[:, 2], phys[:, 3])))


def record_real(cli, observer: CozmoObserver, scripted: np.ndarray) -> tuple[np.ndarray, float]:
    """Run the script on hardware at HZ. Returns the state log and achieved loop rate."""
    log, stamps = [], []

    try:
        next_wake = time.perf_counter()

        for action in scripted:
            _apply_action(cli, denormalize_action(action))

            # Observe after the action has had a full period to take effect, matching sim
            next_wake += 1 / HZ
            time.sleep(max(0.0, next_wake - time.perf_counter()))

            obs = observer.get_obs()
            if obs is None:
                raise RuntimeError("Lost robot stream mid-run.")

            log.append(obs["state"])
            stamps.append(time.perf_counter())

    except KeyboardInterrupt:
        pass

    finally:
        cli.stop_all_motors()

    return np.stack(log), (len(stamps) - 1) / (stamps[-1] - stamps[0])


def replay_sim(sim: CozmoSim, scripted: np.ndarray) -> np.ndarray:
    """Replay the script in sim offline, with friction pinned to nominal."""
    sim.reset()
    for g in [sim.randomizer.floor_geom] + sim.randomizer.wheel_geoms:
        sim.model.geom_friction[g, 0] = NOMINAL_FRICTION

    log = []
    for action in scripted:
        sim.apply(denormalize_action(action))
        sim.step_sim()
        log.append(sim.get_state())

    return np.stack(log)


def report(real_norm: np.ndarray, sim_norm: np.ndarray, spans: dict[str, slice], rate: float) -> None:
    """Print the totals a policy accumulates and the per-channel gap."""
    real_phys, sim_phys = denormalize_state(real_norm), denormalize_state(sim_norm)
    real_deg, sim_deg = heading_deg(real_norm), heading_deg(sim_norm)

    diff, norm_diff = real_phys - sim_phys, real_norm - sim_norm
    fwd, turn = spans["forward"], spans["turn"]

    # sin and cos are only interpretable together, as one angle
    heading_err = np.abs((real_deg - sim_deg + 180) % 360 - 180)

    print(f"\nloop rate  {rate:5.1f} Hz  (target {HZ})")

    print("\n--- Totals ---")
    print(f"forward travel  real {real_phys[fwd.stop - 1, 0] - real_phys[fwd.start, 0]:7.1f}   "
          f"sim {sim_phys[fwd.stop - 1, 0] - sim_phys[fwd.start, 0]:7.1f} mm")
    print(f"turn total      real {real_deg[turn.stop - 1] - real_deg[turn.start]:7.1f}   "
          f"sim {sim_deg[turn.stop - 1] - sim_deg[turn.start]:7.1f} deg")
    print(f"heading error   mean {heading_err.mean():7.1f}   max {heading_err.max():7.1f} deg")

    print("\n--- Per-channel gap ---")
    print(f"{'':12s} {'mean':>9s} {'std':>9s} {'max':>9s} {'range':>9s} {'norm mean':>10s}")
    for i, label in enumerate(STATE_LABELS):
        d = diff[:, i]
        print(f"{label:12s} {d.mean(): 9.3f} {d.std():9.3f} {np.abs(d).max():9.3f} "
              f"{np.ptp(real_phys[:, i]):9.3f} {norm_diff[:, i].mean(): 10.3f}")


def main() -> None:
    scripted, spans = build_scripted_actions()
    observer = CozmoObserver()
    sim = CozmoSim()

    with pycozmo.connect() as cli:  # type: ignore
        cli.add_handler(pycozmo.protocol_encoder.RobotState, observer._on_robot_state)
        cli.add_handler(pycozmo.protocol_encoder.ObjectAccel, observer._on_object_accel)
        cli.add_handler(pycozmo.event.EvtNewRawCameraImage, observer._on_camera)
        cli.enable_camera(enable=True, color=False)

        observer.connect_cube(cli, pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE1)

        print("Waiting for streams...")
        while observer.get_obs() is None:
            time.sleep(1 / HZ)

        input("Press Enter to start.")
        real_norm, rate = record_real(cli, observer, scripted)

    print("Replaying in sim...")
    sim_norm = replay_sim(sim, scripted[:len(real_norm)])

    report(real_norm, sim_norm, spans, rate)


if __name__ == "__main__":
    main()