"""Constants and helpers used throughout codebase."""

import numpy as np

# Refresh rate (sim & reality)
HZ = 30

# Cozmo hardware constants
WHEEL_RADIUS = 13.14

# -- Action space --
# left speed, right speed, lift angle, head angle
ACTION_MIN = np.array([-15.22, -15.22, -0.20, -0.44])
ACTION_MAX = np.array([15.22, 15.22, 0.79, 0.78])

# -- Observation space --
# pose x, pose y, pose sin, pose cos
# left speed, right speed, lift angle, head angle,
# cube accel x, cube accel y, cube accel z
STATE_MIN = np.array(
    [-2400, -2400, -1.0, -1.0,
     -15.22, -15.22, -0.20, -0.44,
     -4.1, -4.1, -4.1]
)
STATE_MAX = np.array(
    [2400, 2400, 1.0, 1.0,
     15.22, 15.22, 0.79, 0.78,
     4.1, 4.1, 4.1]
)
# Camera stream resolution and downsampling dim (stack x height x width)
SENSOR_DIM = (240, 320)
VISION_DOWNSAMPLE = 4
VISION_DIM = (3, SENSOR_DIM[0] // VISION_DOWNSAMPLE, SENSOR_DIM[1] // VISION_DOWNSAMPLE)

# Spawn randomization
COZMO_SPAWN_RADIUS = (0.0, 0.15)
CUBE_SPAWN_RADIUS = (0.20, 0.40)
SPAWN_GAP = 0.09

PPO_DIR = "./models/ppo"
BC_DIR = "./models/bc"
CHECKPOINT_DIR = "./models/checkpoints"


def normalize_state(state: np.ndarray) -> np.ndarray:
    """Physical units state vector -> [-1, 1]."""
    clipped = np.clip(state, STATE_MIN, STATE_MAX)
    return (2.0 * (clipped - STATE_MIN) / (STATE_MAX - STATE_MIN) - 1.0).astype(np.float32)

def denormalize_state(state: np.ndarray) -> np.ndarray:
    """[-1, 1] -> physical units state vector."""
    return (state + 1.0) / 2.0 * (STATE_MAX - STATE_MIN) + STATE_MIN

def normalize_action(scaled: np.ndarray) -> np.ndarray:
    """Task action space -> [-1, 1]."""
    return np.clip(2.0 * (scaled - ACTION_MIN) / (ACTION_MAX - ACTION_MIN) - 1.0, -1.0, 1.0)

def denormalize_action(action: np.ndarray) -> np.ndarray:
    """[-1, 1] -> physical units action vector."""
    return (action + 1.0) / 2.0 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN


def lift_rad_to_mm(rad: float) -> float:
    """Convert lift angle in radians to height in mm for pycozmo command."""
    return 45.12 + 66.0 * np.sin(rad)

def wheel_rad_to_mm(rad: float) -> float:
    """Convert wheel angular velocity in rad/s to translational velocity in mm/s."""
    return rad * WHEEL_RADIUS


def downsample(frame: np.ndarray) -> np.ndarray:
    """Sensor resolution frame -> observation resolution by area averaging."""
    n = VISION_DOWNSAMPLE
    h, w = frame.shape

    return frame.reshape(h // n, n, w // n, n).mean(axis=(1, 3)).round().astype(np.uint8)