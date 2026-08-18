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
# pose x, pose y, pose angle,
# left speed, right speed, lift angle, head angle,
# cube accel x, cube accel y, cube accel z
STATE_MIN = np.array(
    [-2400, -2400, -np.pi,
     -15.22, -15.22, -0.20, -0.44,
     -4.1, -4.1, -4.1]
)
STATE_MAX = np.array(
    [2400, 2400, np.pi,
     15.22, 15.22, 0.79, 0.78,
     4.1, 4.1, 4.1]
)
# Stack x height x width
VISION_DOWNSAMPLE = 4
VISION_DIM = (3, 240 // VISION_DOWNSAMPLE, 320 // VISION_DOWNSAMPLE)

# Spawn randomization
COZMO_SPAWN_RADIUS = (0.0, 0.15)
CUBE_SPAWN_RADIUS = (0.20, 0.40)
SPAWN_GAP = 0.09

def normalize_state(state: np.ndarray) -> np.ndarray:
    """Raw state vector -> [-1, 1]."""
    clipped = np.clip(state, STATE_MIN, STATE_MAX)
    return (2.0 * (clipped - STATE_MIN) / (STATE_MAX - STATE_MIN) - 1.0).astype(np.float32)

def normalize_action(scaled: np.ndarray) -> np.ndarray:
    """Inverse of Task.map_action: task action space -> [-1, 1]."""
    return np.clip(2.0 * (scaled - ACTION_MIN) / (ACTION_MAX - ACTION_MIN) - 1.0, -1.0, 1.0)
