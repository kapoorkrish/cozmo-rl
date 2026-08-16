import numpy as np

# Refresh rate (sim & reality)
HZ = 30

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