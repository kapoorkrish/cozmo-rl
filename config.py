"""Configure task and hyperparameters."""

from train.tasks.lift_cube import LiftCube
from train.tasks.touch_cube import TouchCube

# Specify task to train and optional policy to initialize training from
TASK = TouchCube()
INIT_POLICY = None

# Train: Behavioral Cloning (bc)
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 3e-4

# Train: Proximal Policy Optimization (ppo)
TIMESTEPS = 500_000
NUM_ENVS = 4
ROLLOUT = 2048
VIDEO_EVERY = 50
