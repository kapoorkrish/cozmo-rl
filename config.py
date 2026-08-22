"""Configure task and hyperparameters."""

from train.tasks.lift_cube import LiftCube
from train.tasks.touch_cube import TouchCube

# Specify task to train and optional policy to initialize training from
TASK = TouchCube()
INIT_TASK = None

# Train: Behavioral Cloning (bc)
BC_LR = 3e-4
EPOCHS = 30
BATCH_SIZE = 64

# Train: Proximal Policy Optimization (ppo)
PPO_LR = 1.5e-4
TIMESTEPS = 500_000
NUM_ENVS = 4
ROLLOUT = 2048
VIDEO_EVERY = 50
