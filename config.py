"""Configure task and hyperparameters."""

from train.tasks import *

TASK = RollCube()
INIT_POLICY = "./models/bc/roll_cube"
POLICY_TYPE = "ppo"


# Train: Behavioral Cloning (bc)
BC_LR = 3e-4
BC_EPOCHS = 50
BC_BATCH_SIZE = 64

# Train: Proximal Policy Optimization (ppo)
PPO_LR = 1.5e-4
TARGET_KL = 0.04
# TARGET_KL = None
TIMESTEPS = 1_000_000
NUM_ENVS = 4
ROLLOUT = 8192
PPO_BATCH_SIZE = 256
VIDEO_EVERY = 50
