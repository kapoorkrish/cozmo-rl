"""Configure task and hyperparameters."""

from train.tasks import *

# Specify task to train and optional policy to initialize training from
TASK = RollCube()
INIT_POLICY = None

# Specify whether to run BC or PPO policy when running run_policy
POLICY_TYPE = "ppo"


# Train: Behavioral Cloning (bc)
BC_LR = 3e-4
BC_EPOCHS = 30
BC_BATCH_SIZE = 64

# Train: Proximal Policy Optimization (ppo)
PPO_LR = 1.5e-4
KL_PER_DIM = 0.04
# KL_PER_DIM = None
TIMESTEPS = 500_000
NUM_ENVS = 4
ROLLOUT = 8192
PPO_BATCH_SIZE = 256
VIDEO_EVERY = 50
