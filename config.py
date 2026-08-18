"""Configure task and hyperparameters."""

from train.tasks.lift_cube import LiftCube

TASK = LiftCube()

# Train: Behavioral Cloning (bc)
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 3e-4

# Train: Proximal Policy Optimization (ppo)
VIDEO_EVERY = 50
TIMESTEPS = 250_000
