from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.save_util import load_from_zip_file

import os
from glob import glob

from train.utils.environment import CozmoEnv
from utils import CHECKPOINT_DIR
from config import NUM_ENVS


def make_checkpoint(name: str, freq: int = 15_000,) -> CheckpointCallback:
    """Makes a CheckpointCallback to save training checkpoints at a specified frequency."""
    return CheckpointCallback(save_freq=max(freq // NUM_ENVS, 1), save_path=CHECKPOINT_DIR, name_prefix=name)

def load_checkpoint(name: str, env: CozmoEnv) -> tuple[PPO | None, int]:
    """Returns (model, timesteps_done) or (None, 0) if no checkpoint exists."""
    checkpoints = glob(os.path.join(CHECKPOINT_DIR, f"{name}_*_steps.zip"))
    if not checkpoints:
        return None, 0

    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Resuming training from {latest}")

    model = PPO.load(latest, env=env)
    return model, model.num_timesteps

def load_policy(model: PPO, dir: str) -> None:
    """Copies the policy weights of a previously trained model into current model."""
    _, params, _ = load_from_zip_file(f"{dir}.zip", device=model.device)

    model.set_parameters({"policy": params["policy"]}, exact_match=False)
    print(f"Initializing policy from {dir}")
