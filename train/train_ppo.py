"""PPO training in environment using reward function defined in selected task."""

import os

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from sim.simulation import CozmoSim
from train.utils.environment import CozmoEnv
from train.utils.log_success import LogSuccessRate
from train.utils.load_model import load_checkpoint, load_policy, make_checkpoint

from utils import PPO_DIR
from config import *

MODEL_NAME = str(TASK)


def make_env(idx: int):
    def _init():
        env = CozmoEnv(CozmoSim(), TASK)

        # Only record env 0
        if idx == 0:
            env = RecordVideo(env, video_folder=f"./models/videos/{MODEL_NAME}",
                              episode_trigger=lambda ep: ep % VIDEO_EVERY == 0)
        return env

    return _init


if __name__ == "__main__":
    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)]), info_keywords=("is_success",))
    action_mask = TASK.get_action_mask()
    target_kl = KL_PER_DIM * float(sum(action_mask)) if KL_PER_DIM else None

    model, timesteps_done = load_checkpoint(MODEL_NAME, env)

    if model is None:
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=ROLLOUT // NUM_ENVS,
            batch_size=PPO_BATCH_SIZE,
            learning_rate=PPO_LR,
            target_kl=target_kl,
            verbose=1,
        )

        if INIT_POLICY:
            load_policy(model, INIT_POLICY)
    else:
        model.target_kl = target_kl

    model.learn(
        total_timesteps=TIMESTEPS - timesteps_done,
        callback=[make_checkpoint(MODEL_NAME), LogSuccessRate()],
        reset_num_timesteps=False,
    )
    model.save(os.path.join(PPO_DIR, MODEL_NAME))