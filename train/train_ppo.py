"""PPO training in environment using reward function defined in selected task."""

import os

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from sim.simulation import CozmoSim
from train.utils.load_model import load_checkpoint, load_policy, make_checkpoint
from train.utils.environment import CozmoEnv
from train.utils.mask_policy import MaskedMultiInputPolicy

from utils import PPO_DIR
from config import INIT_POLICY, NUM_ENVS, ROLLOUT, TASK, TIMESTEPS, VIDEO_EVERY

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
    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)]))
    model, timesteps_done = load_checkpoint(MODEL_NAME, env)

    if model is None:
        model = PPO(
            MaskedMultiInputPolicy,
            env,
            policy_kwargs={"action_mask": TASK.get_action_mask()},
            n_steps=ROLLOUT // NUM_ENVS,
            batch_size=64,
            learning_rate=3e-4,
            verbose=1,
        )

        if INIT_POLICY:
            load_policy(model, INIT_POLICY)

    model.learn(
        total_timesteps=TIMESTEPS - timesteps_done,
        callback=make_checkpoint(MODEL_NAME),
        reset_num_timesteps=False,
    )
    model.save(os.path.join(PPO_DIR, MODEL_NAME))