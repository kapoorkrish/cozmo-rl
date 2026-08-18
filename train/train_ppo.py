"""PPO training in environment using reward function defined in selected task."""

from stable_baselines3 import PPO

from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.monitor import Monitor

from sim.simulation import CozmoSim
from train.utils.environment import CozmoEnv
from train.utils.checkpoint import load_checkpoint, make_checkpoint

from config import TASK, VIDEO_EVERY, TIMESTEPS

MODEL_NAME = str(TASK)


# Set up mujoco sim & gynasium environment wrapper
sim = CozmoSim()

env = CozmoEnv(sim, TASK)
env = RecordVideo(
    env,
    video_folder=f"./models/videos/{MODEL_NAME}",
    episode_trigger=lambda ep: ep % VIDEO_EVERY == 0
)
env = Monitor(env)

# Start training, using latest checkpoint if available
model, timesteps_done = load_checkpoint(MODEL_NAME, env)

if model is None:
    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        verbose=1,
    )

model.learn(
    total_timesteps=TIMESTEPS - timesteps_done,
    callback=make_checkpoint(MODEL_NAME),
    reset_num_timesteps=False,
)
model.save(f"./models/ppo/{MODEL_NAME}")