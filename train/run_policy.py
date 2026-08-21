"""Runs trained policy in simulation to visualize performance."""

from stable_baselines3 import PPO
import mujoco.viewer

import time

from sim.simulation import CozmoSim
from train.utils.environment import CozmoEnv

from config import TASK
from utils import HZ

MODEL = f"./models/ppo/{TASK!s}"
NUM_EPISODES = 5


model = PPO.load(MODEL)
env = CozmoEnv(CozmoSim(), TASK)
env.sim.reset()

with mujoco.viewer.launch_passive(env.sim.model, env.sim.data) as viewer:
    viewer.cam.distance = 0.9
    
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        total, done = 0.0, False

        while viewer.is_running() and not done:
            start = time.perf_counter()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated

            viewer.sync()
            time.sleep(max(0.0, (1 / HZ) - (time.perf_counter() - start)))

        print(f"Episode {episode}")
        print(f"Reward: {total:.1f}\n")

        if not viewer.is_running():
            break
