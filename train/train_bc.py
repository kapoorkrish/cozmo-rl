"""Behavioral cloning on teleop demos, saved as PPO model for later fine-tuning."""

from pathlib import Path

import numpy as np
from imitation.algorithms import bc
from imitation.data.types import DictObs, Transitions
from stable_baselines3 import PPO

from sim.simulation import CozmoSim
from train.utils.environment import CozmoEnv

from utils import BC_DIR
from config import TASK, BC_EPOCHS, BC_BATCH_SIZE, BC_LR

MODEL_NAME = str(TASK)
DEMO_DIR = Path(f"./demos/{TASK!s}")


def merge(episodes):
    return DictObs({k: np.concatenate([e[k] for e in episodes]) for k in episodes[0]})

def load_demos(demo_dir):
    """Flatten demos into transitions. BC ignores next_obs/dones, but its collate requires them."""
    files = sorted(Path(demo_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No demos in {demo_dir}")

    obs, next_obs, acts, dones = [], [], [], []
    for f in files:
        d = np.load(f)
        ep = {"state": d["state"], "vision": d["vision"]}

        obs.append(ep)
        next_obs.append({k: np.concatenate([v[1:], v[-1:]]) for k, v in ep.items()})
        acts.append(d["action"])

        done = np.zeros(len(d["action"]), dtype=bool)
        done[-1] = True
        dones.append(done)

    acts = np.concatenate(acts)

    return Transitions(
        obs=merge(obs),
        acts=acts,
        infos=np.array([{}] * len(acts)),
        next_obs=merge(next_obs),
        dones=np.concatenate(dones),
    )


env = CozmoEnv(CozmoSim(), TASK)

model = PPO(
    "MultiInputPolicy",
    env,
    n_steps=2048,
    batch_size=BC_BATCH_SIZE,
    learning_rate=BC_LR,
    verbose=1,
)

demos = load_demos(DEMO_DIR)
print(f"{len(demos)} transitions")

# Trains model.policy in place
trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=demos,
    policy=model.policy,
    batch_size=BC_BATCH_SIZE,
    optimizer_kwargs={"lr": BC_LR},
    rng=np.random.default_rng(0),
)
trainer.train(n_epochs=BC_EPOCHS)

model.save(f"{BC_DIR}/{MODEL_NAME}")