# cozmo-rl

Sim-to-real reinforcement learning, demonstrated on the Anki Cozmo robot. Policies are trained in a MuJoCo
simulation and run directly on hardware via [`pycozmo`](https://pycozmo.readthedocs.io/en/stable/).

**Click to watch demo:**
[![Demo](demo-thumbnail.png)](https://youtu.be/F_28-Gwk0q0)

This repository is an entry point for self-learners, researchers, and robotics enthusiasts who want hands-on experience with sim-to-real training with affordable, real hardware. For those who don't own a Cozmo, the simulation scripts alone are enough to train and test policies virtually.

## Contents
- [Setup](#setup)
- [Configuration](#configuration)
- [Gym Environment](#gym-environment)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
- [Adding a New Task](#adding-a-new-task)
- [Training Pipeline](#training-pipeline)
  - [1. Record expert demos](#1-record-expert-demos)
  - [2. Train using Behavioral cloning (BC)](#2-train-using-behavioral-cloning-bc)
  - [3. Train using Proximal Policy Optimization (PPO)](#3-train-using-proximal-policy-optimization-ppo)
- [Running a Trained Policy](#running-a-trained-policy)
  - [Run in simulation](#run-in-simulation)
  - [Run on real hardware](#run-on-real-hardware)
- [Debugging Scripts](#debugging-scripts)
  - [hardware/test](#hardwaretest)
  - [sim](#sim)
- [Acknowledgements](#acknowledgements)

## Setup
1. Install a Python 3.12 virtual environment (PyCozmo raises errors on newer Python versions)
```bash
py -3.12 -m venv .venv
```
2. Activate the virtual environment
```
source .venv/Scripts/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

Hardware scripts require a Cozmo connected to its Wi-Fi network for [`pycozmo`](https://github.com/gasgiant/PyCozmo) to able to reach it.

## Configuration

Task, policy choice, and hyperparameters live in `config.py`:

```python
from train.tasks import *

TASK = RollCube()
INIT_POLICY = "./models/bc/roll_cube"
POLICY_TYPE = "ppo"
...
```

- **TASK**: Selects the task, which is used when training or running policies.

- **INIT_POLICY**: Sets the path of the policy you would like to warm-start PPO training from, or `None` if you would like to train PPO from scratch.

- **POLICY_TYPE**: Sets the type of policy you would like to run for the task, which is `"bc"` or `"ppo"`.

## Gym Environment
A custom gym environment was built for Cozmo, which respects the real hardware actions and observations available so that transfer learning is possible.

**Note:**
- Some conversions are required from sim to real to map MuJoCo joints to the PyCozmo API.
- Actions are normalized to [-1, 1] for training policies, and denormalized to their action ranges after inference.
- Currently, observations for only cube 1 is supported, not all three
### Action Space
---
- Left wheel velocity
  - Real: [-200, 200] mm/s
  - Sim: [-15.22, 15.22] rad/s

- Right wheel velocity
  - Real: [-200, 200] mm/s
  - Sim: [-15.22, 15.22] rad/s

- Lift angle [-0.2, 0.79] rad

- Head angle [-0.44, 0.78] rad

### Observation Space
---

#### Translational Position (mm)
- pose_x
- pose_y
#### Angular Position (rad)
- sin(pose_angle)
- cos(pose_angle)
#### Translational Velocity (mm/s)
- lwheel
- rwheel
#### Lift Position (rad)
- lift
#### Head Angle (rad)
- head
#### Cube Acceleration (mm/s^2)
- accel_x
- accel_y
- accel_z
#### Vision (pixels)
- Downsampled to 80 x 60 x 1
  - Stack of 3 images for temporal info

## Adding a New Task

A new task is created by defining a new Python file in `train/tasks`. `Task` is an abstract interface that should be implemented when defining new tasks.

- The following methods must be implemented:
```python
@abstractmethod
def reset(self, sim: CozmoSim) -> None:
    """Defines logic to reset task state for new episode."""

@abstractmethod
def reward(self, sim: CozmoSim, action: np.ndarray) -> float:
    """Defines reward function for the task."""

@abstractmethod
def do_terminate(self, sim: CozmoSim) -> bool:
    """Defines condition to terminate the episode (success or failure)."""
```
- The following method should be overriden if the task should pin an action value that does not need training to execute the task:
```python
def get_fixed_actions(self) -> dict[int, float]:
    """Defines actions to keep constant for the task. \n
    {action_id: action_value}"""
    return {}
```


- Ensure that your new task is imported in `train/tasks/__init__.py` so it can be accessed in `config.py`

**Note:**
- CozmoSim provides `get_raw_state()`, which offers privileged info about the cube state and step count, not available in reality. These can be used for defining reward conditions and the termination condition, but note that they are not sent to the policy during training.

  - E.g. TouchCube uses distance from the cube as a reward metric, but the policy must learn a correlation from just its sensors and vision.

See existing tasks for examples.

## Training Pipeline
1. Record expert demos for the task, controlling the robot in sim using `teleop_sim.py`.

2. Train from these demos via Behavioral Cloning (BC).
3. Train the final policy using Proximal Policy Optimization (PPO), warm-starting from the BC policy.

**Note:** It is not required to warm-start from BC; PPO can be used directly, but for complex tasks it is suggested to start with BC for easier training.

### 1. Record expert demos

```bash
python -m sim.teleop_sim
```

Drive the robot and save teleoperated demonstrations of the task using the following  controls:

```
W / S         Drive forward / back
A / D         Turn left / right
Shift         Precision mode
Up / Down     Move head
Left / Right  Move lift
[ / ]         Cycle camera (chase / cozmo_cam / tracking)
Space         Start demo / save demo
Backspace     Discard demo
R             Reset
Q             Quit
```

Demos save to `demos/<task_name>/*.npz`.

### 2. Train using Behavioral cloning (BC)

```bash
python -m train.train_bc
```

Trains a policy via [`imitation.BC`](https://imitation.readthedocs.io/en/latest/) using the expert demos for the configured task, saved to `models/bc/<task_name>.zip` as a PPO policy for warm-starting training.

Tune learning rate, epochs, and batch size in `config.py`.

### 3. Train using Proximal Policy Optimization (PPO)

```bash
python -m train.train_ppo
```

Trains a policy via [`SB3 PPO`](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html). If `INIT_POLICY` is set, the policy is warm-started from it before training.

- Checkpoints save every 15k steps by default to `models/checkpoints/`.
  - `train_ppo` resumes from the latest checkpoint automatically.

- Final policy saves to `models/ppo/<task_name>.zip`.

- Logs include `rollout/success_rate` (calculated from each task's `do_terminate`) alongside the usual SB3 metrics.

- Every `VIDEO_EVERY` episodes, an environment is recorded to `models/videos/<task_name>/`.

Hyperparameters for PPO can be tuned in `config.py`. Target KL is optional and can be set to `None`, but is recommended to avoid overly destructive gradient updates.

## Running a Trained Policy
### Run in simulation

```bash
python -m sim.run_policy
```

Loads `models/<POLICY_TYPE>/<task_name>.zip` and runs 5 episodes in the MuJoCo viewer.

Collect failure samples for the policy selected by `TASK` and `POLICY_TYPE` in
`config.py` (for example, `TASK = TouchCube()` and `POLICY_TYPE = "ppo"`):

```bash
python -m experiments.collect_failures
```

Defaults to 50 failures, seed 0, and a maximum of 1000 episodes. Optional arguments:
`--samples 50 --seed 0 --max-episodes 1000`. The model is loaded from
`models/<POLICY_TYPE>/<TASK>.zip`; failure uses the environment's `is_success` metric.
For TouchCube, failure means the 500-step limit was reached without fork contact.

All files go directly into `experiments/failure_samples/<TASK>/`, with no run, worker, or
sample subdirectories. Open `index.html` to see paired head-camera/world images
and labeled state vectors. Each `sample_001` prefix has `_initial.png`,
`_failure.png`, `.json` (states and seed), and `.npz` (exact policy observations
and trajectory). `summary.json` records the model hash and collection settings;
`episodes.json` and `samples.csv` list the outcomes and sample vectors. RGB images
precede sensor noise; NPZ vision arrays contain the actual policy inputs.
The HTML report and console explicitly show the total number of evaluated
episodes (successes plus failures). `summary.json` stores this as
`episodes_evaluated`, alongside `successes_observed` and `failures_collected`.

The report includes 2D state comparisons, with successes on the left and failures
on the right using identical axis limits. It also plots initial Cozmo and cube
positions together in the spawn frame, connecting each episode's starting pair.
Comparisons cover terminal robot x/y,
left/right wheel speeds, heading versus mean wheel speed, mean versus differential
wheel speed, and cube acceleration where it varies meaningfully. Initial cube
forward/lateral position and distance/bearing are plotted separately and labeled
as privileged simulator data. Fixed head/lift, redundant sine/cosine, and pairs
with negligible variation are omitted. Points represent episode endpoints, not
all trajectory frames; terminal outcomes do not establish advance predictors.

New collections save normalized and raw initial/final states for **every** episode
in `episodes.json` and build the plots when collection finishes. To rebuild plots
or add states to older collections:

```bash
python -m experiments.collect_failures --update-plots
```

Saved failure states are reused. Missing episode states are recovered by replaying
their seeds with the original model and task. Each replay must match the recorded
outcome, step count, and reward sum before its states are saved. Verified states
are checkpointed after each episode, so rerunning resumes missing replays; the
HTML plots refresh every ten replays and at completion. Coverage counts make
partial backfills explicit. Plot PNGs are saved beside `index.html`.

Each collection reports `initial_value` from the loaded PPO critic, evaluated
on the exact initial state **and vision** inputs. The final critic is not evaluated.
`cumulative_reward` is the actual undiscounted sum of trajectory rewards (also
stored as `return`). `discounted_return` sums the same rewards with the model's
discount factor, recorded as `critic_gamma` in `summary.json`. These metrics
appear in each sample JSON, `samples.csv`, `episodes.json`, and `index.html`.

To add or recompute metrics for an existing collection:

```bash
python -m experiments.collect_failures --update-metrics
```

This reads the saved failure endpoint observations and reward arrays in each NPZ
and regenerates the reports. It checks that the configured model's SHA-256 matches
the collection before updating anything; images and trajectory arrays are kept.
Legacy success entries without saved observations retain their original fields.
To record all three metrics uniformly for successes and failures, collect a fresh
set with `python -m experiments.collect_failures`. New collections record the metrics for
every episode directly.

The critic estimates discounted future reward; compare it with
`discounted_return`, rather than the undiscounted sum. The observed discounted
sum ends at the episode boundary and excludes estimated continuation rewards
when a timeout truncates the episode.
These are reward diagnostics for sim-to-real analysis; the reach-avoid value in
[the calibration paper, Eq. (1)](https://arxiv.org/pdf/2603.24894) uses target and
constraint functions and is a different trajectory label.

The same model, settings, and software environment reproduce the collection.
To replay one failed episode, use its JSON seed with `--samples 1 --seed <seed>`.
Move or clear the output directory before rerunning; existing samples are never
overwritten. Generated collections are ignored by Git.

### Run on real hardware

```bash
python -m hardware.run_policy
```

Connects to Cozmo via its Wi-Fi and streams robot, cube, and camera data to create observation vector via `CozmoObserver`. After pressing Enter, the policy in `models/<POLICY_TYPE>/<task_name>.zip` runs and writes the action vector to the Cozmo hardware live. Ctrl+C stops motors and disconnects cleanly.

## Debugging Scripts
### hardware/test

- `get_obs.py`
  - Print the live observation vector.
- `view_cam.py`
  - Show the downsampled frame stack the policy sees.
- `calibrate_vision.py`
  - Side-by-side of real camera feed with sim camera feed for tuning vision noise in `DomainRandomizer`.
- `calibrate_state.py`
  - Drive a scripted action sequence on real hardware, replay the same actions in sim, and report drift for each state value. This helps tune MuJoCo constants to match reality.

### sim
- `view_sim.py`
  - Show domain randomized simulation environment in default MuJoCo viewer.

## Acknowledgements
- Floor/wall textures are from [Poly Haven](https://polyhaven.com/).

- Meshes come from a mix of [nilseuropa/anki_description](https://github.com/nilseuropa/anki_description)
  and [RazuProject/og-cozmo-model](https://codeberg.org/RazuProject/og-cozmo-model).
  The Cozmo fork mesh from `RazuProject` was convex-hulled and edited in Blender for accurate fork collisions.

- The URDFs in [nilseuropa/anki_description](https://github.com/nilseuropa/anki_description) were converted and used as a reference for building the MuJoCo XMLs in `sim/models/`.

An interrupted collection can be continued with `python -m experiments.collect_failures --resume`
using the original model, seed, and sample target.
