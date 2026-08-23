from collections.abc import Callable

from pathlib import Path
import numpy as np
import time

from sim.simulation import CozmoSim
from config import TASK
from utils import HZ, normalize_action

DEMO_DIR = Path(f"./demos/{TASK!s}")


class Recorder:
    """Record states, visions, and actions in teleop demonstration episode,
    to be read later as (observation, action) pairs."""

    def __init__(self,
                 sim: CozmoSim,
                 reset: Callable[..., None],
                 out_dir: Path | str = DEMO_DIR):
        
        self.sim = sim
        self.reset = reset
        self.out_dir = Path(out_dir)

        self.seed = None
        self.states = []
        self.visions = []
        self.actions = []
        self.recording = False
        self.status = ""

    def toggle(self) -> None:
        """Save the episode in progress, or start a new one."""
        self.save() if self.recording else self.start()

    def start(self) -> None:
        """Start a new episode on a fresh seed."""
        seed = int(np.random.SeedSequence().entropy % (2 ** 32))
        self.reset(seed)

        self.seed = seed
        self.recording = True

    def capture(self) -> None:
        """Store the observation the action was chosen from, and the action itself."""
        if not self.recording:
            return

        self.states.append(self.sim.get_state())
        self.visions.append(self.sim.get_frames())
        self.actions.append(normalize_action(self.sim.data.ctrl[self.sim.act_ids]).astype(np.float32))

    def save(self) -> None:
        """Write episode trajectory to disk and stop recording."""
        if not self.recording:
            return
        if not self.actions:
            self.discard()
            return

        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"ep_{time.strftime('%Y%m%d_%H%M%S')}.npz"

        np.savez_compressed(
            path,
            state=np.stack(self.states),
            vision=np.stack(self.visions),
            action=np.stack(self.actions),
            seed=self.seed,
            num_cubes=len(self.sim.cube_joints),
            target=self.sim.target,
            hz=HZ,
        )

        self.status = f"Saved {path.name}  {len(self.actions)} steps"
        self._clear()

    def discard(self) -> None:
        """Discard the episode recording."""
        if self.recording:
            self.status = "Discarded"

        self._clear()

    def _clear(self) -> None:
        """Clear recording status and data."""
        self.recording = False
        self.states, self.visions, self.actions = [], [], []

    def status_lines(self) -> list[str]:
        """Display lines for teleop HUD."""
        if self.recording:
            return [f"REC {len(self.actions) / HZ:5.1f}s  seed {self.seed}"]

        return [self.status] if self.status else []