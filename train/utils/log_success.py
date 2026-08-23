from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

class LogSuccessRate(BaseCallback):
    """Logs success_rate from the is_success key in VecMonitor."""

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.model.ep_info_buffer:
            self.logger.record("rollout/success_rate",
                               safe_mean([ep["is_success"] for ep in self.model.ep_info_buffer]))