from stable_baselines3.common.distributions import DiagGaussianDistribution, sum_independent_dims
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

import numpy as np
import torch as th
from typing import override

from utils import normalize_action


class MaskedDiagGaussian(DiagGaussianDistribution):
    """Gaussian over every action dim, but masked dims do not contribute to loss."""

    def __init__(self, action_dim: int, mask: th.Tensor):
        super().__init__(action_dim)

        self.mask = mask

    def _apply_mask(self, vector: th.Tensor) -> th.Tensor:
        return vector * self.mask.to(vector.device)

    @override
    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return sum_independent_dims(self._apply_mask(self.distribution.log_prob(actions)))

    @override
    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self._apply_mask(self.distribution.entropy()))

class MaskedMultiInputPolicy(MultiInputActorCriticPolicy):
    """Modifies action_dist to use masked gaussian to prevent fixed actions from affecting policy. \n
    Sets bias in model to the task's fixed action values, effectively setting that value (normalized)
    as the mean to sample around.
    """

    def __init__(self, *args, fixed_actions: dict[int, float], **kwargs):
        super().__init__(*args, **kwargs)

        fixed_actions = {int(dim): float(value) for dim, value in fixed_actions.items()}

        # Mask action distribution
        action_dim = int(np.prod(self.action_space.shape))

        mask = th.ones(action_dim, dtype=th.float32, device=self.device)
        mask[list(fixed_actions)] = 0.0
        self.action_dist = MaskedDiagGaussian(action_dim, mask)

        # Set normalized bias according to fixed action values for the task
        scaled = np.zeros(action_dim)
        for dim, value in fixed_actions.items():
            scaled[dim] = value
        
        normalized = normalize_action(scaled)

        with th.no_grad():
            for dim in fixed_actions:
                self.action_net.bias[dim] = float(normalized[dim])