from stable_baselines3.common.distributions import DiagGaussianDistribution, sum_independent_dims
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

import numpy as np
import torch as th
from typing import override


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
    """Modifies action_dist to use masked gaussian to prevent fixed actions from affecting policy."""

    def __init__(self, *args, action_mask, **kwargs):
        super().__init__(*args, **kwargs)

        mask = th.as_tensor(action_mask, dtype=th.float32, device=self.device)
        self.action_dist = MaskedDiagGaussian(int(np.prod(self.action_space.shape)), mask)
