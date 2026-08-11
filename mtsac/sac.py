"""The algorithms a config can select with its ``algo:`` key.

Plain SAC on a task set with a one-hot task id and one shared replay buffer already is
MT-SAC with a shared backbone, so it needs no subclass. ``MTSAC`` adds the parts that a
*shared* critic needs once the tasks differ in difficulty, each switchable on its own so
they can be measured separately.
"""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
from torch import nn
from torch.nn import functional as F

from mtsac.buffer import TaskWeightedReplayBuffer


class LayerNormCritic(ContinuousCritic):
    """A critic with LayerNorm after every hidden layer.

    ``Linear -> LayerNorm -> activation`` is what BRO, SimBa and CrossQ use, and what the
    Meta-World+ parameter-scaling work relies on. It is the standard defence against the
    value blow-up that plain SAC is prone to here, which is what makes a config transfer
    between task sets without being retuned.
    """

    def __init__(
        self, *args, net_arch: list[int], activation_fn: type[nn.Module] = nn.ReLU, **kwargs
    ) -> None:
        super().__init__(*args, net_arch=net_arch, activation_fn=activation_fn, **kwargs)
        action_dim = get_action_dim(self.action_space)
        self.q_networks = []
        for idx in range(self.n_critics):
            q_net = nn.Sequential(
                *create_mlp(
                    self.features_extractor.features_dim + action_dim,
                    1,
                    net_arch,
                    activation_fn,
                    post_linear_modules=[nn.LayerNorm],
                )
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class LayerNormSACPolicy(SACPolicy):
    """``SACPolicy`` whose critics carry LayerNorm; the actor is unchanged."""

    def make_critic(self, features_extractor=None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return LayerNormCritic(**critic_kwargs).to(self.device)


class MTSAC(SAC):
    """SAC with the multi-task parts of the critic made per-task.

    ``normalize_critic_loss`` divides each sample's critic error by a running estimate of
    its task's return scale. This is the measured problem on MT3: with gamma 0.99 a solved
    task is worth 10/(1-gamma) = 1000, so reach's Q sits near 970 while pick-place's sits
    near 5, and one shared critic trained on squared error cannot feel a 5 next to a 970 --
    pick-place is not so much starved of data as invisible in the loss. Scaling the loss
    instead of the targets (as PopArt does) keeps Q in real units, so the actor, the target
    computation and every logged number are unchanged, and there is no output-preservation
    problem when the statistics move.

    ``per_task_alpha`` gives every task its own entropy coefficient, so a solved task can
    stop exploring while an unsolved one carries on. This is the standard disentangled
    alpha, included to measure against the other two rather than instead of them.

    Both read the task from the one-hot that ``AppendTaskOneHot`` puts in the last
    ``num_tasks`` observation dimensions, so neither needs a change to the replay buffer.

    The running scales are not checkpointed; a resumed run re-estimates them within a few
    hundred gradient steps.
    """

    def __init__(
        self,
        *args,
        num_tasks: int = 1,
        per_task_alpha: bool = False,
        normalize_critic_loss: bool = False,
        scale_momentum: float = 1e-3,
        min_value_scale: float = 1e-2,
        **kwargs,
    ) -> None:
        self.num_tasks = num_tasks
        self.per_task_alpha = per_task_alpha
        self.normalize_critic_loss = normalize_critic_loss
        self.scale_momentum = scale_momentum
        self.min_value_scale = min_value_scale
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.per_task_alpha and self.log_ent_coef is not None:
            start = self.log_ent_coef.detach().clone().reshape(1)
            self.log_ent_coef = start.repeat(self.num_tasks).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        self.value_scale = th.ones(self.num_tasks, device=self.device)

    def _task_index(self, observations: th.Tensor) -> th.Tensor:
        """Recover each sample's task from the one-hot block at the end of the observation."""
        return observations[:, -self.num_tasks :].argmax(dim=1)

    def _update_value_scale(self, target_q_values: th.Tensor, task_index: th.Tensor) -> None:
        with th.no_grad():
            for task in range(self.num_tasks):
                selected = target_q_values[task_index == task]
                if selected.numel() < 2:
                    continue
                std = selected.std()
                if th.isfinite(std) and std > 0:
                    self.value_scale[task] = (
                        1 - self.scale_momentum
                    ) * self.value_scale[task] + self.scale_momentum * std

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            task_index = self._task_index(replay_data.observations)

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                assert isinstance(self.target_entropy, float)
                # Per task, this is a (batch, 1) column of that sample's coefficient.
                log_ent_coef = (
                    self.log_ent_coef[task_index].reshape(-1, 1)
                    if self.per_task_alpha
                    else self.log_ent_coef
                )
                ent_coef = th.exp(log_ent_coef.detach())
                ent_coef_loss = -(log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.mean().item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            if self.normalize_critic_loss:
                self._update_value_scale(target_q_values, task_index)
                scale = self.value_scale.clamp_min(self.min_value_scale)
                scale = scale[task_index].reshape(-1, 1)
                critic_loss = 0.5 * sum(
                    (((current_q - target_q_values) / scale) ** 2).mean()
                    for current_q in current_q_values
                )
            else:
                critic_loss = 0.5 * sum(
                    F.mse_loss(current_q, target_q_values) for current_q in current_q_values
                )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        # Per-task numbers are the point of this subclass: an averaged one hides exactly
        # the imbalance it exists to fix.
        if self.per_task_alpha and self.log_ent_coef is not None:
            for task, value in enumerate(self.log_ent_coef.detach().exp().cpu().numpy()):
                self.logger.record(f"train/ent_coef/task{task}", float(value))
        if self.normalize_critic_loss:
            for task, value in enumerate(self.value_scale.cpu().numpy()):
                self.logger.record(f"train/value_scale/task{task}", float(value))


ALGO_TABLE = {
    "sac": SAC,
    "mtsac": MTSAC,
}

# A config names these as strings; `train.py` swaps in the class. Anything SB3 already
# knows ("MlpPolicy") is left alone for it to resolve itself.
POLICY_TABLE = {
    "LayerNormSACPolicy": LayerNormSACPolicy,
}

REPLAY_BUFFER_TABLE = {
    "TaskWeightedReplayBuffer": TaskWeightedReplayBuffer,
}
