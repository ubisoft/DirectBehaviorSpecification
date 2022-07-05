# Implementation of Soft Actor Critic algorithm (SAC)
# Paper: https://arxiv.org/abs/1812.05905
# Modified from: https://github.com/pranz24/pytorch-soft-actor-critic

import math
import numpy as np
import torch
from torch.nn import ModuleList
import torch.nn.functional as F
from torch.optim import Adam
from algorithms.base import BaseAgent
from utils.misc import soft_update, hard_update
from utils.models import GaussianPolicy, DoubleQNetwork
import matplotlib.pyplot as plt

from alfred.utils.plots import plot_curves
from alfred.utils.recorder import remove_nones


class SACmultiCritics(BaseAgent):
    def __init__(self, observation_space, action_space, config, logger, n_critics):
        super().__init__(observation_space, action_space, config)

        self.batch_size = config.batch_size
        self.tau = config.tau
        self.alpha = config.alpha
        self.automatic_entropy_tuning = config.automatic_entropy_tuning
        self.policy_update_delay_relative_to_critic = config.policy_update_delay_relative_to_critic
        self.constraints_to_enforce = config.constraints_to_enforce

        # critic

        self.critic = ModuleList([DoubleQNetwork(self.obs_dim, self.act_dim, config.hidden_size).to(device=self.device) for _ in range(n_critics)])
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.lr, eps=1e-5)

        self.critic_target = ModuleList([DoubleQNetwork(self.obs_dim, self.act_dim, config.hidden_size).to(device=self.device) for _ in range(n_critics)])
        hard_update(self.critic_target, self.critic)

        self.update_i_critic = 0

        if config.constraint_discount_factors is not None:
            self.gammas = torch.FloatTensor([config.gamma] + config.constraint_discount_factors).to(self.device)
        else:
            self.gammas = torch.FloatTensor([config.gamma for _ in range(len(self.critic))]).to(self.device)

        # actor

        if self.automatic_entropy_tuning is True:
            if config.entropy_target is None:
                # Entropy target = −dim(A) (e.g. -6 for HalfCheetah-v2) as given in the paper
                self.entropy_target = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            else:
                self.entropy_target = config.entropy_target
            self.log_alpha = torch.full(size=(1,), fill_value=math.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=config.lr, eps=1e-5)

        self.policy = GaussianPolicy(obs_dim=self.obs_dim, act_dim=self.act_dim, hidden_dim=config.hidden_size,
                                     logstd_type="learned_state_dependent", action_space=action_space).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.lr, eps=1e-5)

        self.update_i_policy = 0

        self.metrics_to_record |= {'alpha_loss', 'alpha'}
        self.metrics_to_record |= {f'q1_loss_{i}' for i in range(len(self.critic))}
        self.metrics_to_record |= {f'q2_loss_{i}' for i in range(len(self.critic))}
        self.metrics_to_record |= {f'reward_{i}' for i in range(len(self.critic))}
        self.metrics_to_record |= {f"eval_greedy_constraint_{k + 1}" for k in range(len(self.constraints_to_enforce))}
        self.metrics_to_record |= {f"eval_sampled_constraint_{k + 1}" for k in range(len(self.constraints_to_enforce))}

        if logger is not None:
            logger.info(f"SAC-multi-critic AGENT\n\tPolicy=\n{self.policy}\n\tCritic=\n{self.critic}")

    def update_parameters(self, replay_buffer, update_i, reward_weight=1., cost_weights=None):

        # Sample a minibatch from replay_buffer

        state_batch, action_batch, reward_batch, next_state_batch, not_done_batch, cost_batch = replay_buffer.sample(self.batch_size)

        # Incorporate costs (from secondary objectives) into reward (if applicable)

        if len(self.critic) > 1:
            assert cost_weights is not None, f"If we have more than one critic, we should have more than one reward."

        if cost_weights is not None:
            assert not np.any(cost_batch == None)

            cost_batch[np.isnan(cost_batch)] = 0.  # changes NaNs to zeros for training the critics
            reward_batch = np.concatenate([np.expand_dims(reward_batch, axis=1), cost_batch], axis=1)
            reward_weights = np.concatenate([[reward_weight], cost_weights], axis=0)

        else:
            reward_batch = np.expand_dims(reward_batch, axis=1)
            reward_weights = [reward_weight]

        # Convert to tensor

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        not_done_batch = torch.FloatTensor(not_done_batch).to(self.device).unsqueeze(1)
        reward_weights = torch.FloatTensor(reward_weights).to(self.device)

        # Builds TD target for critics updates

        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            # Compute policy entropy at next state
            next_state_action, next_state_log_pi, _, _, _, _ = self.policy.sample(next_state_batch)
            next_entropy = -next_state_log_pi

        q1_loss_logging = []
        q2_loss_logging = []

        for k, (critic, critic_target) in enumerate(zip(self.critic, self.critic_target)):
            with torch.no_grad():
                # Compute the target Q value
                qf1_target, qf2_target = critic_target(next_state_batch, next_state_action)
                min_target = torch.min(qf1_target, qf2_target) + self.alpha * next_entropy
                target_q_value = reward_batch[:, :, k] + not_done_batch * self.gammas[k] * min_target

            # Critics update

            current_qf1, current_qf2 = critic(state_batch, action_batch)
            qf1_loss = F.mse_loss(current_qf1, target_q_value)
            qf2_loss = F.mse_loss(current_qf2, target_q_value)
            qf_loss = qf1_loss + qf2_loss

            qf_loss.backward()

            q1_loss_logging.append(qf1_loss.detach().cpu().numpy())
            q2_loss_logging.append(qf2_loss.detach().cpu().numpy())

        self.critic_optimizer.step()

        self.update_i_critic += 1

        new_recordings = {
            'update_i_critic': self.update_i_critic,
        }
        new_recordings.update({f'q1_loss_{k}': q1_loss_logging[k] for k in range(len(self.critic))})
        new_recordings.update({f'q2_loss_{k}': q2_loss_logging[k] for k in range(len(self.critic))})
        new_recordings.update({f'reward_{k}': reward_batch[:, :, k].mean().detach().cpu().numpy() for k in range(len(self.critic))})

        if update_i % self.policy_update_delay_relative_to_critic == 0:

            # Actor update

            pi, log_pi, _, _, _, _ = self.policy.sample(state_batch)
            cur_entropy = -log_pi

            min_qf_pi = []
            for k, critic in enumerate(self.critic):
                qf1_pi, qf2_pi = critic(state_batch, pi)
                min_qf_pi.append(torch.min(qf1_pi, qf2_pi))

            policy_loss = - torch.sum(reward_weights * torch.cat(min_qf_pi, dim=1), dim=1, keepdim=True)
            total_policy_loss = (policy_loss - self.alpha * cur_entropy).mean(dim=0)

            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            self.policy_optimizer.step()

            self.update_i_policy += 1

            # Alpha-parameter update

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (-cur_entropy + self.entropy_target).detach()).mean(dim=0)

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp().detach()
            else:
                alpha_loss = torch.tensor(0.).to(self.device)

            new_recordings.update({
                'update_i_actor': self.update_i_policy,
                'pi_loss': policy_loss.detach().mean(dim=0).item(),
                'total_policy_loss': total_policy_loss.item(),
                'alpha_loss': alpha_loss.item(),
                'alpha': float(self.alpha),
                'policy_entropy': cur_entropy.detach().mean(dim=0).item()
            })

        # Update target networks

        soft_update(self.critic_target, self.critic, self.tau)

        return new_recordings

    # Graphs
    def create_plots(self, train_recorder, save_dir, return_plots=False):
        fig, axes = super().create_plots(train_recorder, save_dir)
        cm = plt.cm.get_cmap("rainbow")

        axes[2, 0].plot(
            remove_nones(train_recorder.tape['update_i_critic']),
            remove_nones(train_recorder.tape[f'q1_loss_0']),
            alpha=0.5,
            label=f"q0",
            color="gray")
        for k in range(1, len(self.critic)):
            color = np.array(cm(float(k - 1) / (len(self.critic) - 1)))
            axes[2, 0].plot(
                    remove_nones(train_recorder.tape['update_i_critic']),
                    remove_nones(train_recorder.tape[f'q1_loss_{k}']),
                    alpha=0.5,
                    label=f"q{k}",
                    color=color)
        axes[2, 0].set_xlabel('update_i_critic')
        axes[2, 0].set_ylabel('q_losses')
        axes[2, 0].legend(loc="best")

        plot_curves(axes[2, 1],
                    xs=[remove_nones(train_recorder.tape['update_i_actor'])],
                    ys=[remove_nones(train_recorder.tape['alpha'])],
                    xlabel='update_i',
                    ylabel='alpha')
        ax_alpha = axes[2, 1].twinx()  # instantiate a second axes that shares the same x-axis
        plot_curves(ax_alpha,
                    xs=[remove_nones(train_recorder.tape['update_i_actor'])],
                    ys=[remove_nones(train_recorder.tape['alpha_loss'])],
                    xlabel='update_i',
                    ylabel='alpha_loss',
                    colors=['red'])

        axes[2, 3].plot(
            remove_nones(train_recorder.tape['update_i_critic']),
            remove_nones(train_recorder.tape[f'reward_0']),
            alpha=0.5,
            label=f"reward_0",
            color="gray")
        for k in range(1, len(self.critic)):
            color = np.array(cm(float(k - 1) / (len(self.critic) - 1)))
            axes[2, 3].plot(
                remove_nones(train_recorder.tape['update_i_critic']),
                remove_nones(train_recorder.tape[f'reward_{k}']),
                alpha=0.5,
                label=f"reward_{k}",
                color=color)
        axes[2, 3].set_xlabel('update_i_critic')
        axes[2, 3].set_ylabel('average reward')
        axes[2, 3].legend(loc="best")

        if return_plots:
            return fig, axes

        else:
            plt.tight_layout()

            fig.savefig(str(save_dir / 'graphs.png'))
            plt.close(fig)
