import torch
from torch.optim import Adam, SGD
from torch.nn.functional import softmax

import numpy as np
import matplotlib.pyplot as plt

from alfred.utils.recorder import remove_nones

class LagrangianWrapper(object):

    def __init__(self, base_agent, config):
        self.base_agent = base_agent

        self.constraints_to_enforce = config.constraints_to_enforce
        self.bootstrap_constraint = config.bootstrap_constraint
        self.constraint_thresholds = torch.FloatTensor(config.constraint_thresholds).to(self.base_agent.device)
        self.constraint_is_reversed = config.constraint_is_reversed
        self.lagrange_multipliers_batch_size = config.lagrange_multipliers_batch_size
        self.multipliers_update_delay_relative_to_agent = config.multipliers_update_delay_relative_to_agent
        self.use_normalized_multipliers = config.use_normalized_multipliers

        # Instantiate Lagrange multipliers

        if self.use_normalized_multipliers:
            n_multipliers = len(self.constraints_to_enforce) + 1  # add dummy variable for complementary reward weight
        else:
            n_multipliers = len(self.constraints_to_enforce)

        self.multiplier_params  = torch.full(size=(n_multipliers,),
                                          fill_value=0.02,
                                          requires_grad=True,
                                          device=self.base_agent.device)

        self.multipliers_optim = Adam([self.multiplier_params ], lr=config.multipliers_lr_relative_to_policy * config.lr, eps=1e-5, betas=(0.9, 0.999))

        self.multiplier_signs = None
        self.update_i_multipliers = 0

        # Book-keeping

        self.base_agent.metrics_to_record |= {"update_i_agent"}
        self.base_agent.metrics_to_record |= {"update_i_multipliers"}
        self.base_agent.metrics_to_record |= {f"online_avg_summed_constraint_{k + 1}" for k in range(len(self.constraints_to_enforce))}
        self.base_agent.metrics_to_record |= {f"eval_greedy_constraint_{k + 1}" for k in range(len(self.constraints_to_enforce))}
        self.base_agent.metrics_to_record |= {f"eval_sampled_constraint_{k + 1}" for k in range(len(self.constraints_to_enforce))}
        self.base_agent.metrics_to_record |= {f"cost_weight_{k + 1}" for k in range(len(self.constraints_to_enforce))}
        self.base_agent.metrics_to_record |= {f"reward_weight"}
        self.base_agent.metrics_to_record |= {f"multiplier_logit_{k + 1}" for k in range(len(self.constraints_to_enforce))}
        if self.use_normalized_multipliers:
            self.base_agent.metrics_to_record |= {f"multiplier_logit_0"}


    def update_parameters(self, replay_buffer, update_i, reward_weight=1., cost_weights=None):
        new_recordings = {}

        # Normalising the multipliers

        if self.use_normalized_multipliers:
            multipliers = torch.nn.functional.softmax(self.multiplier_params, dim=0)[1:]
        else:
            multipliers = torch.clamp(self.multiplier_params, min=0., max=None)

        # Lagrange multipliers update  

        if update_i % self.multipliers_update_delay_relative_to_agent == 0:

            # Computing average constraint satisfaction

            _, _, _, _, _, cost_batch = replay_buffer.get_lasts_transitions(self.lagrange_multipliers_batch_size)

            avg_preference_satisfaction = []
            for k in range(len(self.constraints_to_enforce)):
                nan_free_cost_batch_k = cost_batch[np.logical_not(np.isnan(cost_batch[:, k])), k]  # ignores NaNs for constraint computation
                avg_preference_satisfaction.append(np.mean(nan_free_cost_batch_k, axis=0))

            avg_preference_satisfaction = torch.FloatTensor(avg_preference_satisfaction).to(self.base_agent.device)

            # Identifying which constraint of the double bound is active

            with torch.no_grad():
                active_threshold_idxs = [None for _ in self.constraints_to_enforce]

                for k in range(len(self.constraints_to_enforce)):
                    assert not torch.all(torch.isnan(self.constraint_thresholds[k]))

                    if torch.isnan(self.constraint_thresholds[k, 0]):  # Only upper-threshold is defined
                        active_threshold_idxs[k] = 1
                        continue

                    elif torch.isnan(self.constraint_thresholds[k, 1]):  # Only lower-threshold is defined
                        active_threshold_idxs[k] = 0
                        continue

                    else:
                        raise ValueError("Double-bounds should be implemented as two independent constraints.")

                active_threshold_idxs = torch.LongTensor(active_threshold_idxs).to(self.base_agent.device)
                self.multiplier_signs = 2 * (1 - active_threshold_idxs) - 1

            active_thresholds = self.constraint_thresholds[range(len(self.constraints_to_enforce)), active_threshold_idxs]

            # Updating the multipliers

            multiplier_losses = self.multiplier_signs * multipliers * (avg_preference_satisfaction - active_thresholds)
            multiplier_loss = torch.sum(multiplier_losses, dim=0)
            
            self.multipliers_optim.zero_grad()
            multiplier_loss.backward()
            self.multipliers_optim.step()

            self.update_i_multipliers += 1

            # Bookkeeping
            new_recordings.update({'update_i_multipliers': self.update_i_multipliers})
            new_recordings.update({f"online_avg_summed_constraint_{k + 1}": avg_preference_satisfaction[k].detach().cpu().numpy() for k in range(len(self.constraints_to_enforce))})
            if self.use_normalized_multipliers:
                new_recordings.update({f"multiplier_logit_{k}": self.multiplier_params[k].detach().cpu().numpy() for k in range(len(self.multiplier_params))})
            else:
                new_recordings.update({f"multiplier_logit_{k + 1}": self.multiplier_params[k].detach().cpu().numpy() for k in range(len(self.multiplier_params))})

        # Computing reward weights

        cost_weights = self.multiplier_signs.detach().cpu().numpy() * multipliers.detach().cpu().numpy()

        if self.use_normalized_multipliers:
            reward_weight = 1. - np.sum(np.abs(cost_weights), axis=0)
        else:
            reward_weight = 1.

        if self.bootstrap_constraint is not None:
            reward_weight = np.max([
                np.copy(cost_weights[self.constraints_to_enforce.index(self.bootstrap_constraint)]),
                reward_weight
            ])

        # Book-keeping

        new_recordings.update({f"update_i_agent": update_i})
        new_recordings.update({f"cost_weight_{k + 1}": cost_weights[k] for k in range(len(self.constraints_to_enforce))})
        new_recordings.update({f"reward_weight": reward_weight})

        # Policy update

        new_recordings_policy = self.base_agent.update_parameters(replay_buffer, update_i,
                                                                  reward_weight=reward_weight,
                                                                  cost_weights=cost_weights)

        new_recordings.update(new_recordings_policy)

        return new_recordings

    def create_plots(self, train_recorder, save_dir, return_plots=False):
        fig, axes = self.base_agent.create_plots(train_recorder, save_dir, return_plots=True)
        cm = plt.cm.get_cmap("rainbow")

        # Plotting logits of Lagrange multipliers

        if 'multiplier_logit_0' in train_recorder.tape.keys():
            axes[3, 0].plot(
                remove_nones(train_recorder.tape['update_i_multipliers']),
                remove_nones(train_recorder.tape[f'multiplier_logit_0']),
                linewidth=10.,
                alpha=0.5,
                label=f"multiplier_0",
                color="gray"
            )
        for k in range(len(self.constraints_to_enforce)):
            color = np.array(cm(float(k) / (len(self.constraints_to_enforce))))
            axes[3, 0].plot(
                remove_nones(train_recorder.tape['update_i_multipliers']),
                remove_nones(train_recorder.tape[f'multiplier_logit_{k + 1}']),
                linewidth=3.,
                alpha=0.5,
                label=f"multiplier_{k + 1}",
                color=color
            )
        axes[3, 0].set_xlabel("update_i_multipliers")
        axes[3, 0].set_ylabel("multiplier logits")
        axes[3, 0].legend(loc="best")

        # Plotting normalized Lagrange multipliers

        axes[3, 1].plot(
            remove_nones(train_recorder.tape['update_i_agent']),
            remove_nones(train_recorder.tape[f'reward_weight']),
            alpha=0.50,
            label=f"reward_weight",
            linewidth=10.,
            color="gray"
        )

        for k in range(len(self.constraints_to_enforce)):
            color = np.array(cm(float(k) / (len(self.constraints_to_enforce))))
            axes[3, 1].plot(
                remove_nones(train_recorder.tape['update_i_agent']),
                remove_nones(train_recorder.tape[f'cost_weight_{k + 1}']),
                alpha=0.5,
                label=f"cost_weight_{k + 1}",
                color=color
            )
        axes[3, 1].set_xlabel("update_i_agent")
        axes[3, 1].set_ylabel("multipliers")
        axes[3, 1].legend(loc="best")

        # Plotting constraints

        for k in range(len(self.constraints_to_enforce)):
            color = np.array(cm(float(k) / (len(self.constraints_to_enforce))))
            label_prefix = "NOT-" if self.constraint_is_reversed[k] else ""
            axes[3, 2].plot(
                remove_nones(train_recorder.tape['update_i_multipliers']),
                remove_nones(train_recorder.tape[f'online_avg_summed_constraint_{k + 1}']),
                alpha=0.5,
                label=label_prefix + self.constraints_to_enforce[k],
                color=color
            )

            for i in range(2):
                if torch.isnan(self.constraint_thresholds[k, i]):
                    continue
                else:
                    axes[3, 2].axhline(
                        y=self.constraint_thresholds[k, i].cpu(),
                        linestyle = '--',
                        linewidth=4.,
                        alpha=0.75,
                        label=f"threshold_{k + 1}",
                        color=color
                    )

        axes[3, 2].set_xlabel("update_i_multipliers")
        axes[3, 2].set_ylabel("online_avg_summed_constraints")
        axes[3, 2].set_ylim(0, 1)
        axes[3, 2].legend(loc="best")

        # Plotting constraints (eval)

        for k in range(len(self.constraints_to_enforce)):
            color = np.array(cm(float(k) / (len(self.constraints_to_enforce))))
            label_prefix = "NOT-" if self.constraint_is_reversed[k] else ""
            axes[3, 3].plot(
                remove_nones(train_recorder.tape['eval_total_steps']),
                remove_nones(train_recorder.tape[f'eval_sampled_constraint_{k + 1}']),
                alpha=0.5,
                label=label_prefix + self.constraints_to_enforce[k] + "_sampled",
                color=color,
                linestyle='dotted',
                linewidth=2.,
            )

            axes[3, 3].plot(
                remove_nones(train_recorder.tape['eval_total_steps']),
                remove_nones(train_recorder.tape[f'eval_greedy_constraint_{k + 1}']),
                alpha=0.5,
                label=label_prefix + self.constraints_to_enforce[k] + "_greedy",
                color=color,
                linestyle='-',
                linewidth = 2.,
            )

            for i in range(2):
                if torch.isnan(self.constraint_thresholds[k, i]):
                    continue
                else:
                    axes[3, 3].axhline(
                        y=self.constraint_thresholds[k, i].cpu(),
                        linestyle = '--',
                        linewidth=4.,
                        alpha=0.75,
                        label=f"threshold_{k + 1}",
                        color=color
                    )

        axes[3, 3].set_xlabel("eval_total_steps")
        axes[3, 3].set_ylabel("eval_avg_constraints")
        axes[3, 3].set_ylim(0, 1)
        axes[3, 3].legend(loc="best")

        if return_plots:
            return fig, axes

        else:
            plt.tight_layout()

            fig.savefig(str(save_dir / 'graphs.png'))
            plt.close(fig)
            plt.close('all')

    # CALLBACKS to BaseAgent methods and attributes that were not overridden  
    
    @property
    def metrics_to_record(self):
        return self.base_agent.metrics_to_record

    def select_action(self, state, sample_action=False):
        return self.base_agent.select_action(state, sample_action)

    def save_model(self, path, logger, name):
        return self.base_agent.save_model(path, logger, name)

    def load_model(self, path, logger):
        return self.base_agent.load_model(path, logger)

    def to(self, device):
        return self.base_agent.to(device)
