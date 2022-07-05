# Base Agent for this codebase
# Defines basic methods and common plots

import torch

from alfred.utils.plots import create_fig, plot_curves
from alfred.utils.recorder import remove_nones


class BaseAgent(object):
    def __init__(self, observation_space, action_space, config):

        self.device = torch.device(config.device)

        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]

        self.policy = None
        self.policy_optimizer = None

        self.critic = None
        self.critic_optimizer = None

        self.metrics_to_record = {
            'total_steps',
            'eval_total_steps',
            'update_i_actor',
            'update_i_critic',
            'eval_episode_i',
            'episode_i',
            'episode_len',
            'train_return',
            'eval_return_greedy',
            'eval_return_sampled',
            'pi_loss',
            'policy_entropy',
            'total_policy_loss',
            'wallclock_time',
            'episode_time'
        }

    # Act in environment
    def select_action(self, state, sample_action=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            if sample_action:
                action, _, _, unsquashed_action, _, _ = self.policy.sample(state)
            else:
                _, _, action, _, _, unsquashed_action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0], unsquashed_action.detach().cpu().numpy()[0]

    def update_parameters(self, replay_buffer, update_i, reward_weight=1., cost_weights=None):
        raise NotImplementedError
        new_recordings = dict()
        return new_recordings

    # Save model parameters
    def save_model(self, path, logger, name="models.pt"):
        if logger is not None:
            logger.info(f'Saving models to {path}')
        self.to(torch.device("cpu"))
        models = {"policy": self.policy.state_dict(), "critic": self.critic.state_dict()}
        torch.save(models, str(path / name))
        self.to(torch.device(self.device))

    # Load model parameters
    def load_model(self, path, logger, name="models.pt"):
        if logger is not None:
            logger.info(f'Loading models from {path} and {path}')
        models = torch.load(path / name)
        self.policy.load_state_dict(models['policy'])
        self.critic.load_state_dict(models['critic'])
        self.to(self.device)

    # Send models to different device
    def to(self, device):
        self.policy.to(device)
        self.critic.to(device)

    # Graphs
    def create_plots(self, train_recorder, save_dir):

        fig, axes = create_fig((4, 4))

        plot_curves(axes[0, 0],
                    xs=[remove_nones(train_recorder.tape['episode_i'])],
                    ys=[remove_nones(train_recorder.tape['train_return'])],
                    xlabel='episode_i',
                    ylabel='train_return')

        axes[0, 1].plot(
            remove_nones(train_recorder.tape['eval_total_steps']),
            remove_nones(train_recorder.tape['eval_return_greedy']),
            label="greedy")
        axes[0, 1].plot(
            remove_nones(train_recorder.tape['eval_total_steps']),
            remove_nones(train_recorder.tape['eval_return_sampled']),
            color='green',
            label="sampled")
        axes[0, 1].set_xlabel('eval_total_steps')
        axes[0, 1].set_ylabel('eval_return')
        axes[0, 1].legend(loc="best")

        plot_curves(axes[0, 2],
                    xs=[remove_nones(train_recorder.tape['episode_i'])],
                    ys=[remove_nones(train_recorder.tape['episode_len'])],
                    xlabel='episode_i',
                    ylabel="episode_len")

        plot_curves(axes[1, 0],
                    xs=[remove_nones(train_recorder.tape['update_i_actor'])],
                    ys=[remove_nones(train_recorder.tape['pi_loss'])],
                    xlabel='update_i_actor',
                    ylabel='pi_loss')

        axes[1, 1].plot(
            remove_nones(train_recorder.tape['update_i_actor']),
            remove_nones(train_recorder.tape['policy_entropy']),
            label="exact",
            color="orange")
        axes[1, 1].set_xlabel('update_i_actor')
        axes[1, 1].set_ylabel('policy_entropy')

        plot_curves(axes[1, 2],
                    xs=[remove_nones(train_recorder.tape['update_i_actor'])],
                    ys=[remove_nones(train_recorder.tape['total_policy_loss'])],
                    xlabel='update_i_actor',
                    ylabel="total_policy_loss")
        # extra plots axes[2:,:] to be defined by child class

        return fig, axes
