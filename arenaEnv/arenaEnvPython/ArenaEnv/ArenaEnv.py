import gym
from gym.envs.registration import register

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from arenaEnv.arenaEnvPython.utils.string_channel import StringChannel
from gym_unity.envs import UnityToGymWrapper

from copy import deepcopy
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import random

OBS_IDX_MEANINGS = {
    "agent_pos_x": 0,
    "agent_pos_z": 1,
    "agent_velocity_x": 2,
    "agent_velocity_z": 3,
    "agent_velocity_mag": 4,
    "agent_pos_y": 5,
    "agent_on_the_ground": 6,
    "agent_velocity_y": 7,
    "agent_view_x": 8,
    "agent_view_z": 9,
    "agent_angular_velocity_y": 10,
    "goal_relative_dir_x": 11,
    "goal_relative_dir_z": 12,
    "goal_dist": 13,
    "marker_relative_dir_x": 14,
    "marker_relative_dir_z": 15,
    "marker_dist": 16,
    "marker_view_angle": 17,
    "marker_in_fov": 18,
    "agent_is_recharging": 19,
    "energy_bar": 20,
    "agent_in_lava": 21,
    "agent_lava_raycast": list(range(22, 47))
}

ACT_IDX_MEANINGS = {
    "agent_vel_x": 0,
    "agent_vel_z": 1,
    "agent_rot_y": 2,
    "jump": 3,
    "recharge": 4
}


class ArenaEnvConstraintsWrapper(gym.Wrapper):

    def __init__(self, env, constraints_to_enforce=None, agent_speed_limit=0.5,
                 agent_energy_limit=0.01, agent_energy_loss=0.010, agent_energy_gain=0.030, agent_energy_init=0.25,
                 constraint_rates_to_add_as_obs=None, add_time_as_obs=True):
        super().__init__(env)

        self.possible_constraints = {
            "has-reached-goal-in-episode",
            "is-looking-at-marker",
            "is-on-ground",
            "is-in-lava",
            "is-above-speed-limit",
            "is-above-energy-limit"
        }

        self._step_i = None
        self.add_time_as_obs = add_time_as_obs

        if constraints_to_enforce is not None:
            self.constraints_to_enforce = constraints_to_enforce
        else:
            self.constraints_to_enforce = []

        assert agent_energy_limit < 1. and agent_energy_limit > 0.
        self.agent_energy_limit = agent_energy_limit
        self.agent_energy_loss = agent_energy_loss
        self.agent_energy_gain = agent_energy_gain
        self.agent_energy_init = agent_energy_init

        assert agent_speed_limit < 1. and agent_speed_limit > 0.
        self.agent_speed_limit = agent_speed_limit

        self._remove_unecessary_obs_and_act()

        if constraint_rates_to_add_as_obs is None:
            self.constraint_rates_to_add_as_obs = []
        else:
            self.constraint_rates_to_add_as_obs = constraint_rates_to_add_as_obs

        self._add_constraint_rates_to_obs()

    def _remove_unecessary_obs_and_act(self):

        self.retained_obs_idxs = list(range(self.observation_space.shape[0]))
        self.retained_act_idxs = list(range(self.action_space.shape[0]))

        # About lava

        if "is-in-lava" not in self.constraints_to_enforce:
            self.retained_obs_idxs = [i for i in self.retained_obs_idxs if
                                      i != OBS_IDX_MEANINGS['agent_in_lava'] and i not in OBS_IDX_MEANINGS[
                                          "agent_lava_raycast"]
                                      ]
            self.env.stringChannel.send_string(data=f"LavaIsActive=false")

        # About look-at marker

        if "is-looking-at-marker" not in self.constraints_to_enforce:
            self.retained_obs_idxs = [i for i in self.retained_obs_idxs if
                                      i not in [OBS_IDX_MEANINGS['marker_relative_dir_x'],
                                                OBS_IDX_MEANINGS['marker_relative_dir_z'],
                                                OBS_IDX_MEANINGS['marker_dist'],
                                                OBS_IDX_MEANINGS['marker_view_angle'],
                                                OBS_IDX_MEANINGS['marker_in_fov']]
                                      ]
            self.env.stringChannel.send_string(data=f"LookAtIsActive=false")

        # About energy-bar

        if "is-above-energy-limit" not in self.constraints_to_enforce:
            self.retained_obs_idxs = [i for i in self.retained_obs_idxs if
                                      i != OBS_IDX_MEANINGS['agent_is_recharging'] and i != OBS_IDX_MEANINGS[
                                          "energy_bar"]]

            self.retained_act_idxs = [i for i in self.retained_act_idxs if
                                      i != ACT_IDX_MEANINGS['recharge']]

            self.env.stringChannel.send_string(data="EnergyBarIsActive=false")

        else:
            self.env.stringChannel.send_string(data=f"EnergyLoss={self.agent_energy_loss}")
            self.env.stringChannel.send_string(data=f"EnergyGain={self.agent_energy_gain}")
            self.env.stringChannel.send_string(data=f"EnergyInit={self.agent_energy_init}")

        # Removing unecessary observations from the observation space

        new_observation_space = deepcopy(self.observation_space)
        new_observation_space.bounded_above = self.observation_space.bounded_above[self.retained_obs_idxs]
        new_observation_space.bounded_below = self.observation_space.bounded_below[self.retained_obs_idxs]
        new_observation_space.high = self.observation_space.high[self.retained_obs_idxs]
        new_observation_space.low = self.observation_space.low[self.retained_obs_idxs]
        new_observation_space.shape = tuple([len(new_observation_space.low)])
        self.observation_space = new_observation_space

        # Removing unecessary actions from the action space

        new_action_space = deepcopy(self.action_space)
        new_action_space.bounded_above = self.action_space.bounded_above[self.retained_act_idxs]
        new_action_space.bounded_below = self.action_space.bounded_below[self.retained_act_idxs]
        new_action_space.high = self.action_space.high[self.retained_act_idxs]
        new_action_space.low = self.action_space.low[self.retained_act_idxs]
        new_action_space.shape = tuple([len(new_action_space.low)])
        self.action_space = new_action_space

    def _add_constraint_rates_to_obs(self):
        # Constraint rate to track
        # Some constraint may be hard to enforce when we require a precise rate of occurance of an event (e.g. jump between 30 and 40% of the time)
        # To make this behavior easier to monitor for the agent, we can add an extra observation for this constraint which counts the
        # rate of occurance of that behavior in the ongoing episode. Even if the constraint is enforced across episodes, it may still help
        # the agent to know how much it has been behaving this way in the current episode (e.g. how much it has jumped so far).

        self.constraint_rate_counters = {constraint_name: None for constraint_name in
                                         self.constraint_rates_to_add_as_obs}

        n_added_obs = len(self.constraint_rate_counters) + int(self.add_time_as_obs)

        new_observation_space = deepcopy(self.observation_space)
        new_observation_space.bounded_above = np.concatenate(
            [self.observation_space.bounded_above, [False for _ in range(n_added_obs)]], axis=0)
        new_observation_space.bounded_below = np.concatenate(
            [self.observation_space.bounded_below, [False for _ in range(n_added_obs)]], axis=0)
        new_observation_space.high = np.concatenate([self.observation_space.high, [1. for _ in range(n_added_obs)]],
                                                    axis=0)
        new_observation_space.low = np.concatenate([self.observation_space.low, [0. for _ in range(n_added_obs)]],
                                                   axis=0)
        new_observation_space.shape = tuple([len(new_observation_space.low)])
        self.observation_space = new_observation_space

    def indicator(self, constraint_name, observation, done):
        """
        Defines for each of the constraints how the indicator-cost-function is implemented from the observation vector
        """
        assert constraint_name in self.possible_constraints, \
            f"'{constraint_name}' not in self.possible_constraints={self.possible_constraints}"

        if constraint_name == "has-reached-goal-in-episode":
            if self._step_i == self._max_episode_steps:
                return 0.
            elif done:
                return 1.
            else:
                return float('nan')  # to signal that this is an invalid state to verify this indicator

        if constraint_name == "is-looking-at-marker":
            return observation[OBS_IDX_MEANINGS["marker_in_fov"]]

        elif constraint_name == "is-on-ground":
            return observation[OBS_IDX_MEANINGS["agent_on_the_ground"]]

        elif constraint_name == "is-in-lava":
            return observation[OBS_IDX_MEANINGS["agent_in_lava"]]

        elif constraint_name == "is-above-speed-limit":
            return float(observation[OBS_IDX_MEANINGS["agent_velocity_mag"]] > self.agent_speed_limit)

        elif constraint_name == "is-above-energy-limit":
            return float(observation[OBS_IDX_MEANINGS["energy_bar"]] > self.agent_energy_limit)

        else:
            raise NotImplementedError

    def step(self, action):
        self._step_i += 1

        # Modify action
        if "is-above-energy-limit" not in self.constraints_to_enforce:
            action = np.concatenate([action, [0.]], axis=0)

        # Take env step
        next_observation, reward, done, info = self.env.step(action)

        # Modify info
        for constraint_name in self.possible_constraints:
            info[constraint_name] = self.indicator(constraint_name, next_observation, done)

        # Increments the constraint_rate_counters
        for constraint_name in self.constraint_rate_counters.keys():
            self.constraint_rate_counters[constraint_name] += self.indicator(constraint_name, next_observation, done)

        # Modify the observation
        next_obs = self.observation(next_observation)

        return next_obs, reward, done, info

    def reset(self, **kwargs):
        self._step_i = 0

        # Resets the environment
        observation = self.env.reset(**kwargs)

        # Resets the constraint_rate_counters
        for constraint_name in self.constraint_rate_counters.keys():
            self.constraint_rate_counters[constraint_name] = self.indicator(constraint_name, observation, done=False)

        # Modify the observation
        obs = self.observation(observation)

        return obs

    def observation(self, observation):

        # Removes unecessary observations
        obs = observation[self.retained_obs_idxs]

        # Adds observations about the behavior rates being tracked
        for constraint_name in self.constraint_rate_counters.keys():
            behavior_rate = self.constraint_rate_counters[constraint_name] / (self._step_i + 1)
            assert behavior_rate <= 1. and behavior_rate >= 0.
            obs = np.concatenate([obs, [behavior_rate]], axis=0)

        # Adds remaining time as observation
        if self.add_time_as_obs:
            remaining_time = 1. - (self._step_i / self._max_episode_steps)
            assert remaining_time <= 1. and remaining_time >= 0.
            obs = np.concatenate([obs, [remaining_time]], axis=0)

        return obs

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps


class ArenaEnv(UnityToGymWrapper):
    def __init__(self, exec_path, seed=1, max_episode_step=200, time_scale=1., decision_period=10, render=False,
                 window_width=850, window_height=500, env_comm_worker_id=1000, logger=None):

        # Instantiate UnityEnvironment
        self.engineConfigChannel = EngineConfigurationChannel()
        self.stringChannel = StringChannel(logger=logger)
        unity_env = UnityEnvironment(file_name=exec_path, no_graphics=not render, seed=seed,
                                     timeout_wait=120, side_channels=[self.engineConfigChannel, self.stringChannel],
                                     worker_id=env_comm_worker_id)

        # Sets Unity Engine configurations
        self.engineConfigChannel.set_configuration_parameters(width=window_width, height=window_height, quality_level=1,
                                                              time_scale=time_scale, target_frame_rate=-1,
                                                              capture_frame_rate=60)

        # Sets DecisionPeriod (in frames) for the agent-environment interaction
        self.stringChannel.send_string(data=f"DecisionPeriod={decision_period}")

        # Wraps Unity Environment into GymWrapper
        super().__init__(unity_env=unity_env, uint8_visual=False, flatten_branched=False, allow_multiple_obs=False)

        # Adds useful attributes
        self._max_episode_steps = max_episode_step


register(
    id='ArenaEnv-v0',
    entry_point='arenaEnv.arenaEnvPython.ArenaEnv.ArenaEnv:ArenaEnv'
)
