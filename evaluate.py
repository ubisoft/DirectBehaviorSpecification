import argparse
import numpy as np
import gym
import logging
import sys
import torch
import time

from copy import deepcopy

from algorithms.sac_separate_critics import SACmultiCritics
from algorithms.td3_separate_critics import TD3multiCritics

from utils.alg_env_lists import UNITY_ENVS
from utils.misc import unity_rendering_burn_in, create_env, set_seeds, get_model_file, get_cost_vector

from alfred.utils.config import parse_bool, load_config_from_json
from alfred.utils.directory_tree import DirectoryTree
from alfred.utils.misc import uniquify, create_logger


def wait_for_ENTER_keypress():
    while True:
        answer = input('Press "ENTER" to continue')
        if answer == "":
            return


def get_evaluation_args(overwritten_args=None):
    parser = argparse.ArgumentParser()

    # alfred arguments

    parser.add_argument("--root_dir", default=None, type=str)
    parser.add_argument("--storage_name", default=None, type=str)
    parser.add_argument("--experiment_num", default=1, type=int)
    parser.add_argument("--seed_num", default=1, type=int, help="Seed directory in experiments folder")

    # evaluation arguments

    parser.add_argument("--model_to_load", default="last", type=str, choices=["last", "best", "feasible"])
    parser.add_argument("--eval_seed", default=123, type=int, help="Random seed for evaluation rollouts")
    parser.add_argument("--max_episode_len", default=np.inf, type=int)
    parser.add_argument("--n_episodes", default=5, type=int)
    parser.add_argument("--act_deterministic", type=parse_bool, default=False)

    # rendering arguments

    parser.add_argument("--render", type=parse_bool, default=True)
    parser.add_argument("--auto_wait", type=float, default=0.075)
    parser.add_argument("--user_wait", type=parse_bool, default=False)

    return parser.parse_args(overwritten_args)


def evaluate(args):
    # Load model and config
    dir_tree = DirectoryTree.init_from_branching_info(root_dir=args.root_dir, storage_name=args.storage_name,
                                                      experiment_num=args.experiment_num, seed_num=args.seed_num)

    config = load_config_from_json(dir_tree.seed_dir / "config.json")
    logger = create_logger(name="Eval", loglevel=logging.INFO)

    # Instantiate environment

    env = create_env(config.task_name, render=args.render, seed=args.eval_seed,
                     time_scale=config.time_scale, decision_period=config.decision_period,
                     constraints_to_enforce=config.constraints_to_enforce,
                     constraint_rates_to_add_as_obs=config.constraint_rates_to_add_as_obs,
                     add_time_as_obs=config.add_time_as_obs,
                     logger=logger)

    # Instantiate agent

    agent = create_agent(config, env, logger)

    # Load the models

    model_file = get_model_file(dir_tree=dir_tree, model_type=args.model_to_load)
    agent.load_model(path=model_file.parent, logger=logger, name=model_file.name)

    # Setting seeds

    set_seeds(args.eval_seed, env)

    returns = []

    if args.render and config.task_name in UNITY_ENVS:
        logger.info("RENDERING BURN-IN (SORRY)")
        if sys.platform == "win32":
            unity_rendering_burn_in(env)

    for episode_i in range(args.n_episodes):

        step_i = 0
        done = False
        ret = 0

        # Initialise environment

        state = env.reset()

        # Rendering

        if args.render:
            env.render()

        if args.user_wait:
            wait_for_ENTER_keypress()

        while not done and step_i < args.max_episode_len:

            # Action selection and environment step

            action, _ = agent.select_action(state, sample_action=False if args.act_deterministic else True)
            next_state, reward, done, _ = env.step(action)

            ret += reward
            step_i += 1
            state = next_state

            # Rendering

            if args.render:
                env.render()
                time.sleep(args.auto_wait)

            if args.user_wait:
                wait_for_ENTER_keypress()

        returns.append(ret)
        logger.info(ret)

    return returns


def evaluate_agent(agent, env,
                   n_episodes=10, sample_action=False, record_state_trajectories=False,
                   constraints_to_enforce=None, constraint_is_reversed=None):
    """
    Collect states encountered by the agent for n_episodes
    :param agent:
    :param env: (gym.Env) environment from which to collect trajectories
    :param n_episodes: (int) number of trajectories to collect
    :param sample_action: (bool) whether to sample actions or take greedy action
    :return: list of lists of states
    """
    all_costs = []
    episodes_return = []
    state_trajectories = []
    episode_dones = []
    for episode_i in range(n_episodes):
        step_i = 0
        done = False
        state_traj = []
        ep_return = 0

        state = env.reset()

        state_traj.append(state)
        while not done and step_i < env._max_episode_steps:
            action, _ = agent.select_action(state, sample_action=sample_action)
            next_state, reward, done, info = env.step(action)

            state = next_state
            ep_return += reward
            step_i += 1
            if record_state_trajectories:
                state_traj.append(state)

            if constraints_to_enforce is not None:
                cost = get_cost_vector(info, constraints_to_enforce, constraint_is_reversed)
                all_costs.append(deepcopy(cost))

        episodes_return.append(ep_return)
        if record_state_trajectories:
            state_trajectories.append(state_traj)

        episode_dones.append(done)

    return episodes_return, state_trajectories, all_costs, episode_dones


def create_agent(config, env, logger):
    n_critics = len(config.constraints_to_enforce) + 1 if config.constraint_fixed_weights is None else len(config.constraint_fixed_weights) + 1
    if config.alg_name == "sac":
        agent = SACmultiCritics(env.observation_space, env.action_space, config, logger, n_critics=n_critics)
    elif config.alg_name == "td3":
        agent = TD3multiCritics(env.observation_space, env.action_space, config, logger, n_critics=n_critics)
    else:
        raise NotImplementedError
    return agent


if __name__ == "__main__":
    args = get_evaluation_args()
    evaluate(args)
