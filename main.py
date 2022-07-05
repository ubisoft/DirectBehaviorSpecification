import argparse
import numpy as np
import itertools
import logging
from copy import deepcopy
import time
import torch

from algorithms.lagrangian_methods import LagrangianWrapper
from utils.replay_memory import ReplayBuffer
from utils.misc import set_seeds, create_env, parse_threshold_list, get_model_file, constraints_are_satisfied, get_cost_vector
from evaluate import create_agent, evaluate_agent
from utils.alg_env_lists import *

import alfred
from alfred.utils.misc import create_management_objects
from alfred.utils.config import parse_bool, parse_log_level, save_config_to_json, load_config_from_json
from alfred.utils.recorder import Recorder
from alfred.utils.directory_tree import *
import alfred.defaults


def set_up_alfred():
    alfred.defaults.DEFAULT_DIRECTORY_TREE_ROOT = "./storage"


def get_main_args(overwritten_args=None):
    parser = argparse.ArgumentParser()

    # alfred's arguments

    parser.add_argument('--alg_name', type=str, default='sac', choices=ALL_ALGS)
    parser.add_argument('--task_name', type=str, default="ArenaEnv-v0", choices=ALL_ENVS)
    parser.add_argument("--desc", type=str, default="", help="Description of the experiment to be run")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--root_dir", default="./storage", type=str, help="Root directory")
    parser.add_argument("--log_level", default=logging.INFO, type=parse_log_level, help="Logging level")

    # common arguments

    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate.')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of transitions for each model update.')

    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size of the policy and critic.')

    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Discount factor for return.')

    parser.add_argument('--tau', type=float, default=0.005,
                        help='Target networks smoothing coefficient.')

    parser.add_argument('--policy_update_delay_relative_to_critic', type=int, default=1,
                        help='Number of updates that the critic makes for every update of the policy.')

    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of batch-sizes worth of env steps to warmup the buffer before starting training.')

    parser.add_argument('--start_steps', type=int, default=10000,
                        help='Number of env steps during which to sample random actions only.')

    parser.add_argument('--num_steps', type=int, default=None,
                        help='Maximum number of steps.')

    parser.add_argument('--steps_bw_update', type=int, default=200,
                        help='Number of env steps between agent updates.')

    parser.add_argument('--steps_bw_save', type=int, default=20000,
                        help='Number of env steps between save.')

    parser.add_argument('--steps_bw_eval', type=int, default=20000,
                        help='Number of env episodes between evaluations.')

    parser.add_argument('--n_eval_episodes', type=int, default=10,
                        help='Number of episodes to evaluate over.')

    parser.add_argument('--replay_size', type=int, default=1000000,
                        help='Size of the replay buffer.')

    parser.add_argument('--use_cuda', type=parse_bool, default=False)

    parser.add_argument('--render', type=parse_bool, default=False)

    # SAC arguments

    parser.add_argument('--alpha', type=float, default=0.02,
                        help='Temperature parameter that determines the relative importance of the entropy.')

    parser.add_argument('--entropy_target', type=float, default=None,
                        help='Entropy target for the entropy-maximisation objective. None is the default -dim(A).')

    parser.add_argument('--automatic_entropy_tuning', type=parse_bool, default=True,
                        help='Automatically adjust entropy temperature alpha.')

    # TD3 arguments

    parser.add_argument('--target_noise_amplitude', type=float, default=0.2, metavar='G',
                        help='Noise added to target policy during critic update')

    parser.add_argument('--target_noise_clip', type=float, default=0.5, metavar='G',
                        help='Range to clip target policy noise')

    parser.add_argument('--init_logstd', type=float, default=-3., metavar='G',
                        help='Init log-std of the gaussian policy for fixed-noise injection exploration')

    # constrained RL arguments

    parser.add_argument('--constraints_to_enforce', nargs='+', type=str, default=None,
                        help="List of strings. These constraint names should be found in 'info' of the env.step().")

    parser.add_argument('--constraint_is_reversed', nargs='+', type=parse_bool, default=None,
                        help="List of booleans. Whether the corresponding constraint is a desirable behavior or not.")

    parser.add_argument('--constraint_discount_factors', nargs='+', type=float, default=None,
                        help="List of floats. The discount factor for each cost-constraint.")

    parser.add_argument('--constraint_rates_to_add_as_obs', nargs='+', type=str, default=None,
                        help="Behaviors for which the occurrence rates should be be tracked and added "
                             "to the agent's observations to help it to know whether it is respecting the constraint.")

    parser.add_argument('--add_time_as_obs', type=parse_bool, default=True,
                        help="Whether to ass time indicator to the agent observation vector to make it aware of the time limit.")

    parser.add_argument('--constraint_enforcement_method', type=str, choices=['lagrangian', 'reward_engineering'], default=None,
                        help="reward_engineering uses fixed_weights whereas lagrangian methods automatically adapt the constraint weights.")

    parser.add_argument('--constraint_fixed_weights', nargs='+', type=float, default=None,
                        help="Only used if --constraint_enforcement_method='reward_engineering'")

    parser.add_argument('--bootstrap_constraint', type=str, default=None,
                        help="One special constraint that will lend its multiplier to the reward function.")

    parser.add_argument('--constraint_thresholds', type=parse_threshold_list, default=None,
                        help="Defines a pair of thresholds for each constrain. The first is a lower bound constraint, "
                             "the second is an upper bound constraint. Separate bounds with a dash '-' and constraints "
                             "with a coma ','. Float and nan values are accepted. "
                             "\ne.g. '--constraint_thresholds 0.5-0.7 nan-0.3'.\n")

    parser.add_argument('--lagrange_multipliers_batch_size', type=int, default=2000,
                        help="Number of last transitions to use for the multipliers update.")

    parser.add_argument('--multipliers_update_delay_relative_to_agent', type=int, default=10,
                        help="Number of updates that the agent makes for every update of the multipliers.")

    parser.add_argument('--multipliers_lr_relative_to_policy', type=float, default=100.,
                        help="Size of the multipliers' learning rate relative to that of the agent.")

    parser.add_argument('--use_normalized_multipliers', type=parse_bool, default=True,
                        help="Whether all multipliers (including that of the reward function) should sum to 1.")

    # UNITY environments arguments

    parser.add_argument('--decision_period', type=int, default=5,
                        help='Decision Period (in frames) of the agent-environment interaction.')

    parser.add_argument('--time_scale', type=float, default=1.,
                        help='Time Scale (real-time=1.) in the Unity physics simulator.')

    return parser.parse_args(overwritten_args)


def validate_config(args, logger):
    original_args = deepcopy(args)

    # Check for conflicting configurations of hyper-parameters

    if args.alg_name in ON_POLICY_ALGS:

        if args.start_steps >= 0:
            args.start_steps = -1
            logger.info("CONFIG CHANGED: On-policy algorithms do not use start_steps.")

    if args.alg_name == "sac" and not args.automatic_entropy_tuning:
        if args.entropy_target is not np.nan:
            args.entropy_target = np.nan
            logger.info("CONFIG CHANGED: when training SAC without automatic alpha-tuning, the entropy_target "
                        "is not used and is therefore set to np.nan.")

    # About constrained RL methods

    def set_constraint_arguments_to_none(args):
        args.constraints_to_enforce = None
        logger.info(f"CONFIG CHANGED: --constraints_to_enforce set to None.")

        args.constraint_is_reversed = None
        logger.info(f"CONFIG CHANGED: --constraint_is_reversed set to None.")

        args.constraint_discount_factors = None
        logger.info(f"CONFIG CHANGED: --constraint_discount_factors set to None.")

        args.constraint_rates_to_add_as_obs = None
        logger.info(f"CONFIG CHANGED: --constraint_rates_to_add_as_obs set to None.")

        args.constraint_enforcement_method = None
        logger.info(f"CONFIG CHANGED: --constraint_enforcement_method set to None.")

        return args

    def set_reward_engineering_arguments_to_none(args):
        args.constraint_fixed_weights = None
        logger.info(f"CONFIG CHANGED: --constraint_fixed_weights set to None.")

        return args

    def set_lagrangian_arguments_to_none(args):
        args.bootstrap_constraint = None
        logger.info(f"CONFIG CHANGED: --bootstrap_constraint set to None.")

        args.constraint_thresholds = None
        logger.info(f"CONFIG CHANGED: --constraint_thresholds set to None.")

        args.lagrange_multipliers_batch_size = None
        logger.info(f"CONFIG CHANGED: --lagrange_multipliers_batch_size set to None.")

        args.multipliers_update_delay_relative_to_agent = None
        logger.info(f"CONFIG CHANGED: --multipliers_update_delay_relative_to_agent set to None.")

        args.use_normalized_multipliers = None
        logger.info(f"CONFIG CHANGED: --use_normalized_multipliers set to None.")

        return args

    if args.task_name not in CONSTRAINED_ENVS:
        args = set_constraint_arguments_to_none(args)
        args = set_reward_engineering_arguments_to_none(args)
        args = set_lagrangian_arguments_to_none(args)

    else:
        assert len(args.constraints_to_enforce) == len(args.constraint_is_reversed)
        assert len(args.constraints_to_enforce) == len(args.constraint_discount_factors)

        if args.constraint_enforcement_method == "reward_engineering":
            args = set_lagrangian_arguments_to_none(args)

        elif args.constraint_enforcement_method == "lagrangian":
            args = set_reward_engineering_arguments_to_none(args)
            assert len(args.constraints_to_enforce) == len(args.constraint_thresholds)

            if args.bootstrap_constraint is not None:
                assert args.bootstrap_constraint in args.constraints_to_enforce, \
                    f"args.bootstrap_constraint should be part of args.constraints_to_enforce: args.bootstrap_constraint={args.bootstrap_constraint}, args.constraints_to_enforce={args.constraints_to_enforce}."

            if args.steps_bw_update is None:
                # if --steps_bw_update is not defined, we make sure that the multipliers are updated on-policy
                assert args.lagrange_multipliers_batch_size is not None
                args.steps_bw_update = args.lagrange_multipliers_batch_size / args.policy_update_delay_relative_to_critic
                logger.info(f"CONFIG CHANGED: --steps_bw_update was None so it has been set to --lagrange_multipliers_batch_size={args.lagrange_multipliers_batch_size}.")

            if args.multipliers_update_delay_relative_to_agent is None:
                assert args.policy_update_delay_relative_to_critic is not None
                # if --multipliers_update_delay_relative_to_agent is not defined, the multipliers will be updated with the policy
                args.multipliers_update_delay_relative_to_agent = args.policy_update_delay_relative_to_critic
                logger.info(f"CONFIG CHANGED: --multipliers_update_delay_relative_to_agent was None so we have set it to --policy_update_delay_relative_to_critic={args.policy_update_delay_relative_to_critic}.")

    if args.constraints_to_enforce is None:
        args.constraints_to_enforce = []

    # Re-validate args if they have been changed

    if original_args != args:
        args = validate_config(args, logger)

    return args


def main(config, dir_tree=None, logger=None, pbar="default_pbar"):
    # Setting up alfred's configs

    set_up_alfred()

    # Management objects

    dir_tree, logger, _ = create_management_objects(dir_tree=dir_tree, logger=logger, pbar=None, config=config)

    # Scan arguments for errors

    config = validate_config(config, logger)

    # Determines which device will be used

    config.device = str(torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu"))
    logger.info(f"Device to be used: {config.device}")

    # Saving config

    save_config_to_json(config, filename=str(dir_tree.seed_dir / "config.json"))
    if (dir_tree.seed_dir / "config_unique.json").exists():
        config_unique = load_config_from_json(filename=str(dir_tree.seed_dir / "config_unique.json"))
        for k in config_unique.__dict__.keys():
            config_unique.__dict__[k] = config.__dict__[k]
        save_config_to_json(config_unique, filename=str(dir_tree.seed_dir / "config_unique.json"))

    # Environment

    env = create_env(config.task_name,
                     render=config.render,
                     seed=config.seed,
                     time_scale=config.time_scale,
                     decision_period=config.decision_period,
                     constraints_to_enforce=config.constraints_to_enforce,
                     constraint_rates_to_add_as_obs=config.constraint_rates_to_add_as_obs,
                     add_time_as_obs=config.add_time_as_obs,
                     logger=logger)

    # Seeding

    set_seeds(config.seed, env)

    # Agent

    agent = create_agent(config, env, logger)

    # Constrained RL wrapper

    if config.constraint_enforcement_method is not None:
        if config.constraint_enforcement_method == "lagrangian":
            agent = LagrangianWrapper(base_agent=agent, config=config)
        elif config.constraint_enforcement_method == "reward_engineering":
            pass
        else:
            raise NotImplementedError

    # Recorder

    os.makedirs(dir_tree.recorders_dir, exist_ok=True)
    train_recorder = Recorder(agent.metrics_to_record)

    # Memory

    memory = ReplayBuffer(config.replay_size, config.seed)

    # First evaluation before initiating training

    last_best_eval = -float('inf')
    last_best_eval_feasible = -float('inf')

    train_recorder, env, last_best_eval, last_best_eval_feasible = evaluation_step(agent, env, config, dir_tree,
                                                                                   train_recorder, logger,
                                                                                   total_steps=0,
                                                                                   episode_i=0,
                                                                                   last_best_eval=last_best_eval,
                                                                                   last_best_eval_feasible=last_best_eval_feasible)

    # Training loop

    total_steps = 0
    update_i = 0
    stop_training = False
    last_eval_step = 0

    for episode_i in itertools.count(1):
        step_i = 0
        episode_return = 0
        done = False
        state = env.reset()
        time_start = time.time()

        # Episode loop

        while not done and step_i < env._max_episode_steps:

            if config.render:
                env.render()

            # Take action in the environment

            if config.start_steps > total_steps:
                action = env.action_space.sample()  # Sample random action
                unsquashed_action = None
            else:
                action, unsquashed_action = agent.select_action(state, sample_action=True)  # Sample action from policy

            # Update the parameters

            multiplier_batch_size = config.lagrange_multipliers_batch_size if config.lagrange_multipliers_batch_size is not None else 0
            if len(memory) > config.warmup * config.batch_size and total_steps % config.steps_bw_update == 0 and len(memory) > multiplier_batch_size:
                new_recordings = agent.update_parameters(memory, update_i, reward_weight=1., cost_weights=config.constraint_fixed_weights)

                # Save loss recordings

                new_recordings.update({"total_steps": total_steps})
                train_recorder.write_to_tape(new_recordings)
                update_i += 1

                # Clear memory

                if config.alg_name in ON_POLICY_ALGS:
                    memory.clear_memory()

            # Environment step

            next_state, reward, done, info = env.step(action)
            step_i += 1
            total_steps += 1
            episode_return += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # Check: https://arxiv.org/abs/1712.00378
            not_done = 1. if step_i == env._max_episode_steps else float(not done)

            # Extract constraint-related information from the 'info'
            if config.constraint_enforcement_method is not None:
                cost = get_cost_vector(info, config.constraints_to_enforce, config.constraint_is_reversed)
            else:
                cost = None

            # Append transition to memory

            memory.push(state, action, reward, next_state, not_done, unsquashed_action, cost)

            # Update current state

            state = next_state

            # Save plots and models

            if total_steps % config.steps_bw_save == 0:
                save_plots(agent, train_recorder, dir_tree)

            # Decide whether to continue training

            if total_steps >= config.num_steps:
                stop_training = True
                break

        if stop_training:
            break

        time_stop = time.time()

        # Save return recordings

        new_recordings = {
            'total_steps': total_steps,
            'episode_i': episode_i,
            'episode_len': step_i,
            'train_return': episode_return,
            'episode_time': time_stop - time_start,
            'wallclock_time': time.time(),
        }

        train_recorder.write_to_tape(new_recordings)
        logger.debug(f"Episode: {episode_i}, total numsteps: {total_steps}, episode steps: {step_i} ({time_stop - time_start:.2f}s), reward: {round(episode_return, 2)}")

        # Evaluate the policy

        if total_steps - last_eval_step > config.steps_bw_eval:
            train_recorder, env, last_best_eval, last_best_eval_feasible = evaluation_step(agent, env, config, dir_tree,
                                                                                           train_recorder, logger,
                                                                                           total_steps, episode_i,
                                                                                           last_best_eval=last_best_eval,
                                                                                           last_best_eval_feasible=last_best_eval_feasible)

            last_eval_step = total_steps

    # Final evaluation and save

    train_recorder, env, last_best_eval, last_best_eval_feasible = evaluation_step(agent, env, config, dir_tree,
                                                                                   train_recorder, logger,
                                                                                   total_steps, episode_i,
                                                                                   last_best_eval=last_best_eval,
                                                                                   last_best_eval_feasible=last_best_eval_feasible)
    save_plots(agent, train_recorder, dir_tree)

    last_model_file = get_model_file(dir_tree=dir_tree, model_type="last")
    if last_model_file is not None:
        os.remove(str(last_model_file))
    agent.save_model(path=dir_tree.seed_dir, logger=logger, name=f"models_last_step{total_steps}.pt")

    env.close()


def save_plots(agent, train_recorder, dir_tree):
    train_recorder.save(dir_tree.recorders_dir / 'train_recorder.pkl')
    agent.create_plots(train_recorder, save_dir=dir_tree.seed_dir)


def evaluation_step(agent, env, config, dir_tree, train_recorder, logger, total_steps, episode_i,
                    last_best_eval, last_best_eval_feasible):
    eval_returns_greedy, state_trajectories_greedy, all_costs_greedy, _ = evaluate_agent(
        agent, env,
        n_episodes=config.n_eval_episodes, sample_action=False,
        record_state_trajectories=True,
        constraints_to_enforce=config.constraints_to_enforce, constraint_is_reversed=config.constraint_is_reversed)

    eval_returns_sampled, state_trajectories_sampled, all_costs_sampled, _ = evaluate_agent(
        agent, env,
        n_episodes=config.n_eval_episodes, sample_action=True,
        record_state_trajectories=True,
        constraints_to_enforce=config.constraints_to_enforce, constraint_is_reversed=config.constraint_is_reversed)

    # Log the results of evaluation

    avg_eval_return_greedy = np.mean(eval_returns_greedy)
    avg_eval_return_sampled = np.mean(eval_returns_sampled)

    avg_eval_constraints_greedy = []
    for k in range(len(config.constraints_to_enforce)):
        all_costs_greedy = np.array(all_costs_greedy)
        nan_free_cost_batch_k = all_costs_greedy[
            np.logical_not(np.isnan(all_costs_greedy[:, k])), k]  # ignores NaNs for constraint computation
        avg_eval_constraints_greedy.append(np.mean(nan_free_cost_batch_k, axis=0))

    avg_eval_constraints_sampled = []
    for k in range(len(config.constraints_to_enforce)):
        all_costs_sampled = np.array(all_costs_sampled)
        nan_free_cost_batch_k = all_costs_sampled[
            np.logical_not(np.isnan(all_costs_sampled[:, k])), k]  # ignores NaNs for constraint computation
        avg_eval_constraints_sampled.append(np.mean(nan_free_cost_batch_k, axis=0))

    new_recordings = {
        'eval_total_steps': total_steps,
        'eval_episode_i': episode_i,
        'eval_return_greedy': avg_eval_return_greedy,
        'eval_return_sampled': avg_eval_return_sampled
    }

    if config.constraints_to_enforce is not None:
        new_recordings.update(
            {f"eval_greedy_constraint_{k + 1}": avg_eval_constraints_greedy[k]
             for k in range(len(config.constraints_to_enforce))})

        new_recordings.update(
            {f"eval_sampled_constraint_{k + 1}": avg_eval_constraints_sampled[k]
             for k in range(len(config.constraints_to_enforce))})

    train_recorder.write_to_tape(new_recordings)

    # Saves models

    last_model_file = get_model_file(dir_tree=dir_tree, model_type="last")
    if last_model_file is not None:
        os.remove(str(last_model_file))
    agent.save_model(path=dir_tree.seed_dir, logger=logger, name=f"models_last_step{total_steps}.pt")

    new_best = False
    if avg_eval_return_sampled > last_best_eval:
        best_model_file = get_model_file(dir_tree=dir_tree, model_type="best")
        if best_model_file is not None:
            os.remove(str(best_model_file))
        agent.save_model(path=dir_tree.seed_dir, logger=logger, name=f"models_best_step{total_steps}.pt")
        last_best_eval = deepcopy(avg_eval_return_sampled)
        new_best = True

    new_feasible = False
    if avg_eval_return_sampled > last_best_eval_feasible and all(constraints_are_satisfied(avg_eval_constraints_sampled, config.constraint_thresholds)):
        best_model_file_feasible = get_model_file(dir_tree=dir_tree, model_type="feasible")
        if best_model_file_feasible is not None:
            os.remove(str(best_model_file_feasible))
        agent.save_model(path=dir_tree.seed_dir, logger=logger, name=f"models_feasible_step{total_steps}.pt")
        last_best_eval_feasible = deepcopy(avg_eval_return_sampled)
        new_feasible = True

    print_out = f"Total number of steps: {total_steps}, Number of eval Episodes: {config.n_eval_episodes}, " \
                f"Return (greedy, sampled): ({avg_eval_return_greedy:.2f}, {avg_eval_return_sampled:.2f})"
    if new_best: print_out += " [NEW BEST]"
    if new_feasible: print_out += " [NEW FEASIBLE]"
    logger.info("----------------------------------------")
    logger.info(print_out)
    logger.info("----------------------------------------")

    return train_recorder, env, last_best_eval, last_best_eval_feasible


if __name__ == "__main__":
    config = get_main_args()
    main(config)
