import math
import torch
from math import isnan
import time

from arenaEnv.arenaEnvPython.ArenaEnv.ArenaEnv import *
from utils.alg_env_lists import *


def soft_update(target, source, tau):
    assert 0. < tau and tau < 1.
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


def set_seeds(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)


def parse_threshold_list(str_to_parse):
    """
    Parse a string representing a list of tuples of size 2.
    The two values represent a lower and upper bound.
    :param str_to_parse: The string to be parsed
    :return: The corresponding list of tuples
    """
    # Parse the string. Should be of shape 'lower1-upper1 lower2-upper2 lower3-upper3' etc.
    # Values can be floats or None
    intervals_in_str = str_to_parse.split(',')
    intervals_in_tuples = []
    for interval_str in intervals_in_str:
        interval = []
        for bound in interval_str.split('-'):
            interval.append(float(bound))
        intervals_in_tuples.append(interval)

    # Makes sure each interval only contains 2 values and that the upper-bound is greater than the lower-bound
    for interval in intervals_in_tuples:
        assert len(interval) == 2, f"Wrong number of values: str_to_parse={str_to_parse}, interval={interval}"
        assert not (isnan(interval[0]) and isnan(interval[1])), "Both thresholds cannot be nan."
        if not(isnan(interval[0]) or isnan(interval[1])):
            assert interval[1] > interval[0], f"Upper bound should be greater than lower bound: " \
                                              f":str_to_parse={str_to_parse}, interval={interval}"

    return intervals_in_tuples


def create_env(task_name, render=False, seed=0, time_scale=1., decision_period=10, env_comm_worker_id=None,
               constraints_to_enforce=None, constraint_rates_to_add_as_obs=None, add_time_as_obs=True, logger=None):
    if task_name in UNITY_ENVS:
        env = gym.make(task_name,
                       exec_path=TASK_NAME_TO_EXEC_PATH[task_name],
                       max_episode_step=TASK_NAME_TO_MAX_EPISODE_STEPS[task_name],
                       seed=seed, time_scale=time_scale, decision_period=decision_period,
                       env_comm_worker_id=env_comm_worker_id if env_comm_worker_id is not None else random.randint(10000, 20000),
                       render=render, logger=logger)

        if task_name == "ArenaEnv-v0":
            env = ArenaEnvConstraintsWrapper(env=env,
                                             constraints_to_enforce=constraints_to_enforce,
                                             constraint_rates_to_add_as_obs=constraint_rates_to_add_as_obs,
                                             add_time_as_obs=add_time_as_obs,
                                             agent_speed_limit=0.75)

    else:
        env = gym.make(task_name)

    return env


def unity_rendering_burn_in(env, n_episodes=8, n_steps=100):
    """
    With the Windows executable, the environment
    seems to need a few hundred steps before rendering properly.
    """
    env.reset()

    for ep_i in range(n_episodes):
        time.sleep(0.5)

        for step_i in range(n_steps):
            action = np.random.rand(env.action_space.shape[0])
            s, r, done, info = env.step(action)
            env.render('human')
            time.sleep(0.02)

            if done: break

        env.reset()

    return


def constraints_are_satisfied(constraint_measurements, constraint_thresholds=None):
    if constraint_thresholds is not None:
        assert len(constraint_measurements) == len(constraint_thresholds)
        are_satisfied = []
        for k in range(len(constraint_measurements)):
            if not math.isnan(constraint_thresholds[k][0]) and not math.isnan(constraint_thresholds[k][1]):  # double bounds
                are_satisfied.append(constraint_measurements[k] > constraint_thresholds[k][0] and \
                                     constraint_measurements[k] < constraint_thresholds[k][1])

            elif math.isnan(constraint_thresholds[k][0]) and not math.isnan(constraint_thresholds[k][1]):  # upper bound
                are_satisfied.append(constraint_measurements[k] < constraint_thresholds[k][1])

            elif not math.isnan(constraint_thresholds[k][0]) and math.isnan(constraint_thresholds[k][1]):  # lower bound
                are_satisfied.append(constraint_measurements[k] > constraint_thresholds[k][0])

            else:
                raise ValueError()

    else:
        are_satisfied = [None for _ in constraint_measurements]

    return are_satisfied


def get_model_file(dir_tree, model_type):
    model_files = [path for path in dir_tree.seed_dir.iterdir() if ".pt" in path.name]
    model_file = [path for path in model_files if model_type in path.name]
    if len(model_file) == 0:
        return None
    else:
        assert len(model_file) == 1
        return model_file[0]


def get_cost_vector(info, constraints_to_enforce, constraint_is_reversed):
    cost = []
    for k, constraint_name in enumerate(constraints_to_enforce):
        if math.isnan(info[constraint_name]):
            cost.append(info[constraint_name])
        else:
            assert info[constraint_name] == 1. or info[constraint_name] == 0.
            cost.append(1. - info[constraint_name] if constraint_is_reversed[k] else info[constraint_name])

    return cost
