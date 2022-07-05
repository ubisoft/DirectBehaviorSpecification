import numpy as np
from collections import OrderedDict
from alfred.utils.misc import keep_two_signif_digits, check_params_defined_twice
from alfred.utils.directory_tree import DirectoryTree
from pathlib import Path
import packageName

# (1) Enter the algorithms to be run for each experiment

ALG_NAMES = ['simpleMLP']

# (2) Enter the task (dataset or rl-environment) to be used for each experiment

TASK_NAMES = ['MNIST']

# (3) Enter the seeds to be run for each experiment

N_SEEDS = 3
SEEDS = [1 + x for x in range(N_SEEDS)]

# (4) Enter the number of experiments to sample

N_EXPERIMENTS = 50

# (5) Hyper-parameters. For each hyperparam, enter the function that you want the random-search to sample from.
#     For each experiment, a set of hyperparameters will be sampled using these functions

# Examples:
# int:          np.random.randint(low=64, high=512)
# float:        np.random.uniform(low=-3., high=1.)
# bool:         bool(np.random.binomial(n=1, p=0.5))
# exp_float:    10.**np.random.uniform(low=-3., high=1.)
# fixed_value:  fixed_value

def sample_experiment():
   sampled_config = OrderedDict({
       'learning_rate': 10. ** np.random.uniform(low=-8., high=-3.),
       'optimizer': "sgd",
   })

   # Security check to make sure seed, alg_name and task_name are not defined as hyperparams

   assert "seed" not in sampled_config.keys()
   assert "alg_name" not in sampled_config.keys()
   assert "task_name" not in sampled_config.keys()

   # Simple security check to make sure every specified parameter is defined only once

   check_params_defined_twice(keys=list(sampled_config.keys()))


# (6) Function that returns the hyperparameters for the current search

def get_run_args(overwritten_cmd_line):
    raise NotImplementedError

# Setting up alfred's DirectoryTree

DirectoryTree.default_root = "./storage"
DirectoryTree.git_repos_to_track['mlProject'] = str(Path(__file__).parents[2])
DirectoryTree.git_repos_to_track['someDependency'] = str(Path(packageName.__file__).parents[1])
