from alfred.utils.misc import check_params_defined_twice
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

# (4) Hyper-parameters

# Here, for each hyperparam, enter the values you want to try in a list.
# All possible combinations will be run as a separate experiment
# Unspecified (or commented out) params will be set to default defines in main.get_training_args

VARIATIONS = {
    'learning_rate': [0.1, 0.01, 0.001],
    'optimizer': ["sgd", "adam"],
}

# Security check to make sure seed, alg_name and task_name are not defined as hyperparams

assert "seed" not in VARIATIONS.keys()
assert "alg_name" not in VARIATIONS.keys()
assert "task_name" not in VARIATIONS.keys()

# Simple security check to make sure every specified parameter is defined only once

check_params_defined_twice(keys=list(VARIATIONS.keys()))


# (5) Function that returns the hyperparameters for the current search

def get_run_args(overwritten_cmd_line):
    raise NotImplementedError

# Setting up alfred's DirectoryTree

DirectoryTree.default_root = "./storage"
DirectoryTree.git_repos_to_track['mlProject'] = str(Path(__file__).parents[2])
DirectoryTree.git_repos_to_track['someDependency'] = str(Path(packageName.__file__).parents[1])
