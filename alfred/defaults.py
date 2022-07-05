from collections import OrderedDict

DEFAULT_BENCHMARK_X_METRIC = "episode"
DEFAULT_BENCHMARK_Y_METRIC = "eval_return"
DEFAULT_BENCHMARK_PERFORMANCE_METRIC = "eval_return"

DEFAULT_DIRECTORY_TREE_ROOT = './storage'
DEFAULT_DIRECTORY_TREE_GIT_REPOS_TO_TRACK = OrderedDict()

DEFAULT_PLOTS_ARRAYS_TO_MAKE = [('episode', 'eval_return', (None, None), (None, None)),
                                ('episode', 'return', (None, None), (None, None))]
