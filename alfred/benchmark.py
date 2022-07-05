from alfred.utils.config import parse_bool, load_dict_from_json, save_dict_to_json, parse_log_level
from alfred.utils.misc import create_logger, select_storage_dirs
from alfred.utils.directory_tree import DirectoryTree, get_root, sanity_check_exists
from alfred.utils.recorder import Recorder, remove_nones
from alfred.utils.plots import create_fig, bar_chart, plot_curves, plot_vertical_densities
from alfred.utils.stats import get_95_confidence_interval_of_sequence, get_95_confidence_interval
import alfred.defaults

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import pickle
import os
import logging
from shutil import copyfile
from collections import OrderedDict
import seaborn as sns
import pathlib
from pathlib import Path
import shutil
import math

sns.set()
sns.set_style('whitegrid')


def get_benchmark_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_type', type=str, choices=['compare_models', 'compare_searches'], required=True)
    parser.add_argument('--log_level', type=parse_log_level, default=logging.INFO)

    parser.add_argument('--from_file', type=str, default=None)
    parser.add_argument('--storage_names', type=str, nargs='+', default=None)

    parser.add_argument('--x_metric', default=alfred.defaults.DEFAULT_BENCHMARK_X_METRIC, type=str)
    parser.add_argument('--y_metric', default=alfred.defaults.DEFAULT_BENCHMARK_Y_METRIC, type=str)
    parser.add_argument('--y_error_bars', default="bootstrapped_CI",
                        choices=["bootstrapped_CI", "stderr", "10th_quantile", "None"])

    parser.add_argument('--re_run_if_exists', type=parse_bool, default=False,
                        help="Whether to re-compute seed_scores if 'seed_scores.pkl' already exists")
    parser.add_argument('--n_eval_runs', type=int, default=100,
                        help="Only used if performance_metric=='evaluation_runs'")
    parser.add_argument('--performance_metric', type=str, default=alfred.defaults.DEFAULT_BENCHMARK_PERFORMANCE_METRIC,
                        help="Can fall into either of two categories: "
                             "(1) 'evaluation_runs': evaluate() will be called on model in seed_dir for 'n_eval_runs'"
                             "(2) OTHER METRIC: this metric must have been recorded in training and be a key of train_recorder")
    parser.add_argument('--performance_aggregation', type=str, choices=['min', 'max', 'avg', 'last',
                                                                        'mean_on_last_20_percents'],
                        default='mean_on_last_20_percents',
                        help="How gathered 'performance_metric' should be aggregated to quantify performance of seed_dir")

    parser.add_argument('--root_dir', default="./storage", type=str)

    return parser.parse_args()


# utility functions for curves (should not be called alone) -------------------------------------------------------

def _compute_seed_scores(storage_dir, performance_metric, performance_aggregation, group_key, bar_key,
                         re_run_if_exists, save_dir, logger, root_dir, n_eval_runs):
    if (storage_dir / save_dir / f"{save_dir}_seed_scores.pkl").exists() and not re_run_if_exists:
        logger.info(f" SKIPPING {storage_dir} - {save_dir}_seed_scores.pkl already exists")
        return

    else:
        logger.info(f"Benchmarking {storage_dir}...")

    assert group_key in ['task_name', 'storage_name', 'experiment_num', 'alg_name']
    assert bar_key in ['task_name', 'storage_name', 'experiment_num', 'alg_name']

    # Initialize container

    scores = OrderedDict()

    # Get all experiment directories

    all_experiments = DirectoryTree.get_all_experiments(storage_dir=storage_dir)

    for experiment_dir in all_experiments:

        # For that experiment, get all seed directories

        experiment_seeds = DirectoryTree.get_all_seeds(experiment_dir=experiment_dir)

        # Initialize container

        all_seeds_scores = []

        for i, seed_dir in enumerate(experiment_seeds):
            # Prints which seed directory is being treated

            logger.debug(f"{seed_dir}")

            # Loads training config

            config_dict = load_dict_from_json(str(seed_dir / "config.json"))

            # Selects how data will be identified

            keys = {
                "task_name": config_dict["task_name"],
                "storage_name": seed_dir.parents[1].name,
                "alg_name": config_dict["alg_name"],
                "experiment_num": seed_dir.parents[0].name.strip('experiment')
            }

            outer_key = keys[bar_key]
            inner_key = keys[group_key]

            # Evaluation phase

            if performance_metric == 'evaluation_runs':

                assert n_eval_runs is not None

                try:
                    from evaluate import evaluate, get_evaluation_args
                except ImportError as e:
                    raise ImportError(
                        f"{e}\nTo evaluate models based on --performance_metric='evaluation_runs' "
                        f"alfred.benchmark assumes the following structure that the working directory contains "
                        f"a file called evaluate.py containing two functions: "
                        f"\n\t1. a function evaluate() that returns a score for each evaluation run"
                        f"\n\t2. a function get_evaluation_args() that returns a Namespace of arguments for evaluate()"
                    )

                # Sets config for evaluation phase

                eval_config = get_evaluation_args(overwritten_args="")
                eval_config.storage_name = seed_dir.parents[1].name
                eval_config.experiment_num = int(seed_dir.parents[0].name.strip("experiment"))
                eval_config.seed_num = int(seed_dir.name.strip("seed"))
                eval_config.render = False
                eval_config.n_episodes = n_eval_runs
                eval_config.root_dir = root_dir

                # Evaluates agent and stores the return

                performance_data = evaluate(eval_config)

            else:

                # Loads training data

                loaded_recorder = Recorder.init_from_pickle_file(
                    filename=str(seed_dir / 'recorders' / 'train_recorder.pkl'))

                performance_data = remove_nones(loaded_recorder.tape[performance_metric])

            # Aggregation phase

            if performance_aggregation == 'min':
                score = np.min(performance_data)

            elif performance_aggregation == 'max':
                score = np.max(performance_data)

            elif performance_aggregation == 'avg':
                score = np.mean(performance_data)

            elif performance_aggregation == 'last':
                score = performance_data[-1]

            elif performance_aggregation == 'mean_on_last_20_percents':
                eighty_percent_index = int(0.8*len(performance_data))
                score = np.mean(performance_data[eighty_percent_index:])
            else:
                raise NotImplementedError

            all_seeds_scores.append(score)

        if outer_key not in scores.keys():
            scores[outer_key] = OrderedDict()

        scores[outer_key][inner_key] = np.stack(all_seeds_scores)

    os.makedirs(storage_dir / save_dir, exist_ok=True)

    with open(storage_dir / save_dir / f"{save_dir}_seed_scores.pkl", "wb") as f:
        pickle.dump(scores, f)

    scores_info = {'n_eval_runs': n_eval_runs,
                   'performance_metric': performance_metric,
                   'performance_aggregation': performance_aggregation}

    save_dict_to_json(scores_info, filename=str(storage_dir / save_dir / f"{save_dir}_seed_scores_info.json"))


def _gather_scores(storage_dirs, save_dir, y_error_bars, logger, normalize_with_first_model=True, sort_bars=False):
    # Initialize containers

    scores_means = OrderedDict()
    scores_err_up = OrderedDict()
    scores_err_down = OrderedDict()

    # Loads performance benchmark data

    individual_scores = OrderedDict()
    for storage_dir in storage_dirs:
        with open(storage_dir / save_dir / f"{save_dir}_seed_scores.pkl", "rb") as f:
            individual_scores[storage_dir.name] = pickle.load(f)

    # Print keys so that user can verify all these benchmarks make sense to compare (e.g. same tasks)

    for storage_name, idv_score in individual_scores.items():
        logger.debug(storage_name)
        for outer_key in idv_score.keys():
            logger.debug(f"{outer_key}: {list(idv_score[outer_key].keys())}")
        logger.debug(f"\n")

    # Reorganize all individual_scores in a single dictionary

    scores = OrderedDict()
    for storage_name, idv_score in individual_scores.items():
        for outer_key in idv_score:
            if outer_key not in list(scores.keys()):
                scores[outer_key] = OrderedDict()
            for inner_key in idv_score[outer_key]:
                if inner_key not in list(scores.keys()):
                    scores[outer_key][inner_key] = OrderedDict()
                _, _, _, task_name, _ = DirectoryTree.extract_info_from_storage_name(storage_name)
                scores[outer_key][inner_key] = idv_score[outer_key][inner_key]

    # First storage_dir will serve as reference if normalize_with_first_model is True

    reference_key = list(scores.keys())[0]
    reference_means = OrderedDict()
    for inner_key in scores[reference_key].keys():
        if normalize_with_first_model:
            reference_means[inner_key] = scores[reference_key][inner_key].mean()
        else:
            reference_means[inner_key] = 1.

    # Sorts inner_keys (bars among groups)

    sorted_inner_keys = list(reversed(sorted(reference_means.keys(),
                                             key=lambda item: (scores[reference_key][item].mean(), item))))

    if sort_bars:
        inner_keys = sorted_inner_keys
    else:
        inner_keys = scores[reference_key].keys()

    # Computes means and error bars

    for inner_key in inner_keys:
        for outer_key in scores.keys():
            if outer_key not in scores_means.keys():
                scores_means[outer_key] = OrderedDict()
                scores_err_up[outer_key] = OrderedDict()
                scores_err_down[outer_key] = OrderedDict()

            if y_error_bars == "stderr":
                scores_means[outer_key][inner_key] = np.mean(
                    scores[outer_key][inner_key] / (reference_means[inner_key] + 1e-8))

                scores_err_down[outer_key][inner_key] = np.std(
                    scores[outer_key][inner_key] / (reference_means[inner_key] + 1e-8)) / len(
                    scores[outer_key][inner_key]) ** 0.5
                scores_err_up[outer_key][inner_key] = scores_err_down[outer_key][inner_key]

            elif y_error_bars == "10th_quantiles":
                scores_means[outer_key][inner_key] = np.mean(
                    scores[outer_key][inner_key] / (reference_means[inner_key] + 1e-8))

                quantile = 0.10
                scores_err_down[outer_key][inner_key] = np.abs(
                    np.quantile(a=scores[outer_key][inner_key] / (reference_means[inner_key] + 1e-8), q=0. + quantile) \
                    - scores_means[outer_key][inner_key])
                scores_err_up[outer_key][inner_key] = np.abs(
                    np.quantile(a=scores[outer_key][inner_key] / (reference_means[inner_key] + 1e-8), q=1. - quantile) \
                    - scores_means[outer_key][inner_key])

            elif y_error_bars == "bootstrapped_CI":
                scores_samples = scores[outer_key][inner_key] / (reference_means[inner_key] + 1e-8)

                mean, err_up, err_down = get_95_confidence_interval(samples=scores_samples, method=y_error_bars)
                scores_means[outer_key][inner_key] = mean
                scores_err_up[outer_key][inner_key] = err_up
                scores_err_down[outer_key][inner_key] = err_down

            elif y_error_bars == "None":
                scores_means[outer_key][inner_key] = np.mean(
                    scores[outer_key][inner_key] / (reference_means[inner_key] + 1e-8))
                scores_err_down[outer_key][inner_key] = None
                scores_err_up[outer_key][inner_key] = None

            else:
                raise NotImplementedError

    return scores, scores_means, scores_err_up, scores_err_down, sorted_inner_keys, reference_key


def _make_benchmark_performance_figure(storage_dirs, save_dir, y_error_bars, logger, normalize_with_first_model=True,
                                       sort_bars=False):
    scores, scores_means, scores_err_up, scores_err_down, sorted_inner_keys, reference_key = _gather_scores(
        storage_dirs=storage_dirs,
        save_dir=save_dir,
        y_error_bars=y_error_bars,
        logger=logger,
        normalize_with_first_model=normalize_with_first_model,
        sort_bars=sort_bars)

    # Creates the graph

    n_bars_per_group = len(scores_means.keys())
    n_groups = len(scores_means[reference_key].keys())
    fig, ax = create_fig((1, 1), figsize=(n_bars_per_group * n_groups, n_groups))

    bar_chart(ax,
              scores=scores_means,
              err_up=scores_err_up,
              err_down=scores_err_down,
              group_names=scores_means[reference_key].keys(),
              title="Average Return"
              )

    n_training_seeds = scores[reference_key][list(scores_means[reference_key].keys())[0]].shape[0]

    scores_info = load_dict_from_json(filename=str(storage_dirs[0] / save_dir / f"{save_dir}_seed_scores_info.json"))

    info_str = f"{n_training_seeds} training seeds" \
               f"\nn_eval_runs={scores_info['n_eval_runs']}" \
               f"\nperformance_metric={scores_info['performance_metric']}" \
               f"\nperformance_aggregation={scores_info['performance_aggregation']}"

    ax.text(0.80, 0.95, info_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='gray', alpha=0.1))

    plt.tight_layout()

    # Saves storage_dirs from which the graph was created for traceability

    for storage_dir in storage_dirs:
        os.makedirs(storage_dir / save_dir, exist_ok=True)
        fig.savefig(storage_dir / save_dir / f'{save_dir}_performance.png')
        save_dict_to_json({'sources': str(storage_dir) in storage_dirs,
                           'n_training_seeds': n_training_seeds,
                           'n_eval_runs': scores_info['n_eval_runs'],
                           'performance_metric': scores_info['performance_metric'],
                           'performance_aggregation': scores_info['performance_aggregation']
                           },
                          storage_dir / save_dir / f'{save_dir}_performance_sources.json')

    plt.close(fig)

    # SANITY-CHECKS that no seeds has a Nan score to avoid making create best on it

    expe_with_nan_scores=[]
    for outer_key in scores.keys():
        for inner_key, indiv_score in scores[outer_key].items():
            if math.isnan(indiv_score.mean()):
                expe_with_nan_scores.append(outer_key+"/experiment" + inner_key)

    if len(expe_with_nan_scores)>0:
        raise ValueError(f'Some experiments have nan scores. Remove them from storage and clean summary folder to continue\n'
                         f'experiments with Nan Scores:\n' + '\n'.join(expe_with_nan_scores))

    return sorted_inner_keys


def _gather_experiments_training_curves(storage_dir, graph_key, curve_key, logger, x_metric, y_metric,
                                        x_data=None, y_data=None):

    # Initialize containers

    if x_data is None:
        x_data = OrderedDict()
    else:
        assert type(x_data) is OrderedDict

    if y_data is None:
        y_data = OrderedDict()
    else:
        assert type(y_data) is OrderedDict

    # Get all experiment directories

    all_experiments = DirectoryTree.get_all_experiments(storage_dir=storage_dir)

    for experiment_dir in all_experiments:

        # For that experiment, get all seed directories

        experiment_seeds = DirectoryTree.get_all_seeds(experiment_dir=experiment_dir)

        for i, seed_dir in enumerate(experiment_seeds):

            # Prints which seed directory is being treated

            logger.debug(f"{seed_dir}")

            # Loads training config

            config_dict = load_dict_from_json(str(seed_dir / "config.json"))

            # Keys can be any information stored in config.json
            # We also handle a few special cases (e.g. "experiment_num")

            keys = config_dict.copy()
            keys['experiment_num'] = seed_dir.parent.stem.strip('experiment')
            keys['storage_name'] = seed_dir.parents[1]

            outer_key = keys[graph_key]  # number of graphs to be made
            inner_key = keys[curve_key]  # number of curves per graph

            # Loads training data

            loaded_recorder = Recorder.init_from_pickle_file(
                filename=str(seed_dir / 'recorders' / 'train_recorder.pkl'))

            # Stores the data

            if outer_key not in y_data.keys():
                x_data[outer_key] = OrderedDict()
                y_data[outer_key] = OrderedDict()

            if inner_key not in y_data[outer_key].keys():
                x_data[outer_key][inner_key] = []
                y_data[outer_key][inner_key] = []

            x_data[outer_key][inner_key].append(loaded_recorder.tape[x_metric])
            y_data[outer_key][inner_key].append(loaded_recorder.tape[y_metric])  # TODO: make sure that this is a scalar metric, even for eval_return (and not 10 points for every eval_step). All metrics saved in the recorder should be scalars for every time point.

    return x_data, y_data


def _make_benchmark_learning_figure(x_data, y_data, x_metric, y_metric, y_error_bars, storage_dirs, save_dir, logger,
                                    n_labels=np.inf, visuals_file=None, additional_curves_file=None):
    # Initialize containers

    y_data_means = OrderedDict()
    y_data_err_up = OrderedDict()
    y_data_err_down = OrderedDict()
    long_labels = OrderedDict()
    titles = OrderedDict()
    x_axis_titles = OrderedDict()
    y_axis_titles = OrderedDict()
    labels = OrderedDict()
    colors = OrderedDict()
    markers = OrderedDict()

    for outer_key in y_data:
        y_data_means[outer_key] = OrderedDict()
        y_data_err_up[outer_key] = OrderedDict()
        y_data_err_down[outer_key] = OrderedDict()

    # Initialize figure

    n_graphs = len(y_data.keys())

    if n_graphs == 3:
        axes_shape = (1, 3)

    elif n_graphs > 1:
        i_max = int(np.ceil(np.sqrt(len(y_data.keys()))))
        axes_shape = (int(np.ceil(len(y_data.keys()) / i_max)), i_max)
    else:
        axes_shape = (1, 1)

    # Creates figure

    gs = gridspec.GridSpec(*axes_shape)
    fig = plt.figure(figsize=(8 * axes_shape[1], 4 * axes_shape[0]))

    # Remove nones

    for data in [x_data, y_data]:
        for outer_key in data.keys():
            for inner_key in data[outer_key].keys():
                for seed_i, seed_data in enumerate(data[outer_key][inner_key]):
                    data[outer_key][inner_key][seed_i] = remove_nones(seed_data)

    # Compute means and stds for all inner_key curve from raw data

    for i, outer_key in enumerate(y_data.keys()):
        for inner_key in y_data[outer_key].keys():

            if y_error_bars == "stderr":
                x_data[outer_key][inner_key] = x_data[outer_key][inner_key][0]  # assumes all x_data are the same
                y_data_means[outer_key][inner_key] = np.stack(y_data[outer_key][inner_key], axis=-1).mean(-1)
                y_data_err_up[outer_key][inner_key] = np.stack(y_data[outer_key][inner_key], axis=-1).std(-1) \
                                                      / len(y_data[outer_key][inner_key]) ** 0.5
                y_data_err_down = y_data_err_up

            elif y_error_bars == "bootstrapped_CI":
                x_data[outer_key][inner_key] = x_data[outer_key][inner_key][0]  # assumes all x_data are the same
                y_data_samples = np.stack(y_data[outer_key][inner_key],
                                          axis=-1)  # dim=0 is accross time (n_time_steps, n_samples)
                mean, err_up, err_down = get_95_confidence_interval_of_sequence(list_of_samples=y_data_samples,
                                                                                method=y_error_bars)
                y_data_means[outer_key][inner_key] = mean
                y_data_err_up[outer_key][inner_key] = err_up
                y_data_err_down[outer_key][inner_key] = err_down

            elif y_error_bars == "None":
                y_data_means[outer_key][inner_key] = y_data[outer_key][inner_key]
                y_data_err_up[outer_key][inner_key] = None
                y_data_err_down[outer_key][inner_key] = None

                # Transpose list of lists (necessary for matplotlib to properly plot all curves in one call)
                # see: https://stackoverflow.com/questions/6473679/transpose-list-of-lists
                y_data_means[outer_key][inner_key] = list(map(list, zip(*y_data_means[outer_key][inner_key])))
                x_data[outer_key][inner_key] = list(map(list, zip(*x_data[outer_key][inner_key])))

            else:
                raise NotImplementedError

        long_labels[outer_key] = list(y_data_means[outer_key].keys())

        # Limits the number of labels to be displayed (only displays labels of n_labels best experiments)

        if n_labels < np.inf:
            mean_over_entire_curves = np.array([array.mean() for array in y_data_means[outer_key].values()])
            n_max_idxs = (-mean_over_entire_curves).argsort()[:n_labels]

            for k in range(len(long_labels[outer_key])):
                if k in n_max_idxs:
                    continue
                else:
                    long_labels[outer_key][k] = None

        # Selects right ax object

        if axes_shape == (1, 1):
            current_ax = fig.add_subplot(gs[0, 0])
        elif any(np.array(axes_shape) == 1):
            current_ax = fig.add_subplot(gs[0, i])
        else:
            current_ax = fig.add_subplot(gs[i // axes_shape[1], i % axes_shape[1]])

        # Collect algorithm names

        if all([type(long_label) is pathlib.PosixPath for long_label in long_labels[outer_key]]):
            algs = []
            for path in long_labels[outer_key]:
                _, _, alg, _, _ = DirectoryTree.extract_info_from_storage_name(path.name)
                algs.append(alg)

        # Loads visuals dictionaries

        if visuals_file is not None:
            visuals = load_dict_from_json(visuals_file)
        else:
            visuals = None

        # Loads additional curves file

        if additional_curves_file is not None:
            additional_curves = load_dict_from_json(additional_curves_file)
        else:
            additional_curves = None

        # Sets visuals

        if type(visuals) is dict and 'titles_dict' in visuals.keys():
            titles[outer_key] = visuals['titles_dict'][outer_key]
        else:
            titles[outer_key] = outer_key

        if type(visuals) is dict and 'axis_titles_dict' in visuals.keys():
            x_axis_titles[outer_key] = visuals['axis_titles_dict'][x_metric]
            y_axis_titles[outer_key] = visuals['axis_titles_dict'][y_metric]
        else:
            x_axis_titles[outer_key] = x_metric
            y_axis_titles[outer_key] = y_metric

        if type(visuals) is dict and 'labels_dict' in visuals.keys():
            labels[outer_key] = [visuals['labels_dict'][inner_key] for inner_key in y_data_means[outer_key].keys()]
        else:
            labels[outer_key] = long_labels[outer_key]

        if type(visuals) is dict and 'colors_dict' in visuals.keys():
            colors[outer_key] = [visuals['colors_dict'][inner_key] for inner_key in y_data_means[outer_key].keys()]
        else:
            colors[outer_key] = [None for _ in long_labels[outer_key]]

        if type(visuals) is dict and 'markers_dict' in visuals.keys():
            markers[outer_key] = [visuals['markers_dict'][inner_key] for inner_key in y_data_means[outer_key].keys()]
        else:
            markers[outer_key] = [None for _ in long_labels[outer_key]]

        logger.info(f"Graph for {outer_key}:\n\tlabels={labels}\n\tcolors={colors}\n\tmarkers={markers}")

        if additional_curves_file is not None:
            hlines = additional_curves['hlines'][outer_key]
            n_lines = len(hlines)
        else:
            hlines = None
            n_lines = 0

        # Plots the curves

        plot_curves(current_ax,
                    xs=list(x_data[outer_key].values()),
                    ys=list(y_data_means[outer_key].values()),
                    fill_up=list(y_data_err_up[outer_key].values()) if y_error_bars != "None" else None,
                    fill_down=list(y_data_err_down[outer_key].values()) if y_error_bars != "None" else None,
                    labels=labels[outer_key],
                    colors=colors[outer_key],
                    markers=markers[outer_key],
                    xlabel=x_axis_titles[outer_key],
                    ylabel=y_axis_titles[outer_key] if i == 0 else "",
                    title=titles[outer_key].upper(),
                    add_legend=True if i == (len(list(y_data.keys())) - 1) else False,
                    legend_outside=True,
                    legend_loc="upper right",
                    legend_pos=(0.95, -0.2),
                    legend_n_columns=len(list(y_data_means[outer_key].values())) + n_lines,
                    hlines=hlines,
                    tick_font_size=22,
                    axis_font_size=26,
                    legend_font_size=26,
                    title_font_size=28)

    plt.tight_layout()

    for storage_dir in storage_dirs:
        os.makedirs(storage_dir / save_dir, exist_ok=True)
        fig.savefig(storage_dir / save_dir / f'{save_dir}_learning.pdf', bbox_inches='tight')

    plt.close(fig)


def _make_vertical_densities_figure(storage_dirs, visuals_file, additional_curves_file, make_box_plot, queried_performance_metric,
                                    queried_performance_aggregation, save_dir, load_dir, logger):
    # Initialize container

    all_means = OrderedDict()
    long_labels = OrderedDict()
    titles = OrderedDict()
    labels = OrderedDict()
    colors = OrderedDict()
    markers = OrderedDict()
    all_performance_metrics = []
    all_performance_aggregation = []

    # Gathers data

    for storage_dir in storage_dirs:
        logger.debug(storage_dir)

        # Loads the scores and scores_info saved by summarize_search

        with open(str(storage_dir / load_dir / f"{load_dir}_seed_scores.pkl"), "rb") as f:
            scores = pickle.load(f)

        scores_info = load_dict_from_json(str(storage_dir / "summary" / f"summary_seed_scores_info.json"))
        all_performance_metrics.append(scores_info['performance_metric'])
        all_performance_aggregation.append(scores_info['performance_aggregation'])

        x = list(scores.keys())[0]
        storage_name = storage_dir.name

        # Adding task_name if first time it is encountered

        _, _, _, outer_key, _ = DirectoryTree.extract_info_from_storage_name(storage_name)
        if outer_key not in list(all_means.keys()):
            all_means[outer_key] = OrderedDict()

        # Taking the mean across evaluations and seeds

        _, _, _, outer_key, _ = DirectoryTree.extract_info_from_storage_name(storage_name)
        all_means[outer_key][storage_name] = [array.mean() for array in scores[x].values()]

        if outer_key not in long_labels.keys():
            long_labels[outer_key] = [storage_dir]
        else:
            long_labels[outer_key].append(storage_dir)

    # Security checks

    assert len(set(all_performance_metrics)) == 1 and len(set(all_performance_aggregation)) == 1, \
        "Error: all seeds do not have scores computed using the same performance metric or performance aggregation. " \
        "You should benchmark with --re_run_if_exists=True using the desired --performance_aggregation and " \
        "--performance_metric so that all seeds that you want to compare have the same metrics."
    actual_performance_metric = all_performance_metrics.pop()
    actual_performance_aggregation = all_performance_aggregation.pop()

    assert queried_performance_metric == actual_performance_metric and \
           queried_performance_aggregation == actual_performance_aggregation, \
        "Error: The performance_metric or performance_aggregation that was queried for the vertical_densities " \
        "is not the same as what was saved by summarize_search. You should benchmark with --re_run_if_exists=True " \
        "using the desired --performance_aggregation and  --performance_metric so that all seeds that you want " \
        "to compare have the same metrics."

    # Initialize figure

    n_graphs = len(all_means.keys())

    if n_graphs == 3:
        axes_shape = (1, 3)

    elif n_graphs > 1:
        i_max = int(np.ceil(np.sqrt(len(all_means.keys()))))
        axes_shape = (int(np.ceil(len(all_means.keys()) / i_max)), i_max)
    else:
        axes_shape = (1, 1)

    # Creates figure

    gs = gridspec.GridSpec(*axes_shape)
    fig = plt.figure(figsize=(12 * axes_shape[1], 5 * axes_shape[0]))

    for i, outer_key in enumerate(all_means.keys()):

        # Selects right ax object

        if axes_shape == (1, 1):
            current_ax = fig.add_subplot(gs[0, 0])
        elif any(np.array(axes_shape) == 1):
            current_ax = fig.add_subplot(gs[0, i])
        else:
            current_ax = fig.add_subplot(gs[i // axes_shape[1], i % axes_shape[1]])

        # Collect algorithm names

        if all([type(long_label) is pathlib.PosixPath for long_label in long_labels[outer_key]]):
            algs = []
            for path in long_labels[outer_key]:
                _, _, alg, _, _ = DirectoryTree.extract_info_from_storage_name(path.name)
                algs.append(alg)

        # Loads visuals dictionaries

        if visuals_file is not None:
            visuals = load_dict_from_json(visuals_file)
        else:
            visuals = None

        # Loads additional curves file

        if additional_curves_file is not None:
            additional_curves = load_dict_from_json(additional_curves_file)
        else:
            additional_curves = None

        # Sets visuals

        if type(visuals) is dict and 'titles_dict' in visuals.keys():
            titles[outer_key] = visuals['titles_dict'][outer_key]
        else:
            titles[outer_key] = outer_key

        if type(visuals) is dict and 'labels_dict' in visuals.keys():
            labels[outer_key] = [visuals['labels_dict'][alg] for alg in algs]
        else:
            labels[outer_key] = long_labels[outer_key]

        if type(visuals) is dict and 'colors_dict' in visuals.keys():
            colors[outer_key] = [visuals['colors_dict'][alg] for alg in algs]
        else:
            colors[outer_key] = [None for _ in long_labels[outer_key]]

        if type(visuals) is dict and 'markers_dict' in visuals.keys():
            markers[outer_key] = [visuals['markers_dict'][alg] for alg in algs]
        else:
            markers[outer_key] = [None for _ in long_labels[outer_key]]

        logger.info(f"Graph for {outer_key}:\n\tlabels={labels}\n\tcolors={colors}\n\tmarkers={markers}")

        if additional_curves_file is not None:
            hlines = additional_curves['hlines'][outer_key]
        else:
            hlines = None

        # Makes the plots

        plot_vertical_densities(ax=current_ax,
                                ys=list(all_means[outer_key].values()),
                                labels=labels[outer_key],
                                colors=colors[outer_key],
                                make_boxplot=make_box_plot,
                                title=titles[outer_key].upper(),
                                ylabel=f"{actual_performance_aggregation}-{actual_performance_metric}",
                                hlines=hlines)

    # Saves the figure

    plt.tight_layout()

    filename_addon = "boxplot" if make_box_plot else ""

    for storage_dir in storage_dirs:
        os.makedirs(storage_dir / save_dir, exist_ok=True)

        fig.savefig(storage_dir / save_dir / f'{save_dir}_vertical_densities_{filename_addon}.pdf', bbox_inches="tight")

        save_dict_to_json([str(storage_dir) in storage_dirs],
                          storage_dir / save_dir / f'{save_dir}_vertical_densities_sources.json')

    plt.close(fig)


# benchmark interface ---------------------------------------------------------------------------------------------

def compare_models(storage_names, n_eval_runs, re_run_if_exists, logger, root_dir, x_metric, y_metric, y_error_bars,
                   visuals_file, additional_curves_file, performance_metric, performance_aggregation,
                   make_performance_chart=True, make_learning_plots=True):
    """
    compare_models compare several storage_dirs
    """

    assert type(storage_names) is list

    if make_learning_plots:
        logger.debug(f'\n{"benchmark_learning".upper()}:')

        x_data = OrderedDict()
        y_data = OrderedDict()
        storage_dirs = []

        for storage_name in storage_names:
            x_data, y_data = _gather_experiments_training_curves(
                storage_dir=get_root(root_dir) / storage_name,
                graph_key="task_name",
                curve_key="storage_name" if logger.level == 10 else "alg_name",
                logger=logger,
                x_metric=x_metric,
                y_metric=y_metric,
                x_data=x_data,
                y_data=y_data)

            storage_dirs.append(get_root(root_dir) / storage_name)

        _make_benchmark_learning_figure(x_data=x_data,
                                        y_data=y_data,
                                        x_metric=x_metric,
                                        y_metric=y_metric,
                                        y_error_bars=y_error_bars,
                                        storage_dirs=storage_dirs,
                                        n_labels=np.inf,
                                        save_dir="benchmark",
                                        logger=logger,
                                        visuals_file=visuals_file,
                                        additional_curves_file=additional_curves_file)

    if make_performance_chart:
        logger.debug(f'\n{"benchmark_performance".upper()}:')

        storage_dirs = []

        for storage_name in storage_names:
            _compute_seed_scores(storage_dir=get_root(root_dir) / storage_name,
                                 performance_metric=performance_metric,
                                 performance_aggregation=performance_aggregation,
                                 n_eval_runs=n_eval_runs,
                                 group_key="task_name",
                                 bar_key="storage_name" if logger.level == 10 else "alg_name",
                                 re_run_if_exists=re_run_if_exists,
                                 save_dir="benchmark",
                                 logger=logger,
                                 root_dir=root_dir)

            storage_dirs.append(get_root(root_dir) / storage_name)

        _make_benchmark_performance_figure(storage_dirs=storage_dirs,
                                           logger=logger,
                                           normalize_with_first_model=True,
                                           sort_bars=False,
                                           y_error_bars=y_error_bars,
                                           save_dir="benchmark")

    return


def summarize_search(storage_name, n_eval_runs, re_run_if_exists, logger, root_dir, x_metric, y_metric, y_error_bars,
                     performance_metric, performance_aggregation, make_performance_chart=True,
                     make_learning_plots=True):
    """
    Summaries act inside a single storage_dir
    """

    assert type(storage_name) is str

    storage_dir = get_root(root_dir) / storage_name

    if re_run_if_exists and (storage_dir / "summary").exists():
        shutil.rmtree(storage_dir / "summary")


    if make_learning_plots:
        logger.debug(f'\n{"benchmark_learning".upper()}:')

        x_data, y_data = _gather_experiments_training_curves(storage_dir=storage_dir,
                                                             graph_key="task_name",
                                                             curve_key="experiment_num",
                                                             logger=logger,
                                                             x_metric=x_metric,
                                                             y_metric=y_metric)

        _make_benchmark_learning_figure(x_data=x_data,
                                        y_data=y_data,
                                        x_metric=x_metric,
                                        y_metric=y_metric,
                                        y_error_bars=y_error_bars,
                                        storage_dirs=[storage_dir],
                                        n_labels=10,
                                        save_dir="summary",
                                        logger=logger,
                                        visuals_file=None)

    if make_performance_chart:
        logger.debug(f'\n{"benchmark_performance".upper()}:')

        _compute_seed_scores(storage_dir=storage_dir,
                             n_eval_runs=n_eval_runs,
                             performance_metric=performance_metric,
                             performance_aggregation=performance_aggregation,
                             group_key="experiment_num",
                             bar_key="storage_name" if logger.level == 10 else "alg_name",
                             re_run_if_exists=re_run_if_exists,
                             save_dir="summary",
                             logger=logger,
                             root_dir=root_dir)

        sorted_inner_keys = _make_benchmark_performance_figure(storage_dirs=[storage_dir],
                                                               logger=logger,
                                                               normalize_with_first_model=False,
                                                               sort_bars=True,
                                                               y_error_bars=y_error_bars,
                                                               save_dir="summary")

        best_experiment_num = sorted_inner_keys[0]
        seed_dirs_for_best_exp = [path for path in (storage_dir / f"experiment{best_experiment_num}").iterdir()]
        copyfile(src=seed_dirs_for_best_exp[0] / "config.json",
                 dst=storage_dir / "summary" / f"bestConfig_exp{best_experiment_num}.json")

    return


def compare_searches(storage_names, x_metric, y_metric, y_error_bars, performance_metric, performance_aggregation,
                     visuals_file, additional_curves_files, re_run_if_exists, logger, root_dir, n_eval_runs):
    """
    compare_searches compare several storage_dirs
    """

    assert type(storage_names) is list

    logger.debug(f'\n{"benchmark_vertical_densities".upper()}:')

    storage_dirs = []
    for storage_name in storage_names:
        storage_dirs.append(get_root(root_dir) / storage_name)

    for storage_dir in storage_dirs:
        if not (storage_dir / "summary" / f"summary_seed_scores.pkl").exists() or re_run_if_exists:
            summarize_search(storage_name=storage_dir.name,
                             n_eval_runs=n_eval_runs,
                             x_metric=x_metric,
                             y_metric=y_metric,
                             y_error_bars=y_error_bars,
                             performance_metric=performance_metric,
                             performance_aggregation=performance_aggregation,
                             re_run_if_exists=re_run_if_exists,
                             make_performance_chart=True,
                             make_learning_plots=True,
                             logger=logger,
                             root_dir=root_dir)

    _make_vertical_densities_figure(storage_dirs=storage_dirs,
                                    visuals_file=visuals_file,
                                    additional_curves_file=additional_curves_file,
                                    make_box_plot=True,
                                    queried_performance_metric=performance_metric,
                                    queried_performance_aggregation=performance_aggregation,
                                    load_dir="summary",
                                    save_dir="benchmark",
                                    logger=logger)

    return


if __name__ == '__main__':
    benchmark_args = get_benchmark_args()
    logger = create_logger(name="BENCHMARK - MAIN", loglevel=benchmark_args.log_level)

    # Gets storage_dirs list

    storage_dirs = select_storage_dirs(from_file=benchmark_args.from_file,
                                       storage_name=benchmark_args.storage_names,
                                       root_dir=benchmark_args.root_dir)

    # Sanity-check that storages exist

    storage_dirs = [storage_dir for storage_dir in storage_dirs if sanity_check_exists(storage_dir, logger)]

    # convert them to storage_name to be compatible with the function called down the line

    benchmark_args.storage_names = [storage_dir_path.name for storage_dir_path in storage_dirs]

    # Gets visuals_file for plotting

    if benchmark_args.from_file is not None:

        # Gets path of visuals_file

        schedule_name = Path(benchmark_args.from_file).parent.stem
        visuals_file = Path(benchmark_args.from_file).parent / f"visuals_{schedule_name}.json"
        additional_curves_file = Path(benchmark_args.from_file).parent / f"additional_curves_{schedule_name}.json"
        if not visuals_file.exists():
            visuals_file = None
        if not additional_curves_file.exists():
            additional_curves_file = None

    else:
        visuals_file = None
        additional_curves_file = None

    # Launches the requested benchmark type (comparing searches [vertical densities] or comparing final models [learning curves])

    if benchmark_args.benchmark_type == "compare_models":
        compare_models(storage_names=benchmark_args.storage_names,
                       x_metric=benchmark_args.x_metric,
                       y_metric=benchmark_args.y_metric,
                       y_error_bars=benchmark_args.y_error_bars,
                       visuals_file=visuals_file,
                       additional_curves_file=additional_curves_file,
                       n_eval_runs=benchmark_args.n_eval_runs,
                       performance_metric=benchmark_args.performance_metric,
                       performance_aggregation=benchmark_args.performance_aggregation,
                       make_performance_chart=False,  # TODO: add support for that chart in a compare_models context
                       make_learning_plots=True,
                       re_run_if_exists=benchmark_args.re_run_if_exists,
                       logger=logger,
                       root_dir=get_root(benchmark_args.root_dir))

    elif benchmark_args.benchmark_type == "compare_searches":
        compare_searches(storage_names=benchmark_args.storage_names,
                         x_metric=benchmark_args.x_metric,
                         y_metric=benchmark_args.y_metric,
                         y_error_bars=benchmark_args.y_error_bars,
                         performance_metric=benchmark_args.performance_metric,
                         performance_aggregation=benchmark_args.performance_aggregation,
                         n_eval_runs=benchmark_args.n_eval_runs,
                         visuals_file=visuals_file,
                         additional_curves_files=additional_curves_file,
                         re_run_if_exists=benchmark_args.re_run_if_exists,
                         logger=logger,
                         root_dir=get_root(benchmark_args.root_dir))

    else:
        raise NotImplementedError
