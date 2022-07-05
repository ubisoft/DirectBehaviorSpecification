import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import numpy as np

def get_95_confidence_interval(samples, method):
    if method == "stderr":
        mean = samples.mean(-1)
        number_of_samples = len(samples)
        samples_std = samples.std(-1)
        samples_stderr = 1.96 * samples_std / (number_of_samples ** 0.5)
        err_up = samples_stderr
        err_down = samples_stderr

    elif method == "bootstrapped_CI":
        bootstrapped_result = bs.bootstrap(samples, stat_func=bs_stats.mean)
        mean = bootstrapped_result.value
        err_up = bootstrapped_result.upper_bound - bootstrapped_result.value
        err_down = bootstrapped_result.value - bootstrapped_result.lower_bound

    else:
        raise NotImplementedError(method)

    return mean, err_up, err_down


def get_95_confidence_interval_of_sequence(list_of_samples, method):
    # list_of_samples must be of shape (n_time_steps, n_samples)
    means = []
    err_ups = []
    err_downs = []
    for samples in list_of_samples:
        mean, err_up, err_down = get_95_confidence_interval(samples=samples, method=method)
        means.append(mean)
        err_ups.append(err_up)
        err_downs.append(err_down)
    return np.asarray(means), np.asarray(err_ups), np.asarray(err_downs)