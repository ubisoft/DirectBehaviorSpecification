from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import seaborn as sns

sns.set()
sns.set_style('whitegrid')


def bar_chart(ax, scores, err_up=None, err_down=None, capsize=10., colors=None,
              group_names=None, xlabel="", ylabel="", title="", cmap="viridis"):
    # data to plot
    n_groups = len(list(scores.values())[0])

    # chooses colors
    if colors is None:
        cm = plt.cm.get_cmap(cmap)
        colors = {alg_name: np.array(cm(float(i) / float(n_groups))[:3]) for i, alg_name in enumerate(scores.keys())}

    if err_up is None:
        err_up = {alg_name: None for alg_name in scores.keys()}

    if err_down is None:
        err_down = {alg_name: None for alg_name in scores.keys()}

    # create plot
    bar_width = (1. / n_groups) * 2. * len(scores.keys())
    index = np.arange(n_groups) * (float(len(scores.keys())) + 1) * bar_width

    for i, alg_name in enumerate(scores.keys()):
        ax.bar(index + i * bar_width, scores[alg_name].values(), bar_width,
               yerr=[err_down[alg_name].values(), err_up[alg_name].values()],
               ecolor="cyan", capsize=capsize, color=colors[alg_name], label=alg_name)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if title is not None:
        plt.title(title, fontsize=12, fontweight='bold')

    if group_names is not None:
        plt.xticks(index, group_names)

    ax.legend(loc='upper right')


def plot_curves(ax, ys, xs=None, colors=None, markers=None, markersize=15, markevery=None, labels=None,
                xlabel="", ylabel="", xlim=(None, None), ylim=(None, None), axis_font_size=22, tick_font_size=18,
                n_x_ticks=4, n_y_ticks=4, title="", title_font_size=24, fill_up=None, fill_down=None, alpha_fill=0.1,
                error_up=None, error_down=None, smooth=False, add_legend=True, legend_outside=False,
                legend_font_size=20, legend_pos=(0.5, -0.2), legend_loc="upper center", legend_n_columns=1,
                legend_marker_first=True, hlines=None):
    if xs is None:
        xs = [range(len(y)) for y in ys]

    if colors is None:
        colors = [None] * len(ys)

    if markers is None:
        markers = [None] * len(ys)

    if labels is None:
        labels = [None] * len(ys)

    # Plots losses and smoothed losses for every agent
    n = len(xs)
    for i, (x, y) in enumerate(zip(xs, ys)):

        if markevery is None:
            markevery = len(y) // 10

        # Adds filling around curve (central tendency)

        if fill_up is not None and fill_down is not None:
            ax.plot(x, y, color=colors[i], marker=markers[i], markevery=markevery, markersize=markersize,
                    label=labels[i], zorder=n-i)
            ax.fill_between(x, y - fill_down[i], y + fill_up[i], color=colors[i], alpha=alpha_fill, zorder=n-i)

        # OR: Adds error bars above and below each datapoint

        elif error_up is not None and error_down is not None:
            ax.errorbar(x, y, color=colors[i], marker=markers[i], markevery=markevery, markersize=markersize,
                    label=labels[i], zorder=n - i, yerr=[error_down[i], error_up[i]])

        # OR: Smooth curve using running average

        elif smooth:
            ax.plot(x, y, color=colors[i], alpha=3 * alpha_fill)
            ax.plot(x, smooth_out(y), color=colors[i], marker=markers[i], markevery=markevery, markersize=markersize,
                    label=labels[i], zorder=n-i)

        # Just regular curve

        else:
            ax.plot(x, y, color=colors[i], marker=markers[i], markevery=markevery, markersize=markersize,
                    label=labels[i], zorder=n-i)

    # Axis settings

    ax.set_title(title, fontsize=title_font_size)
    ax.set_xlabel(xlabel, fontsize=axis_font_size)
    ax.set_ylabel(ylabel, fontsize=axis_font_size)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.xaxis.set_major_locator(plt.MaxNLocator(n_x_ticks))
    ax.yaxis.set_major_locator(plt.MaxNLocator(n_y_ticks))
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)

    # Plots horizontal lines

    if hlines is not None:
        xmin, xmax = ax.get_xlim()

        for hline in hlines:
            hline.update({'xmin': xmin, 'xmax': xmax})
            ax.hlines(**hline)

    # Legend settings

    if not all(label is None for label in labels) and add_legend:

        if legend_outside:
            if add_legend:
                legend = ax.legend(loc=legend_loc, framealpha=0.25, bbox_to_anchor=legend_pos, markerfirst=legend_marker_first,
                                   fancybox=True, shadow=False, ncol=legend_n_columns, fontsize=legend_font_size)
                for legobj in legend.legendHandles:
                    legobj.set_linewidth(2.0)
                for text in legend.get_texts():
                    text.set_ha('left')

        else:
            ax.legend(loc=legend_loc, framealpha=0.25, fancybox=True, shadow=False)


def plot_sampled_hyperparams(ax, param_samples, log_params):
    cm = plt.cm.get_cmap('viridis')
    for i, param in enumerate(param_samples.keys()):
        args = param_samples[param], np.zeros_like(param_samples[param])
        kwargs = {'linestyle': '', 'marker': 'o', 'label': param, 'alpha': 0.2,
                  'color': cm(float(i) / float(len(param_samples)))}
        if param in log_params:
            ax[i].semilogx(*args, **kwargs)
        else:
            ax[i].plot(*args, **kwargs)
            ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

        ax[i].get_yaxis().set_ticks([])
        ax[i].legend(loc='upper right')


def plot_vertical_densities(ax, ys, colors=None, labels=None,
                            xlabel="", ylabel="", axis_font_size=22, tick_font_size=18, title="", title_font_size=24,
                            make_boxplot=False, whis=(0, 99), hlines=None):
    if colors is None:
        colors = [None] * len(ys)

    if labels is None:
        labels = [None] * len(ys)

    # Plots vertical densities (either every point or summarised in box-plots)

    if make_boxplot:
        bp1 = ax.boxplot(ys,
                         labels=labels,
                         sym='o',
                         whis=whis,
                         positions=np.arange(1, len(ys) + 1),
                         widths=0.50,
                         patch_artist=True)

        for box, fli, med, col in zip(bp1['boxes'], bp1['fliers'], bp1['medians'], colors):
            box.set(facecolor=col)
            fli.set(markerfacecolor=col)
            med.set(color='black', linewidth=2)

    else:
        for i, y in enumerate(ys):
            pos = [i] * len(ys)
            ax.plot(np.array(pos) + 1, ys, linestyle='', marker='o', label=labels[i], alpha=0.5, color=colors[i])

    # Adds horizontal lines

    if hlines is not None:
        xmin, xmax = 0.5, len(ys) + 0.5
        for hline in hlines:
            hline.update({'xmin': xmin - 1, 'xmax': xmax + 1})
            ax.hlines(**hline)

    # Axis settings

    ax.set_xlim(0.5, len(ys) + 0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(len(ys)))
    ax.set_xticklabels([""] + labels, rotation=45, ha="right")

    ax.set_title(title, fontsize=title_font_size)
    ax.set_xlabel(xlabel, fontsize=axis_font_size)
    ax.set_ylabel(ylabel, fontsize=axis_font_size)

    ## Commented below because looks hugly

    # if hlines is not None:
    #     legend = ax.legend(framealpha=0.25, fancybox=True, shadow=False, fontsize=20, bbox_to_anchor=(0.5, -0.2),
    #                        loc="upper center")
    #     for legobj in legend.legendHandles:
    #         legobj.set_linewidth(2.0)

    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)


def smooth_out(data_serie, smooth_factor=0.2):
    assert smooth_factor > 0. and smooth_factor < 1.
    mean = None

    new_serie = []
    for value in data_serie:
        if value is None:
            new_serie.append(None)
        else:
            if mean is None:
                mean = value
            else:
                mean = smooth_factor * mean + (1 - smooth_factor) * value

            new_serie.append(mean)

    return new_serie


def create_fig(axes_shape, figsize=None):
    figsize = (8 * axes_shape[1], 5 * axes_shape[0]) if figsize is None else figsize
    fig, axes = plt.subplots(axes_shape[0], axes_shape[1], figsize=figsize)
    return fig, axes
