from scipy.stats import mstats
import numpy
import pingouin as pg
import os


SCRIPTS_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(SCRIPTS_DIR)


def iqr_outlier_filter(dist):
    q1, q3 = numpy.quantile(dist, (.25, .75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [v for v in dist if lower_bound < v < upper_bound]


def filtered_dist(df, column, filter_column, allowed_value):
    return [v for v in df[df[filter_column] == allowed_value][column]]


def prepare_latex_boxplot(dist, show_outliers=False):
    dist = iqr_outlier_filter(dist)

    if show_outliers:
        outliers = [v for v in dist if v not in dist]
        outliers = r'\\ '.join(str(round(v, 2)) for v in outliers) + r'\\'
        outliers_str = f'table [row sep=\\\\,y index=0] {{{outliers}}};'
    else:
        outliers_str = 'coordinates {};'
    q0, q1, q2, q3, q4 = numpy.quantile(dist, (0, .25, .5, .75, 1))
    return (f'\\addplot+ [boxplot prepared={{lower whisker={q0:.2f}, '
            f'lower quartile={q1:.2f}, median={q2:.2f}, '
            f'upper quartile={q3:.2f}, upper whisker={q4:.2f}}}] '
            + outliers_str)


def prepare_comparable_latex_boxplots(x_value, y_value, data,
                                      show_outliers=False):

    tick_labels = list(set(data[x_value]))  # unique labels ordered

    top_string_begin = r"\begin{axis}[boxplot/draw direction=y, "
    ticks = ', '.join(str(i) for i in range(1, len(tick_labels) + 1))
    tick_string = f"xtick={{{ticks}}}, "
    tick_labels_str = [str(v) for v in tick_labels]
    tick_label_string = f"xticklabels={{{', '.join(tick_labels_str)}}}]"
    top_string = top_string_begin + tick_string + tick_label_string + '\n'

    plots_string = ''
    for label in tick_labels:
        dist = filtered_dist(data, y_value, x_value, label)
        plots_string += prepare_latex_boxplot(dist, show_outliers) + '\n'

    end_string = "\end{axis}"

    return top_string + plots_string + end_string

