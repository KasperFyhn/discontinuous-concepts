from scipy.stats import mstats
import numpy


def welch_t_test(dist1: list, dist2: list, exclude_outliers=True):
    if exclude_outliers:
        dist1 = iqr_outlier_filter(dist1)
        dist2 = iqr_outlier_filter(dist2)

    return mstats.ttest_ind(dist1, dist2, equal_var=False)


def iqr_outlier_filter(dist):
    q1, q3 = numpy.quantile(dist, (.25, .75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [v for v in dist if lower_bound < v < upper_bound]


def filtered_dist(dataframe, column, allowed_value):
    return [v for v in dataframe[dataframe[column] == allowed_value][column]]


def prepare_latex_boxplot(dist, show_outliers=False):
    filtered_dist = iqr_outlier_filter(dist)

    if show_outliers:
        outliers = [v for v in dist if v not in filtered_dist]
        outliers = r'\\ '.join(str(round(v, 2)) for v in outliers) + r'\\'
        outliers_str = f'table [row sep=\\\\,y index=0] {{{outliers}}};'
    else:
        outliers_str = 'coordinates {};'
    q0, q1, q2, q3, q4 = numpy.quantile(filtered_dist, (0, .25, .5, .75, 1))
    return (f'\\addplot+ [boxplot prepared={{lower whisker={q0:.2f}, '
            f'lower quartile={q1:.2f}, median={q2:.2f}, '
            f'upper quartile={q3:.2f}, upper whisker={q4:.2f}}}] '
            + outliers_str)


def prepare_comparable_latex_boxplots(y_value, x_value, data):
    pass

