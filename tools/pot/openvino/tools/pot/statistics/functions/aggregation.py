# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ..function_selector import AGGREGATION_FN as aggregator

@aggregator.register()
def batch_mean(x):
    return np.mean(x, axis=0, keepdims=True)


@aggregator.register()
def mean(x):
    return np.mean([np.mean(val, axis=0) for val in x], axis=0)


@aggregator.register('max')
def amax(x):
    return np.max([np.max(val, axis=0) for val in x], axis=0)


@aggregator.register('min')
def amin(x):
    return np.min([np.min(val, axis=0) for val in x], axis=0)


@aggregator.register()
def median(x):
    return np.median(x, axis=(0, 1))


@aggregator.register()
def mean_no_outliers(x):
    return no_outliers_estimator(np.mean, x)


@aggregator.register()
def median_no_outliers(x):
    return no_outliers_estimator(np.median, x)


@aggregator.register('hl_estimator')
def hodges_lehmann_mean(x):
    """ Outlier-robust mean estimator
    """

    def hl_functor(x):
        m = np.add.outer(x, x)
        ind = np.tril_indices(len(x), -1)
        return 0.5 * np.median(m[ind])

    x = np.array(x)
    if len(x.shape) == 3:
        result = np.zeros(x.shape[2], x.dtype)
        for i in range(x.shape[2]):
            x_ch = x[:, :, i].flatten()
            result[i] = hl_functor(x_ch)
    else:
        result = hl_functor(x.flatten())
    return result


def no_outliers_estimator(base_estimator, x, alpha=0.01):
    """ Calculate base_estimator function after removal of extreme quantiles
    from the sample
    """
    x = np.array(x)
    if len(x.shape) < 3:
        x = np.expand_dims(x, -1)
    low_value = np.quantile(x, alpha, axis=(0, 1))
    high_value = np.quantile(x, 1 - alpha, axis=(0, 1))

    result = np.zeros(x.shape[2], x.dtype)
    for i in range(x.shape[2]):
        x_ch = x[:, :, i]
        x_ch = x_ch[(x_ch >= low_value[i]) & (x_ch <= high_value[i])]
        result[i] = base_estimator(x_ch)
    return result
