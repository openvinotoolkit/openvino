# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import scipy.stats


def calc_clipped_mean(mean, sigma, a, b):
    normal = scipy.stats.norm()
    alpha = (a - mean) / sigma
    beta = (b - mean) / sigma
    result = sigma * (normal.pdf(alpha) - normal.pdf(beta)) + \
        mean * (normal.cdf(beta) - normal.cdf(alpha))

    if a < np.inf:
        result += a * normal.cdf(alpha)
    if b < np.inf:
        result += b * (1 - normal.cdf(beta))

    return result


def calc_clipped_sigma(mean, sigma, a, b):
    normal = scipy.stats.norm()
    alpha = (a - mean) / sigma
    beta = (b - mean) / sigma
    Z = normal.cdf(beta) - normal.cdf(alpha)
    cmean = calc_clipped_mean(mean, sigma, a, b)
    result = Z * ((mean ** 2) + (sigma ** 2) + (cmean ** 2) - 2 * cmean * mean) + \
        sigma * (mean - 2 * cmean) * (normal.pdf(alpha) - normal.pdf(beta))

    if a < np.inf:
        result += sigma * a * normal.pdf(alpha) + ((a - cmean) ** 2) * normal.cdf(alpha)
    if b < np.inf:
        result += -sigma * b * normal.pdf(beta) + ((b - cmean) ** 2) * (1 - normal.cdf(beta))

    if ((result < 0) & (abs(result) > 1e-10)).any():
        raise RuntimeError('Negative variance')

    result[result < 0] = 0
    result = np.sqrt(result)
    return result
