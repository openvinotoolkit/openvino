# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility module."""

import numpy as np

# Define a range to cut outliers which are < Q1 − IQR_CUTOFF * IQR, and > Q3 + IQR_CUTOFF * IQR
# https://en.wikipedia.org/wiki/Interquartile_range
IQR_CUTOFF = 1.5


def calculate_iqr(stats: list):
    """IQR is calculated as the difference between the 3th and the 1th quantile of the data."""
    q1 = np.quantile(stats, 0.25)
    q3 = np.quantile(stats, 0.75)
    iqr = q3 - q1
    return iqr, q1, q3


def filter_timetest_result(stats: dict):
    """Identify and remove outliers from time_results."""
    filtered_stats = {}
    for step_name, time_results in stats.items():
        iqr, q1, q3 = calculate_iqr(time_results)
        cut_off = iqr * IQR_CUTOFF
        upd_time_results = [x for x in time_results if (q1 - cut_off < x < q3 + cut_off)]
        filtered_stats.update({step_name: upd_time_results if upd_time_results else time_results})
    return filtered_stats
