import logging as log
import os
import statistics
import tempfile

import numpy as np
import yaml

from .common_utils import get_executable_cmd
from ...common_utils.test_utils import shell

# Define a range to cut outliers which are < Q1 ? IQR_CUTOFF * IQR, and > Q3 + IQR_CUTOFF * IQR
# https://en.wikipedia.org/wiki/Interquartile_range
IQR_CUTOFF = 1.5


def parse_stats(stats: list, res: dict):
    """Parse raw statistics from nested list to flatten dict"""
    for element in stats:
        if isinstance(element, (int, float)):
            for k, v in res.items():
                if v is None:
                    res.update({k: element})
        else:
            for k, v in element.items():
                if len(v) == 1:
                    res.update({k: v[0]})
                else:
                    res.update({k: None})
                    parse_stats(v, res)


def aggregate_stats(stats: dict):
    """Aggregate provided statistics"""
    return {step_name: {"avg": statistics.mean(duration_list)}
            for step_name, duration_list in stats.items()}


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


def run_timetest(args: dict):
    """Run provided executable several times and aggregate collected statistics"""
    cmd_common = get_executable_cmd(args)
    # Run executable and collect statistics
    stats = {}
    for run_iter in range(args["niter"]):
        tmp_stats_path = tempfile.NamedTemporaryFile().name
        retcode, _, stderr = shell(cmd_common + ["-s", str(tmp_stats_path)], log=True)
        if retcode != 0:
            log.error(f"Run of executable '{args['executable']}' failed with return code '{retcode}'. "
                      f"Error: {stderr}\n"
                      f"Statistics aggregation is skipped.")
            return retcode, stderr, {}, {}

        # Read raw statistics
        with open(tmp_stats_path, "r") as file:
            raw_data = list(yaml.load_all(file, Loader=yaml.SafeLoader))

        os.unlink(tmp_stats_path)

        # Parse raw data
        flatten_data = {}
        parse_stats(raw_data[0], flatten_data)

        log.debug(f"Statistics after run of executable #{run_iter}: {flatten_data}")

        # Combine statistics from several runs
        stats = dict((step_name, stats.get(step_name, []) + [duration])
                     for step_name, duration in flatten_data.items())

    # Remove outliers
    filtered_stats = filter_timetest_result(stats)

    # Aggregate results
    aggregated_stats = aggregate_stats(filtered_stats)
    log.debug(f"Aggregated statistics after full run: {aggregated_stats}")

    return aggregated_stats


def get_compared_time_results(static_res, dynamic_res):
    """Returns the difference between static and dynamic results as percentage"""
    if "reshape" in dynamic_res:
        del dynamic_res["reshape"]
    compared_res = {}
    for static_step_name, static_avg in static_res.items():
        compared_res.update(
            {static_step_name: '{:.3f}'.format(dynamic_res[static_step_name]['avg'] / static_avg['avg'] * 100 - 100)})
    return compared_res


def get_compared_with_refs_results(ref_res, cur_res):
    """Returns the difference between reference and current test results"""
    ref_compared_res = {}
    if isinstance(ref_res, str):
        return ref_compared_res
    for metric_ref_name, metric_ref_value in ref_res.items():
        ref_compared_res.update({metric_ref_name: {
            "ref - cur": float(metric_ref_value) - float(cur_res[metric_ref_name]),
            "ref": float(metric_ref_value),
            "cur": float(cur_res[metric_ref_name])
        }})
    return ref_compared_res
