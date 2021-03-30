#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
This script runs timetest executable several times and aggregate
collected statistics.
"""

# pylint: disable=redefined-outer-name

import statistics
import tempfile
import subprocess
import logging
import argparse
import sys
import os
import yaml

from pathlib import Path
from pprint import pprint
import numpy as np

# Define a range to cut outliers which are < Q1 − IRQ_CUTOFF * IQR, and > Q3 + IRQ_CUTOFF * IQR
# https://en.wikipedia.org/wiki/Interquartile_range
IRQ_CUTOFF = 1.5


def run_cmd(args: list, log=None, verbose=True):
    """ Run command
    """
    if log is None:
        log = logging.getLogger('run_cmd')
    log_out = log.info if verbose else log.debug

    log.info(f'========== cmd: {" ".join(args)}')  # pylint: disable=logging-fstring-interpolation

    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            encoding='utf-8',
                            universal_newlines=True)
    output = []
    for line in iter(proc.stdout.readline, ''):
        log_out(line.strip('\n'))
        output.append(line)
        if line or proc.poll() is None:
            continue
        break
    outs = proc.communicate()[0]

    if outs:
        log_out(outs.strip('\n'))
        output.append(outs)
    log.info('========== Completed. Exit code: %d', proc.returncode)
    return proc.returncode, ''.join(output)


def aggregate_stats(stats: dict):
    """Aggregate provided statistics"""
    return {step_name: {"avg": statistics.mean(duration_list),
                        "stdev": statistics.stdev(duration_list) if len(duration_list) > 1 else 0}
            for step_name, duration_list in stats.items()}


def calculate_iqr(stats: list):
    """IQR is calculated as the difference between the 3th and the 1th quantile of the data"""
    q1 = np.quantile(stats, 0.25)
    q3 = np.quantile(stats, 0.75)
    iqr = q3 - q1
    return iqr, q1, q3


def prepare_executable_cmd(args: dict):
    """Generate common part of cmd from arguments to execute"""
    return [str(args["executable"].resolve(strict=True)),
            "-m", str(args["model"].resolve(strict=True)),
            "-d", args["device"]]


def run_timetest(args: dict, log=None):
    """Run provided executable several times and aggregate collected statistics"""

    if log is None:
        log = logging.getLogger('run_timetest')

    cmd_common = prepare_executable_cmd(args)

    # Run executable and collect statistics
    stats = {}
    for run_iter in range(args["niter"]):
        tmp_stats_path = tempfile.NamedTemporaryFile().name
        retcode, msg = run_cmd(cmd_common + ["-s", str(tmp_stats_path)], log=log)
        if retcode != 0:
            log.error("Run of executable '{}' failed with return code '{}'. Error: {}\n"
                      "Statistics aggregation is skipped.".format(args["executable"], retcode, msg))
            return retcode, {}

        # Read raw statistics
        with open(tmp_stats_path, "r") as file:
            raw_data = yaml.safe_load(file)

        os.unlink(tmp_stats_path)
        log.debug("Raw statistics after run of executable #{}: {}".format(run_iter, raw_data))

        # Combine statistics from several runs
        stats = dict((step_name, stats.get(step_name, []) + [duration])
                     for step_name, duration in raw_data.items())

    # Remove outliers
    for step_name, time_results in stats.items():
        iqr, q1, q3 = calculate_iqr(time_results)
        cut_off = iqr * IRQ_CUTOFF
        upd_time_results = [x for x in time_results if x > q1 - cut_off or x < q3 + cut_off]
        stats.update({step_name: upd_time_results})

    # Aggregate results
    aggregated_stats = aggregate_stats(stats)
    log.debug("Aggregated statistics after full run: {}".format(aggregated_stats))

    return 0, aggregated_stats


def check_positive_int(val):
    """Check argsparse argument is positive integer and return it"""
    value = int(val)
    if value < 1:
        msg = "%r is less than 1" % val
        raise argparse.ArgumentTypeError(msg)
    return value


def cli_parser():
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Run timetest executable')
    parser.add_argument('executable',
                        type=Path,
                        help='binary to execute')
    parser.add_argument('-m',
                        required=True,
                        dest="model",
                        type=Path,
                        help='path to an .xml/.onnx/.prototxt file with a trained model or'
                             ' to a .blob files with a trained compiled model')
    parser.add_argument('-d',
                        required=True,
                        dest="device",
                        type=str,
                        help='target device to infer on')
    parser.add_argument('-niter',
                        default=3,
                        type=check_positive_int,
                        help='number of times to execute binary to aggregate statistics of')
    parser.add_argument('-s',
                        dest="stats_path",
                        type=Path,
                        help='path to a file to save aggregated statistics')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = cli_parser()

    logging.basicConfig(format="[ %(levelname)s ] %(message)s",
                        level=logging.DEBUG, stream=sys.stdout)

    exit_code, aggr_stats = run_timetest(dict(args._get_kwargs()), log=logging)  # pylint: disable=protected-access

    if args.stats_path:
        # Save aggregated results to a file
        with open(args.stats_path, "w") as file:
            yaml.safe_dump(aggr_stats, file)
        logging.info("Aggregated statistics saved to a file: '{}'".format(
            args.stats_path.resolve()))
    else:
        logging.info("Aggregated statistics:")
        pprint(aggr_stats)

    sys.exit(exit_code)
