#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
This script runs test executable several times and aggregate
collected statistics.
"""

# pylint: disable=redefined-outer-name

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from pprint import pprint

import yaml

UTILS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(UTILS_DIR))

from proc_utils import cmd_exec
from scripts.stats_parsers import StatisticsParser


def prepare_executable_cmd(args: dict):
    """Generate common part of cmd from arguments to execute"""
    return [str(args["executable"].resolve(strict=True)),
            "-m", str(args["model"].resolve(strict=True)),
            "-d", args["device"]]


def run_test(args: dict, log=None):
    """Run provided executable several times and aggregate collected statistics"""

    if log is None:
        log = logging.getLogger('run_test')

    cmd_common = prepare_executable_cmd(args)

    # Run executable and collect statistics
    stats_parser = StatisticsParser()
    for run_iter in range(args["niter"]):
        tmp_stats_path = tempfile.NamedTemporaryFile().name
        retcode, msg = cmd_exec(cmd_common + ["-s", str(tmp_stats_path)], log=log)
        if retcode != 0:
            log.error("Run of executable '{}' failed with return code '{}'. Error: {}\n"
                      "Statistics aggregation is skipped.".format(args["executable"], retcode, msg))
            return retcode, msg, {}, {}

        # Read raw statistics
        with open(tmp_stats_path, "r") as file:
            raw_data = list(yaml.load_all(file, Loader=yaml.SafeLoader))

        os.unlink(tmp_stats_path)

        # Parse and combine raw data
        stats_parser.append_stats(raw_data)

        log.debug("Statistics after run of executable #{}: {}".format(run_iter, stats_parser.executor.last_stats))

    # Aggregate results
    stats_parser.aggregate_stats()
    log.debug("Aggregated statistics after full run: {}".format(stats_parser.executor.aggregated_stats))

    return 0, "", stats_parser.executor.aggregated_stats, stats_parser.executor.combined_stats


def check_positive_int(val):
    """Check argsparse argument is positive integer and return it"""
    value = int(val)
    if value < 1:
        msg = "%r is less than 1" % val
        raise argparse.ArgumentTypeError(msg)
    return value


def cli_parser():
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Run test executable')
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
                        default=10,
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

    exit_code, _, aggr_stats, _ = run_test(dict(args._get_kwargs()), log=logging)  # pylint: disable=protected-access

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


def test_parser():
    # Example of test yml file
    raw_data_example = [{'full_run': [1, {'first_inference_latency': [2, {'load_plugin': [3]}, {
        'create_exenetwork': [4, {'read_network': [5]}, {'load_network': [6]}]}]},
                                      {'first_inference': [7, {'fill_inputs': [8]}]}]}]

    # Refactoring raw data from yml
    StatisticsParser().parse_stats(raw_data_example)

    expected_result = {'full_run': [1], 'first_inference_latency': [2], 'load_plugin': [3], 'create_exenetwork': [4],
                       'read_network': [5], 'load_network': [6], 'first_inference': [7], 'fill_inputs': [8]}

    assert StatisticsParser().executor.last_stats == expected_result, "Statistics parsing is performed incorrectly!"
