#!/usr/bin/env python3

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
This script runs timetest executable several times and aggregate
collected statistics.
"""

# pylint: disable=redefined-outer-name

import statistics
import tempfile
import logging
import argparse
import sys
import os
import yaml

from pathlib import Path
from pprint import pprint

TIME_TESTS_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(TIME_TESTS_DIR)

from test_runner.utils import filter_timetest_result

UTILS_DIR = os.path.join(Path(__file__).parent.parent.parent, "utils")
sys.path.insert(0, str(UTILS_DIR))

from proc_utils import cmd_exec
from path_utils import check_positive_int


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
    return {step_name: {"avg": statistics.mean(duration_list),
                        "stdev": statistics.stdev(duration_list) if len(duration_list) > 1 else 0}
            for step_name, duration_list in stats.items()}


def prepare_executable_cmd(args: dict):
    """Generate common part of cmd from arguments to execute"""
    return [
        str(args["executable"].resolve(strict=True)),
        "-m", str(args["model"].resolve(strict=True)),
        "-d", args["device"],
        "-c" if args["model_cache"] else "",
        "-f" if args["config_message"] else "",
        str(args["config_message"].resolve(strict=True)) if args["config_message"] else ""
    ]


def run_timetest(args: dict, log=None):
    """Run provided executable several times and aggregate collected statistics"""
    if log is None:
        log = logging.getLogger("run_timetest")

    cmd_common = prepare_executable_cmd(args)

    # Run executable and collect statistics
    stats = {}
    for run_iter in range(args["niter"]):
        tmp_stats_path = tempfile.NamedTemporaryFile().name
        retcode, msg = cmd_exec(cmd_common + ["-s", str(tmp_stats_path)], log=log)
        if retcode != 0:
            log.error(f"Run of executable '{args['executable']}' failed with return code '{retcode}'. Error: {msg}\n"
                      f"Statistics aggregation is skipped.")
            return retcode, msg, {}, {}

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

    return 0, "", aggregated_stats, stats


def cli_parser():
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run timetest executable")
    parser.add_argument("executable",
                        type=Path,
                        help="Binary to execute")
    parser.add_argument("-m",
                        required=True,
                        dest="model",
                        type=Path,
                        help="Path to an .xml/.onnx file with a trained model or"
                             " to a .blob files with a trained compiled model")
    parser.add_argument("-d",
                        required=True,
                        dest="device",
                        type=str,
                        help="Target device to infer on")
    parser.add_argument("-niter",
                        default=10,
                        type=check_positive_int,
                        help="Number of times to execute binary to aggregate statistics of")
    parser.add_argument("-s",
                        dest="stats_path",
                        type=Path,
                        help="Path to a file to save aggregated statistics")
    parser.add_argument("-c",
                        dest="model_cache",
                        action="store_true",
                        help="Enable model cache usage")
    parser.add_argument("-f",
                        dest="config_message",
                        type=Path,
                        help="Path to configuration file")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = cli_parser()

    logging.basicConfig(format="[ %(levelname)s ] %(message)s",
                        level=logging.DEBUG, stream=sys.stdout)

    exit_code, _, aggr_stats, _ = run_timetest(
        dict(args._get_kwargs()), log=logging)  # pylint: disable=protected-access
    if args.stats_path:
        # Save aggregated results to a file
        with open(args.stats_path, "w") as file:
            yaml.safe_dump(aggr_stats, file)
        logging.info(f"Aggregated statistics saved to a file: '{args.stats_path.resolve()}'")
    else:
        logging.info("Aggregated statistics:")
        pprint(aggr_stats)

    sys.exit(exit_code)


def test_timetest_parser():
    # Example of timetest yml file
    raw_data_example = [{"full_run": [1, {"first_inference_latency": [2, {"load_plugin": [3]}, {
        "create_exenetwork": [4, {"read_network": [5]}, {"load_network": [6]}]}]},
                              {"first_inference": [7, {"fill_inputs": [8]}]}]}]

    # Refactoring raw data from yml
    flatten_dict = {}
    parse_stats(raw_data_example, flatten_dict)

    expected_result = {"full_run": 1, "first_inference_latency": 2, "load_plugin": 3, "create_exenetwork": 4,
                       "read_network": 5, "load_network": 6, "first_inference": 7, "fill_inputs": 8}

    assert flatten_dict == expected_result, "Statistics parsing is performed incorrectly!"
