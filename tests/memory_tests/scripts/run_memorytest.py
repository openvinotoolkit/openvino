#!/usr/bin/env python3

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
This script runs test executable several times and aggregate
collected statistics.
"""

# pylint: disable=redefined-outer-name

import argparse
import copy
import logging
import os
import statistics
import sys
import tempfile
import yaml
from pathlib import Path
from pprint import pprint

UTILS_DIR = os.path.join(Path(__file__).parent.parent.parent, "utils")
sys.path.insert(0, str(UTILS_DIR))

from proc_utils import cmd_exec
from path_utils import check_positive_int


def prepare_executable_cmd(args: dict):
    """Generate common part of cmd from arguments to execute"""
    return [str(args["executable"].resolve(strict=True)),
            "-m", str(args["model"].resolve(strict=True)),
            "-d", args["device"]]


def parse_stats(stats: dict, res: dict):
    """Parse statistics to dict"""
    for k, v in stats.items():
        if k not in res.keys():
            res.update({k: {}})
        if isinstance(v, list):
            for element in v:
                for metric, value in element.items():
                    res[k].update({metric: [value]})


def append_stats(stats: dict, parsed_dict: dict):
    if not stats:
        return copy.deepcopy(parsed_dict)
    for step_name, vm_values in parsed_dict.items():
        for vm_metric, vm_value in vm_values.items():
            stats[step_name][vm_metric].extend(vm_value)
    return stats


def aggregate_stats(stats: dict):
    return {step_name: {vm_metric: {"avg": statistics.mean(vm_values_list),
                                    "stdev": statistics.stdev(vm_values_list) if len(vm_values_list) > 1 else 0}
                        for vm_metric, vm_values_list in vm_values.items()}
            for step_name, vm_values in stats.items()}


def run_memorytest(args: dict, log=None):
    """Run provided executable several times and aggregate collected statistics"""

    if log is None:
        log = logging.getLogger('run_test')

    cmd_common = prepare_executable_cmd(args)

    # Run executable and collect statistics
    stats = {}
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

        # Parse raw data
        parsed_data = {}
        parse_stats(raw_data[0], parsed_data)

        log.debug("Statistics after run of executable #{}: {}".format(run_iter, parsed_data))

        stats = append_stats(stats, parsed_data)

    # Aggregate results
    aggregated_stats = aggregate_stats(stats)
    log.debug("Aggregated statistics after full run: {}".format(aggregated_stats))

    return 0, "", aggregated_stats, stats


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

    exit_code, _, aggr_stats, _ = run_memorytest(dict(args._get_kwargs()),
                                                 log=logging)  # pylint: disable=protected-access

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


def test_memorytest_parser():
    # Example of test yml file
    raw_data_example = {'after_objects_release': [{'vmrss': 1}, {'vmhwm': 1},
                                                  {'vmsize': 1}, {'vmpeak': 1}, {'threads': 1}],
                        'create_exenetwork': [{'vmrss': 1}, {'vmhwm': 1},
                                              {'vmsize': 1}, {'vmpeak': 1}, {'threads': 1}],
                        'fill_inputs': [{'vmrss': 1}, {'vmhwm': 1}, {'vmsize': 1},
                                        {'vmpeak': 1}, {'threads': 1}],
                        'first_inference': [{'vmrss': 1}, {'vmhwm': 1},
                                            {'vmsize': 1}, {'vmpeak': 1}, {'threads': 1}],
                        'full_run': [{'vmrss': 1}, {'vmhwm': 1}, {'vmsize': 1},
                                     {'vmpeak': 1}, {'threads': 1}],
                        'load_network': [{'vmrss': 1}, {'vmhwm': 1},
                                         {'vmsize': 1}, {'vmpeak': 1}, {'threads': 1}],
                        'load_plugin': [{'vmrss': 1}, {'vmhwm': 1}, {'vmsize': 1},
                                        {'vmpeak': 1}, {'threads': 1}],
                        'read_network': [{'vmrss': 1}, {'vmhwm': 1}, {'vmsize': 1},
                                         {'vmpeak': 1}, {'threads': 1}]}

    # Refactoring raw data from yml
    parsed_data = {}
    parse_stats(raw_data_example, parsed_data)

    expected_result = {'after_objects_release': {'vmrss': [1], 'vmhwm': [1],
                                                 'vmsize': [1], 'vmpeak': [1], 'threads': [1]},
                       'create_exenetwork': {'vmrss': [1], 'vmhwm': [1],
                                             'vmsize': [1], 'vmpeak': [1], 'threads': [1]},
                       'fill_inputs': {'vmrss': [1], 'vmhwm': [1], 'vmsize': [1],
                                       'vmpeak': [1], 'threads': [1]},
                       'first_inference': {'vmrss': [1], 'vmhwm': [1],
                                           'vmsize': [1], 'vmpeak': [1], 'threads': [1]},
                       'full_run': {'vmrss': [1], 'vmhwm': [1], 'vmsize': [1],
                                    'vmpeak': [1], 'threads': [1]},
                       'load_network': {'vmrss': [1], 'vmhwm': [1],
                                        'vmsize': [1], 'vmpeak': [1], 'threads': [1]},
                       'load_plugin': {'vmrss': [1], 'vmhwm': [1], 'vmsize': [1],
                                       'vmpeak': [1], 'threads': [1]},
                       'read_network': {'vmrss': [1], 'vmhwm': [1],
                                        'vmsize': [1], 'vmpeak': [1], 'threads': [1]}}

    assert parsed_data == expected_result, "Statistics parsing is performed incorrectly!"
