#!/usr/bin/env python3
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
This script runs TimeTests executable several times to aggregate
collected statistics.
"""

# pylint: disable=redefined-outer-name

import statistics
from pathlib import Path
import tempfile
import subprocess
import logging
import argparse
import sys
from pprint import pprint
import yaml


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


def read_stats(stats_path, stats: dict):
    """Read statistics from a file and extend provided statistics"""
    with open(stats_path, "r") as file:
        parsed_data = yaml.load(file, Loader=yaml.FullLoader)
    return dict((step_name, stats.get(step_name, []) + [duration])
                for step_name, duration in parsed_data.items())


def aggregate_stats(stats: dict):
    """Aggregate provided statistics"""
    return {step_name: {"avg": statistics.mean(duration_list),
                        "stdev": statistics.stdev(duration_list)}
            for step_name, duration_list in stats.items()}


def write_aggregated_stats(stats_path, stats: dict):
    """Write aggregated statistics to a file in YAML format"""
    with open(stats_path, "w") as file:
        yaml.dump(stats, file)


def prepare_executable_cmd(args: dict):
    """Generate common part of cmd from arguments to execute"""
    return [str(args["executable"].resolve()),
            "-m", str(args["model"].resolve()),
            "-d", args["device"]]


def generate_tmp_path():
    """Generate temporary file path without file's creation"""
    tmp_stats_file = tempfile.NamedTemporaryFile()
    path = tmp_stats_file.name
    tmp_stats_file.close()  # remove temp file in order to create it by executable
    return path


def run_executable(args: dict, log=None):
    """Run provided executable several times and aggregate collected statistics"""

    if log is None:
        log = logging.getLogger('run_executable')

    cmd_common = prepare_executable_cmd(args)

    # Run executable and collect statistics
    stats = {}
    for run_iter in range(args["niter"]):
        tmp_stats_path = generate_tmp_path()
        retcode, msg = run_cmd(cmd_common + ["-s", str(tmp_stats_path)], log=log)
        if retcode != 0:
            log.error("Run of executable '{}' failed with return code '{}'. Error: {}\n"
                      "Statistics aggregation is skipped.".format(args["executable"], retcode, msg))
            return retcode, {}

        stats = read_stats(tmp_stats_path, stats)

    # Aggregate results
    aggregated_stats = aggregate_stats(stats)

    return 0, aggregated_stats


def cli_parser():
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Run TimeTests executable')
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
                        type=int,
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

    exit_code, aggr_stats = run_executable(dict(args._get_kwargs()), log=logging)  # pylint: disable=protected-access

    if args.stats_path:
        # Save aggregated results to a file
        write_aggregated_stats(args.stats_path, aggr_stats)
        logging.info("Aggregated statistics saved to a file: '{}'".format(
            args.stats_path.resolve()))
    else:
        logging.info("Aggregated statistics:")
        pprint(aggr_stats)

    sys.exit(exit_code)
