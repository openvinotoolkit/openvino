#!/usr/bin/env python3
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
This script runs timetest executable several times and aggregate
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


def aggregate_stats(stats: dict):
    """Aggregate provided statistics"""
    return {step_name: {"avg": statistics.mean(duration_list),
                        "stdev": statistics.stdev(duration_list) if len(duration_list) > 1 else 0}
            for step_name, duration_list in stats.items()}


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
        with tempfile.NamedTemporaryFile() as tf:
            retcode, msg = run_cmd(cmd_common + ["-s", str(tf.name)], log=log)
            if retcode != 0:
                log.error("Run of executable '{}' failed with return code '{}'. Error: {}\n"
                          "Statistics aggregation is skipped.".format(args["executable"], retcode, msg))
                return retcode, {}

            # Read raw statistics
            raw_data = yaml.safe_load(tf)
            log.debug("Raw statistics after run of executable #{}: {}".format(run_iter, raw_data))

        # Combine statistics from several runs
        stats = dict((step_name, stats.get(step_name, []) + [duration])
                     for step_name, duration in raw_data.items())

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
