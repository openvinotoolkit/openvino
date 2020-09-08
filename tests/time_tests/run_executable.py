#!/usr/bin/env python3
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
This script runs TimeTests executable several times to aggregate
collected statistics.
"""

import statistics as Statistics
from pathlib import Path
import tempfile
import subprocess
import logging
import argparse
import sys
import yaml


def run_cmd(args: list, log=None, verbose=True):
    """ Run command
    """
    if log is None:
        log = logging.getLogger('run_cmd')
    log_out = log.info if verbose else log.debug

    log.info(f'========== cmd: {" ".join(args)}')  # pylint: disable=logging-format-interpolation

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


def read_statistics(statistics_path, statistics: dir):
    """Read statistics from a file and extend provided statistics"""
    with open(statistics_path, "r") as file:
        parsed_data = yaml.load(file, Loader=yaml.FullLoader)
    return dict((step_name, statistics.get(step_name, []) + [duration])
                for step_name, duration in parsed_data.items())


def aggregate_statistics(statistics: dir):
    """Aggregate provided statistics"""
    return {step_name: {"avg": Statistics.mean(duration_list),
                        "stdev": Statistics.stdev(duration_list)}
            for step_name, duration_list in statistics.items()}


def write_aggregated_stats(statistics_path, statistics: dir):
    """Write aggregated statistics to a file in YAML format"""
    with open(statistics_path, "w") as file:
        yaml.dump(statistics, file)


def run_executable(executable: Path, model: Path, device, niter, log=None):
    """Run provided executable several times and aggregate collected statistics"""

    if log is None:
        log = logging.getLogger('run_executable')

    cmd_common = [str(executable.resolve()),
                  "-m", str(model.resolve()),
                  "-d", device]

    # Create folder to save statistics files
    statistics_dir = (Path(".") / "statistics_dir").absolute()
    statistics_dir.mkdir(parents=True, exist_ok=True)

    # Run executable and collect statistics
    statistics = {}
    for run_iter in range(niter):
        statistics_path = statistics_dir / Path(tempfile.NamedTemporaryFile().name).stem
        log.info("Statistics file path of #{} iteration: {}".format(run_iter, statistics_path))
        retcode, msg = run_cmd(cmd_common + ["-s", str(statistics_path)], log=log)
        if retcode != 0:
            log.error("Run of executable '{}' failed with return code '{}'. Error: {}\n"
                      "Statistics aggregation is skipped.".format(executable, retcode, msg))
            return retcode, {}

        statistics = read_statistics(statistics_path, statistics)

    # Aggregate results
    aggregated_stats = aggregate_statistics(statistics)

    # Save aggregated results to a file
    aggr_stats_path = statistics_dir / "aggregated_stats_{}.yml".format(
        Path(tempfile.NamedTemporaryFile().name).stem)
    write_aggregated_stats(aggr_stats_path, aggregated_stats)
    log.info("Aggregated statistics saved to a file: '{}'".format(aggr_stats_path))

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

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = cli_parser()
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.DEBUG, stream=sys.stdout)
    exit_code, _ = run_executable(args.executable, args.model, args.device, args.niter, log=logging)
    sys.exit(exit_code)
