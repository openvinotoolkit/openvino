#!/usr/bin/env python3
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
This script runs memcheck tests isolated with help of gtest_parallel. It can
upload memory measurment results to database and generate reports.
"""

import argparse
from glob import glob
import json
import logging
import os
import subprocess
import sys

from memcheck_upload import create_memcheck_records, upload_memcheck_records, create_memcheck_report

DATABASE = 'memcheck'
COLLECTIONS = ["commit", "nightly", "weekly"]


def run(args, log=None, verbose=True):
    """ Run command
    """
    if log is None:
        log = logging.getLogger('run_memcheck')
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


def json_load(path_or_string):
    """ Load json as file or as string
    """
    if os.path.isfile(path_or_string):
        with open(path_or_string, 'r') as json_fp:
            return json.load(json_fp)
    else:
        return json.loads(path_or_string)


def main():
    """Main entry point.
    """

    # remove additional args (arguments after --)
    binary_args = []
    for idx, arg in enumerate(sys.argv):
        if arg == '--':
            binary_args = sys.argv[idx+1:].split()
            sys.argv = sys.argv[:idx]
            break

    init_parser = argparse.ArgumentParser(add_help=False)
    init_parser.add_argument('--timeline_report',
                             help=f'Create timeline HTML report file name.')
    init_parser.add_argument('--upload', action="store_true",
                             help=f'Upload results to database.')
    args = init_parser.parse_known_args()[0]

    parser = argparse.ArgumentParser(
        description='Run memcheck tests',
        usage='%(prog)s [options] binary -- [additional args]',
        parents=[init_parser])
    parser.add_argument('binary', help='test binary to execute')
    parser.add_argument('--gtest_parallel', help='Path to gtest-parallel to use.',
                        default='gtest_parallel')
    parser.add_argument('-d', '--output_dir',
                        required=args.timeline_report or args.upload,
                        help='output directory for test logs')
    parser.add_argument('-w', '--workers', help='number of gtest-parallel workers to spawn')

    parser.add_argument('--db_url',
                        required=args.timeline_report or args.upload,
                        help='MongoDB URL in a form "mongodb://server:port"')
    parser.add_argument('--db_collection',
                        required=args.timeline_report or args.upload,
                        help=f'use collection name in {DATABASE} database',
                        choices=COLLECTIONS)
    parser.add_argument('--metadata',
                        default='{}',
                        help=f'add extra runtime information, json formated')
    parser.add_argument('--strip_log_path',
                        metavar='REMOVE[,REPLACE]',
                        default='',
                        help='remove or replace parts of log path')

    args = parser.parse_args()

    logging.basicConfig(format="{file}: [ %(levelname)s ] %(message)s".format(
        file=os.path.basename(__file__)), level=logging.INFO, stream=sys.stdout)

    if args.output_dir:
        if os.path.exists(args.output_dir):
            logging.error(
                '%s already exists. Please specify a new directory for output logs',
                args.output_dir)
            sys.exit(1)
        os.makedirs(args.output_dir)
    returncode, _ = run([args.gtest_parallel] +
                        (['--output_dir', f'{args.output_dir}'] if args.output_dir else []) +
                        (['--workers', f'{args.workers}'] if args.workers else []) +
                        [args.binary] +
                        binary_args)

    if args.upload or args.timeline_report:
        # prepare memcheck records from logs
        logs = list(glob(os.path.join(args.output_dir, '**', '*.log'), recursive=True))
        strip = args.strip_log_path.split(',') + ['']
        records = create_memcheck_records(logs, strip[1], strip[0], append=json_load(args.metadata))
        logging.info('Prepared %d records', len(records))
        if len(records) != len(logs):
            logging.warning('Skipped %d logs of %d', len(logs) - len(records), len(logs))

        # upload
        if args.upload:
            if records:
                upload_memcheck_records(records, args.db_url, args.db_collection)
                logging.info('Uploaded to %s %s as %s', args.db_url, DATABASE, args.db_collection)
            else:
                logging.warning('No records to upload')

        # create timeline report
        if args.timeline_report:
            create_memcheck_report(records, args.db_url, args.db_collection, args.timeline_report)
            logging.info('Created memcheck report %s', args.timeline_report)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
