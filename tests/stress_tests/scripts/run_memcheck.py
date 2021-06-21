#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
from pathlib import Path

from memcheck_upload import create_memcheck_records, \
    upload_memcheck_records, \
    create_memcheck_report, \
    metadata_from_manifest, \
    info_from_test_config
from compare_memcheck_2_runs import compare_memcheck_2_runs, \
    get_memcheck_records, get_db_memcheck_records

# Database arguments
from memcheck_upload import DATABASE, DB_COLLECTIONS


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
            binary_args = sys.argv[idx+1:]
            sys.argv = sys.argv[:idx]
            break

    init_parser = argparse.ArgumentParser(add_help=False)
    init_parser.add_argument('--timeline_report',
                             help=f'create timeline HTML report file name')
    init_parser.add_argument('--upload', action="store_true",
                             help=f'upload results to database')
    init_parser.add_argument('--compare',
                             metavar='REFERENCE',
                             help='compare run with reference.'
                                  ' Should contain path to a folder with MemCheckTests logs or'
                                  ' query to request data from DB in "key=value[,key=value]" format')
    args = init_parser.parse_known_args()[0]

    parser = argparse.ArgumentParser(
        description='Run memcheck tests',
        usage='%(prog)s [options] binary -- [additional args]',
        parents=[init_parser])
    parser.add_argument('binary', help='test binary to execute')
    parser.add_argument('--gtest_parallel', help='path to gtest-parallel to use',
                        default='gtest_parallel')
    parser.add_argument('--timeout', help='timeout for tests run within gtest-parallel')
    parser.add_argument('-d', '--output_dir',
                        required=args.timeline_report or args.upload or args.compare,
                        help='output directory for test logs')
    parser.add_argument('-w', '--workers', help='number of gtest-parallel workers to spawn')

    parser.add_argument('--db_url',
                        required=args.timeline_report or args.upload or
                                 (args.compare and not os.path.isdir(args.compare)),
                        help='MongoDB URL in a form "mongodb://server:port"')
    parser.add_argument('--db_collection',
                        required=args.timeline_report or args.upload,
                        help=f'use collection name in {DATABASE} database',
                        choices=DB_COLLECTIONS)
    parser.add_argument('--manifest',
                        help=f'extract commit information from build manifest')
    parser.add_argument('--metadata',
                        help=f'add extra commit information, json formated')
    parser.add_argument('--strip_log_path',
                        metavar='REMOVE[,REPLACE]',
                        default='',
                        help='remove or replace parts of log path')

    parser.add_argument('--ref_db_collection',
                        required=args.compare and not os.path.isdir(args.compare),
                        help=f'use collection name in {DATABASE} database to query'
                             f' reference data',
                        choices=DB_COLLECTIONS)
    parser.add_argument('--comparison_report',
                        required=args.compare,
                        help='create comparison report file name')

    args = parser.parse_args()

    logging.basicConfig(format="{file} %(levelname)s: %(message)s".format(
        file=os.path.basename(__file__)), level=logging.INFO, stream=sys.stdout)

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        else:
            if list(glob(os.path.join(args.output_dir, '**', '*.log'), recursive=True)):
                logging.error(
                    'Output directory %s already has test logs.' \
                    'Please specify an empty directory for output logs',
                    args.output_dir)
                sys.exit(1)

    returncode, _ = run([sys.executable, args.gtest_parallel] +
                        (['--output_dir', f'{args.output_dir}'] if args.output_dir else []) +
                        (['--workers', f'{args.workers}'] if args.workers else []) +
                        (['--timeout', f'{args.timeout}'] if args.timeout else []) +
                        [args.binary] +
                        ['--'] + binary_args)

    if args.upload or args.timeline_report or args.compare:
        # prepare commit information
        append = {}
        if args.manifest:
            append.update(metadata_from_manifest(args.manifest))
        if args.metadata:
            append.update(json_load(args.metadata))

        # prepare memcheck records from logs
        logs = list(glob(os.path.join(args.output_dir, '**', '*.log'), recursive=True))
        strip = args.strip_log_path.split(',') + ['']
        records = create_memcheck_records(logs, strip[1], strip[0], append=append)
        logging.info('Prepared %d records', len(records))
        if len(records) != len(logs):
            logging.warning('Skipped %d logs of %d', len(logs) - len(records), len(logs))

        # extend memcheck records with info from test config
        test_conf_parser = argparse.ArgumentParser()
        test_conf_parser.add_argument('--test_conf')
        test_conf = test_conf_parser.parse_known_args(binary_args)[0].test_conf
        if test_conf:
            info = info_from_test_config(test_conf)
            for record in records:
                record.update(info.get(record["model_name"], {}))

        # upload
        if args.upload:
            if records:
                upload_memcheck_records(records, args.db_url, args.db_collection)
                logging.info('Uploaded to %s/%s.%s', args.db_url, DATABASE, args.db_collection)
            else:
                logging.warning('No records to upload')

        # create timeline report
        if args.timeline_report:
            create_memcheck_report(records, args.db_url, args.db_collection, args.timeline_report)
            logging.info('Created memcheck timeline report %s', args.timeline_report)

        # compare runs and prepare report
        if args.compare:
            if os.path.isdir(args.compare):
                references = get_memcheck_records(source=args.compare)
            else:
                query = dict(item.split("=") for item in args.compare.split(","))
                references = get_db_memcheck_records(query=query,
                                                     db_collection=args.ref_db_collection,
                                                     db_name=DATABASE, db_url=args.db_url)
            compare_retcode = compare_memcheck_2_runs(cur_values=records, references=references,
                                                      output_file=args.comparison_report)
            returncode = returncode if returncode else compare_retcode

    sys.exit(returncode)


if __name__ == "__main__":
    main()
