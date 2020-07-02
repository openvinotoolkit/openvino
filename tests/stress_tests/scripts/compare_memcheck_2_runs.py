#!/usr/bin/env python3
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
Create comparison table based on MemCheckTests results from 2 runs
Usage: ./scrips/compare_memcheck_2_runs.py cur_source ref_source \
       --db_collection collection_name --out_file file_name
"""
# pylint:disable=line-too-long

import argparse
import csv
import json
import os
from collections import OrderedDict
from glob import glob
from operator import itemgetter
from pathlib import Path

from memcheck_upload import create_memcheck_records
from pymongo import MongoClient

# Database arguments
DATABASE = 'memcheck'


def get_db_memcheck_records(query, db_collection, db_name, db_url):
    """Request MemCheckTests records from database by provided query"""
    client = MongoClient(db_url)
    collection = client[db_name][db_collection]
    items = list(collection.find(query))
    return items


def get_memcheck_records(source, db_collection=None, db_name=None, db_url=None):
    """provide MemCheckTests records"""
    if os.path.isdir(source):
        logs = list(glob(os.path.join(source, '**', '*.log'), recursive=True))
        items = create_memcheck_records(logs, build_url=None, artifact_root=source)
    else:
        assert db_collection and db_name and db_url
        query = json.loads(source)
        items = get_db_memcheck_records(query, db_collection, db_name, db_url)

    return items


def prepare_comparison_table_csv(data, data_metrics, output_file):
    """generate .csv file with table based on provided data"""
    fields = list(data[0].keys())
    metrics_names = list(data_metrics[0].keys())
    HEADERS = fields + metrics_names
    with open(output_file, 'w', newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(HEADERS)
        for i in range(len(data)):
            row = []
            for field in fields:
                row.append(data[i][field])
            for metric_name in metrics_names:
                row.append(data_metrics[i][metric_name])
            csvwriter.writerow(row)


def compare_memcheck_2_runs(cur_values, references, output_file=None):
    """Compares 2 MemCheckTests runs and prepares a report on specified path"""

    # Fields should be presented in both `references` and `cur_values`.
    # Some of metrics may be missing for one of `references` and `cur_values`.
    # Report will contain data with order defined in `required_fields` and `required_metrics`
    required_fields = [
        # "metrics" should be excluded because it will be handled automatically
        "model", "device", "test_name"
    ]
    required_metrics = [
        "vmrss", "vmhwm",
        # "vmsize", "vmpeak"    # temporarily disabled as unused
    ]
    # `Ops` is a template applied for every metric defined in `required_metrics`
    ops = OrderedDict([
        # x means ref, y means cur
        ("ref", lambda x, y: x),
        ("cur", lambda x, y: y),
        ("cur-ref", lambda x, y: y - x if (x is not None and y is not None) else None),
        ("ref/cur", lambda x, y: x / y if (x is not None and y is not None) else None)
    ])

    filtered_refs = []
    filtered_refs_metrics = []
    for record in references:
        filtered_rec = {key: val for key, val in record.items() if key in required_fields}
        filtered_rec_metrics = {key: val for key, val in record["metrics"].items() if key in required_metrics}
        filtered_refs.append(filtered_rec)
        filtered_refs_metrics.append(filtered_rec_metrics)
    assert len(filtered_refs) == len(filtered_refs_metrics), \
        "Filtered references and metrics should contain equal number of records. " \
        "References len: {}, metrics len: {}".format(len(filtered_refs), len(filtered_refs_metrics))

    filtered_cur_val = []
    filtered_cur_val_metrics = []
    for record in cur_values:
        filtered_rec = {key: val for key, val in record.items() if key in required_fields}
        filtered_rec_metrics = {key: val for key, val in record["metrics"].items() if key in required_metrics}
        filtered_cur_val.append(filtered_rec)
        filtered_cur_val_metrics.append(filtered_rec_metrics)
    assert len(filtered_cur_val) == len(filtered_cur_val_metrics), \
        "Filtered current values and metrics should contain equal number of records. " \
        "Current values len: {}, metrics len: {}".format(len(filtered_refs), len(filtered_refs_metrics))

    comparison_data = []
    for data in [filtered_refs, filtered_cur_val]:
        for record in data:
            rec = OrderedDict()
            for field in required_fields:
                rec.update({field: record[field]})
                rec.move_to_end(field)
            if rec not in comparison_data:
                # Comparison data should contain unique records combined from references and current values
                comparison_data.append(rec)
    comparison_data = sorted(comparison_data, key=itemgetter("model"))

    comparison_data_metrics = []
    for record in comparison_data:
        try:
            i = filtered_refs.index(record)
        except ValueError:
            i = -1

        try:
            j = filtered_cur_val.index(record)
        except ValueError:
            j = -1

        metrics_rec = OrderedDict()
        for metric in required_metrics:
            ref = filtered_refs_metrics[i][metric] if i != -1 and metric in filtered_refs_metrics[i] else None
            cur = filtered_cur_val_metrics[j][metric] if j != -1 and metric in filtered_cur_val_metrics[j] else None
            for op_name, op in ops.items():
                op_res = op(ref, cur)
                metric_name = "{} {}".format(metric, op_name)
                metrics_rec.update({metric_name: op_res})
                metrics_rec.move_to_end(metric_name)

        comparison_data_metrics.append(metrics_rec)

    assert len(comparison_data) == len(comparison_data_metrics), \
        "Data and metrics for comparison should contain equal number of records. Data len: {}, metrics len: {}" \
            .format(len(comparison_data), len(comparison_data_metrics))

    if output_file:
        prepare_comparison_table_csv(comparison_data, comparison_data_metrics, output_file)


def cli_parser():
    """parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Tool to create a table with comparison '
                                                 'of 2 runs of MemCheckTests')
    parser.add_argument('cur_source',
                        help='Source of current values of MemCheckTests. '
                             'Should contain path to a folder with logs or '
                             'JSON-format query to request data from DB.')
    parser.add_argument('ref_source',
                        help='Source of reference values of MemCheckTests. '
                             'Should contain path to a folder with logs or '
                             'JSON-format query to request data from DB.')
    parser.add_argument('--db_url',
                        help='MongoDB URL in a for "mongodb://server:port".')
    parser.add_argument('--db_collection',
                        help=f'Collection name in "{DATABASE}" database to query'
                             f' data using current source.',
                        choices=["commit", "nightly", "weekly"])
    parser.add_argument('--ref_db_collection',
                        help=f'Collection name in "{DATABASE}" database to query'
                             f' data using reference source.',
                        choices=["commit", "nightly", "weekly"])
    parser.add_argument('--out_file', dest='output_file', type=Path,
                        help='Path to a file (with name) to save results. '
                             'Example: /home/.../file.csv')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = cli_parser()
    references = get_memcheck_records(args.ref_source, args.ref_db_collection, DATABASE, args.db_url)
    cur_values = get_memcheck_records(args.cur_source, args.db_collection, DATABASE, args.db_url)
    compare_memcheck_2_runs(cur_values, references, output_file=args.output_file)
