#!/usr/bin/env python3

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Upload metrics gathered by MemCheckTests into Mongo DB
Usage: ./scrips/memcheck_upload.py https://ci.intel.com/job/memchek/1234/ \
    ./gtest-parallel-logs/**/*.log \
    --artifact_root ./gtest-parallel-logs --dryrun
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import defusedxml.ElementTree as ET
from glob import glob
from inspect import getsourcefile
from types import SimpleNamespace
import requests
from copy import deepcopy

import yaml
from pymongo import MongoClient

# Database arguments
DATABASE = 'memcheck'  # database name for memcheck results
DB_COLLECTIONS = ["commit", "nightly", "weekly"]

PRODUCT_NAME = 'dldt'  # product name from build manifest
RE_GTEST_MODEL_XML = re.compile(r'<model[^>]*>')
RE_GTEST_CUR_MEASURE = re.compile(r'\[\s*MEASURE\s*\]')
RE_GTEST_REF_MEASURE = re.compile(
    r'Reference values of virtual memory consumption')
RE_GTEST_PASSED = re.compile(r'\[\s*PASSED\s*\]')
RE_GTEST_FAILED = re.compile(r'\[\s*FAILED\s*\]')
GTEST_INFO = '[ INFO ]'
PRECISSIONS = ('FP32', 'FP16', 'INT8')
KEY_FIELDS = ('test_name', 'model', 'device', 'build_url')


def abs_path(relative_path):
    """Return absolute path given path relative to the current file.
    """
    return os.path.realpath(
        os.path.join(os.path.dirname(getsourcefile(lambda: 0)), relative_path))


def metadata_from_manifest(manifest):
    """ Extract commit metadata for memcheck record from manifest
    """
    with open(manifest, 'r') as manifest_file:
        manifest = yaml.safe_load(manifest_file)
    repo_trigger = next(
        repo for repo in manifest['components'][PRODUCT_NAME]['repository'] if repo['trigger'])
    # parse OS name/version
    product_type_str = manifest['components'][PRODUCT_NAME]['product_type']
    product_type = product_type_str.split('_')
    if len(product_type) != 5 or product_type[2] != 'ubuntu':
        logging.error('Product type %s is not supported', product_type_str)
        return {}
    return {
        'os_name': product_type[2],
        'os_version': [product_type[3], product_type[4]],
        'commit_sha': repo_trigger['revision'],
        'commit_date': repo_trigger['commit_time'],
        'repo_url': repo_trigger['url'],
        'branch': repo_trigger['branch'],
        'target_branch': repo_trigger['target_branch'] if repo_trigger["target_branch"] else repo_trigger["branch"],
        'event_type': manifest['components'][PRODUCT_NAME]['build_event'].lower(),
        f'{PRODUCT_NAME}_version': manifest['components'][PRODUCT_NAME]['version'],
    }


def info_from_test_config(test_conf):
    """ Extract models information for memcheck record from test config
    """
    test_conf_obj = ET.parse(test_conf)
    test_conf_root = test_conf_obj.getroot()
    records = {}
    for model_rec in test_conf_root.find("models"):
        model_name = model_rec.attrib["name"]
        records[model_name] = {
            "framework": model_rec.attrib.get("framework"),
            "source": model_rec.attrib.get("source"),
        }
    return records


def parse_memcheck_log(log_path):
    """ Parse memcheck log
    """
    try:
        with open(log_path, 'r') as log_file:
            log = log_file.read()
    except FileNotFoundError:
        # Skip read of broken files
        return None

    passed_match = RE_GTEST_PASSED.search(log)
    failed_match = RE_GTEST_FAILED.search(log)
    model_match = RE_GTEST_MODEL_XML.search(log)
    if not model_match:
        return None
    model = ET.fromstring(model_match.group(0)).attrib

    log_lines = log.splitlines()
    for index, line in enumerate(log_lines):
        if RE_GTEST_REF_MEASURE.search(line):
            heading = [name.lower() for name in log_lines[index + 1]
            [len(GTEST_INFO):].split()]
            values = [int(val) for val in log_lines[index + 2]
            [len(GTEST_INFO):].split()]
            ref_metrics = dict(zip(heading, values))
    for index in reversed(range(len(log_lines))):
        if RE_GTEST_CUR_MEASURE.search(log_lines[index]):
            test_name = log_lines[index].split()[-1]
            heading = [name.lower() for name in log_lines[index + 1]
            [len(GTEST_INFO):].split()]
            values = [int(val) for val in log_lines[index + 2]
            [len(GTEST_INFO):].split()]
            entry = SimpleNamespace(
                metrics=dict(zip(heading, values)),
                test_name=test_name,
                model_name=os.path.splitext(os.path.basename(model['path']))[0],
                precision=next(pr for pr in PRECISSIONS if pr.upper() in model['precision'].upper()),
                model=model['path'],
                device=model['device'].upper(),
                status='passed' if passed_match else 'failed' if failed_match else 'started'
            )
            if ref_metrics:
                entry.ref_metrics = ref_metrics
            return vars(entry)
    return None


def create_memcheck_records(logs, build_url, artifact_root, append=None):
    """ Parse memcheck logs and create records for MongoDB
    """
    records = []
    for log in logs:
        data = parse_memcheck_log(log)
        if not data:
            continue
        data['build_url'] = build_url
        data['log_path'] = os.path.relpath(log, artifact_root)
        if append:
            data.update(append)

        data['_id'] = hashlib.sha256(
            ''.join([str(data[key]) for key in KEY_FIELDS]).encode()).hexdigest()
        records += [data]
    return records


def upload_memcheck_records(records, db_url, db_collection):
    """ Upload records created by create_memcheck_records
    """
    client = MongoClient(db_url)
    collection = client[DATABASE][db_collection]
    for record in records:
        collection.replace_one({'_id': record['_id']}, record, upsert=True)


def modify_data_for_push_to_new_db(records):
    new_records = deepcopy(records)
    records_to_push = []

    for record in new_records:
        if '_id' in record:
            del record['_id']
        if 'os_name' in record and 'os_version' in record:
            record['os'] = '{} {}.{}'.format(record['os_name'], record['os_version'][0], record['os_version'][1])
            del record['os_name']
            del record['os_version']
        if 'repo_url' in record:
            del record['repo_url']
        if 'commit_sha' in record:
            del record['commit_sha']
        if 'event_type' in record:
            del record['event_type']
        if 'framework' in record:
            record['framework'] = str(record['framework'])
        try:
            with open(record['log_path'], 'r') as log_file:
                log = log_file.read()
        except FileNotFoundError:
            log = ''
        record['log'] = log
        record['ext'] = {}
        records_to_push.append({'data': [record]})

    return records_to_push


def push_to_db_facade(records, db_api_handler):
    headers = {"Content-Type": "application/json", "accept": "application/json"}
    uploaded = False
    errors = []
    for record in records:
        try:
            response = requests.post(db_api_handler, json=record, headers=headers)
            if response.ok:
                uploaded = True
            else:
                errors.append(str(response.json()))
        except Exception as e:
            errors.append(e)

    if uploaded and not errors:
        logging.info("Uploaded records by API url {}".format(db_api_handler))
    else:
        logging.info("Failed to upload records by API url {} due to errors {}".format(db_api_handler, errors))



def _transpose_dicts(items, template=None):
    """ Build dictionary of arrays from array of dictionaries
    Example:
    > in = [{'a':1, 'b':3}, {'a':2}]
    > _transpose_dicts(in, template=in[0])
    {'a':[1,2], 'b':[3, None]}
    """
    result = {}
    if not items:
        return result
    if not template:
        template = items[0]
    for key, template_val in template.items():
        if isinstance(template_val, dict):
            result[key] = _transpose_dicts(
                [item[key] for item in items if key in item], template_val)
        else:
            result[key] = [item.get(key, None) for item in items]
    return result


TIMELINE_SIMILARITY = ('test_name', 'model', 'device', 'target_branch')


def query_timeline(records, db_url, db_collection, max_items=20, similarity=TIMELINE_SIMILARITY):
    """ Query database for similar memcheck items committed previously
    """

    def timeline_key(item):
        """ Defines order for timeline report entries
        """
        if len(item['metrics']['vmhwm']) <= 1:
            return 1
        order = item['metrics']['vmhwm'][-1] - item['metrics']['vmhwm'][-2] + \
                item['metrics']['vmrss'][-1] - item['metrics']['vmrss'][-2]
        if not item['status']:
            # ensure failed cases are always on top
            order += sys.maxsize / 2
        return order

    client = MongoClient(db_url)
    collection = client[DATABASE][db_collection]
    result = []
    for record in records:
        items = []
        try:
            query = dict((key, record[key]) for key in similarity)
            query['commit_date'] = {'$lt': record['commit_date']}
            pipeline = [
                {'$match': query},
                {'$addFields': {
                    'commit_date': {'$dateFromString': {'dateString': '$commit_date'}}}},
                {'$sort': {'commit_date': -1}},
                {'$limit': max_items},
                {'$sort': {'commit_date': 1}},
            ]
            items += list(collection.aggregate(pipeline))
        except KeyError:
            pass  # keep only the record if timeline failed to generate
        items += [record]
        timeline = _transpose_dicts(items, template=record)
        timeline['status'] = bool(timeline['metrics']['vmrss'][-1] < timeline['ref_metrics']['vmrss'][-1] and
                                  timeline['metrics']['vmhwm'][-1] < timeline['ref_metrics']['vmhwm'][-1])
        result += [timeline]

    result.sort(key=timeline_key, reverse=True)
    return result


def create_memcheck_report(records, db_url, db_collection, output_path):
    """ Create memcheck timeline HTML report for records.
    """
    records.sort(
        key=lambda item: f"{item['status']}{item['device']}{item['model_name']}{item['test_name']}")
    timelines = query_timeline(records, db_url, db_collection)
    import jinja2  # pylint: disable=import-outside-toplevel
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            searchpath=os.path.join(abs_path('.'), 'memcheck-template')),
        autoescape=False)
    template = env.get_template('timeline_report.html')
    template.stream(records=records, timelines=timelines).dump(output_path)


def globber(paths):
    """Generator extending paths with wildcards"""
    for path in paths:
        if any(magic in path for magic in ['*', '?', '!', '[', ']']):
            for resolved in glob(path, recursive=True):
                yield resolved
        else:
            yield path


def main():
    """Main entry point.
    """
    parser = argparse.ArgumentParser(
        description='Upload metrics gathered by memcheck into Mongo DB',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dryrun', action="store_true",
                        help='Parse logs, not modify database.')
    is_dryrun = parser.parse_known_args()[0].dryrun
    parser.add_argument('build_url', help='A place where memcheck execution logs can be found.')
    parser.add_argument('log', nargs='+', help='Local path to log. Extended wildcards supported.')
    parser.add_argument('--db_url', required=not is_dryrun,
                        help='MongoDB URL in a for "mongodb://server:port".')
    parser.add_argument('--db_collection', required=not is_dryrun,
                        help=f'Collection name in {DATABASE} database to upload.',
                        choices=DB_COLLECTIONS)
    parser.add_argument('--db_api_handler',
                        help='API handler url for push data to database',
                        default='',
                        )
    parser.add_argument('--artifact_root', required=True,
                        help=f'A root directory to strip from log path before upload.')
    parser.add_argument('--append', help='JSON to append to each item.')
    args = parser.parse_args()

    logging.basicConfig(format="{file}: [ %(levelname)s ] %(message)s".format(
        file=os.path.basename(__file__)), level=logging.INFO, stream=sys.stdout)

    if args.append:
        with open(args.append, 'r') as append_file:
            append = json.load(append_file)
    else:
        append = None

    logs = list(globber(args.log))
    records = create_memcheck_records(
        logs, args.build_url, args.artifact_root, append=append)
    logging.info('Prepared %d records', len(records))
    if len(records) != len(logs):
        logging.warning(
            'Skipped %d logs of %d', len(logs) - len(records), len(logs))
    if not args.dryrun:
        upload_memcheck_records(records, args.db_url, args.db_collection)
        logging.info('Uploaded to %s', args.db_url)
        if args.db_api_handler:
            new_format_records = modify_data_for_push_to_new_db(records)
            push_to_db_facade(new_format_records, args.db_api_handler)
    else:
        print(json.dumps(records, sort_keys=True, indent=4))


if __name__ == "__main__":
    main()
