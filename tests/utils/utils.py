# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility module."""

from pathlib import Path

import yaml
from pymongo import MongoClient
from copy import deepcopy
import requests
import logging
import re

# constants
DATABASES = ['timetests', 'memorytests']
DB_COLLECTIONS = ["commit", "nightly", "weekly"]
PRODUCT_NAME = 'dldt'   # product name from build manifest


def upload_data(data, db_url, db_name, db_collection):
    """ Upload timetest data to database."""
    client = MongoClient(db_url)
    collection = client[db_name][db_collection]
    collection.replace_one({'_id': data['_id']}, data, upsert=True)


def push_to_db_facade(data, db_api_handler):
    headers = {"Content-Type": "application/json", "accept": "application/json"}
    response = requests.post(db_api_handler, json=data, headers=headers)
    if response.ok:
        logging.info("Uploaded records by API url {}".format(db_api_handler))
    else:
        raise ConnectionError("Failed to upload records by API url {} due to error {}".format(db_api_handler,
                                                                                              str(response.json())))


def modify_data_for_push_to_new_db(data):
    new_data = deepcopy(data)

    if '_id' in new_data:
        del new_data['_id']
    if 'run_id' in new_data:
        del new_data['run_id']
        new_data['build_url'] = data['run_id']
    if 'os' in new_data:
        platform, os_version_major, os_version_minor = data['os'].split("_")
        new_data['os'] = "{} {}.{}".format(platform, os_version_major, os_version_minor)
    if 'model' in new_data:
        new_data['model_name'] = data['model']['name']
        new_data['model'] = data['model']['path']
        new_data['precision'] = data['model']['precision']
        new_data['framework'] = data['model']['framework']
    if 'device' in new_data:
        new_data['device'] = data['device']['name']
    if 'test_name' in new_data:
        del new_data['test_name']
    if 'commit_sha' in new_data:
        del new_data['commit_sha']
    if 'repo_url' in new_data:
        del new_data['repo_url']
    if 'product_type' in new_data:
        del new_data['product_type']
    if 'version' in new_data:
        del new_data['version']
        new_data['dldt_version'] = re.findall(r"\d{4}\.\d+.\d-\d+-\w+", data['version'])[0]
    if 'raw_results' in new_data:
        del new_data['raw_results']
        for raw_result_name, raw_result in data['raw_results'].items():
            new_data['results'][raw_result_name]['raw_results'] = raw_result
    new_data['ext'] = {}
    new_data = {'data': [new_data]}
    return new_data


def metadata_from_manifest(manifest: Path):
    """ Extract commit metadata from manifest."""
    with open(manifest, 'r') as manifest_file:
        manifest = yaml.safe_load(manifest_file)
    repo_trigger = next(
        repo for repo in manifest['components'][PRODUCT_NAME]['repository'] if repo['trigger'])
    return {
        'product_type': manifest['components'][PRODUCT_NAME]['product_type'],
        'commit_sha': repo_trigger['revision'],
        'commit_date': repo_trigger['commit_time'],
        'repo_url': repo_trigger['url'],
        'branch': repo_trigger['branch'],
        'target_branch': repo_trigger['target_branch'] if repo_trigger["target_branch"] else repo_trigger["branch"],
        'version': manifest['components'][PRODUCT_NAME]['version']
    }
