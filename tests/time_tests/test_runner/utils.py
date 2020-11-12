# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility module."""

import os
from pathlib import Path
import yaml
from pymongo import MongoClient

# constants
DATABASE = 'timetests'   # database name for timetests results
DB_COLLECTIONS = ["commit", "nightly", "weekly"]
PRODUCT_NAME = 'dldt'   # product name from build manifest


def expand_env_vars(obj):
    """Expand environment variables in provided object."""

    if isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = expand_env_vars(value)
    elif isinstance(obj, dict):
        for name, value in obj.items():
            obj[name] = expand_env_vars(value)
    else:
        obj = os.path.expandvars(obj)
    return obj


def upload_timetest_data(data, db_url, db_collection):
    """ Upload timetest data to database
    """
    client = MongoClient(db_url)
    collection = client[DATABASE][db_collection]
    collection.replace_one({'_id': data['_id']}, data, upsert=True)


def metadata_from_manifest(manifest: Path):
    """ Extract commit metadata from manifest
    """
    with open(manifest, 'r') as manifest_file:
        manifest = yaml.safe_load(manifest_file)
    repo_trigger = next(
        repo for repo in manifest['components'][PRODUCT_NAME]['repository'] if repo['trigger'])
    return {
        'product_type': manifest['components'][PRODUCT_NAME]['product_type'],
        'commit_sha': repo_trigger['revision'],
        'commit_date': repo_trigger['commit_time'],
        'repo_url': repo_trigger['url'],
        'target_branch': repo_trigger['target_branch'],
        'version': manifest['components'][PRODUCT_NAME]['version']
    }
