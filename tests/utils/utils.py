# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility module."""

from pathlib import Path

import yaml
from pymongo import MongoClient

# constants
DATABASES = ['timetests', 'memorytests']
DB_COLLECTIONS = ["commit", "nightly", "weekly"]
PRODUCT_NAME = 'dldt'   # product name from build manifest


def upload_data(data, db_url, db_name, db_collection):
    """ Upload timetest data to database."""
    client = MongoClient(db_url)
    collection = client[db_name][db_collection]
    collection.replace_one({'_id': data['_id']}, data, upsert=True)


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
