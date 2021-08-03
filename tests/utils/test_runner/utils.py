# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility module."""

from pathlib import Path
import sys

import yaml
from pymongo import MongoClient

# constants
DATABASES = ['timetests', 'memcheck']
DB_COLLECTIONS = ["commit", "nightly", "weekly"]
PRODUCT_NAME = 'dldt'   # product name from build manifest
TIMELINE_SIMILARITY = ('test_name', 'model', 'device', 'target_branch')


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


def query_memory_timeline(records, db_url, db_name, db_collection, max_items=20, similarity=TIMELINE_SIMILARITY):
    """ Query database for similar memory items committed previously
    """
    def timeline_key(item):
        """ Defines order for timeline report entries
        """
        if len(item['results']['vmhwm']) <= 1:
            return 1
        order = item['results']['vmhwm'][-1] - item['results']['vmhwm'][-2] + \
            item['results']['vmrss'][-1] - item['results']['vmrss'][-2]
        if not item['status']:
            # ensure failed cases are always on top
            order += sys.maxsize/2
        return order

    client = MongoClient(db_url)
    collection = client[db_name][db_collection]
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
        timeline['status'] = bool(timeline['results']['vmrss'][-1] < timeline['ref_results']['vmrss'][-1] and
                                  timeline['results']['vmhwm'][-1] < timeline['ref_results']['vmhwm'][-1])
        result += [timeline]

    result.sort(key=timeline_key, reverse=True)
    return result
