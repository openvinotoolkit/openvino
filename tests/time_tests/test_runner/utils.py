# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility module."""

import os
from pymongo import MongoClient

# constants
DATABASE = 'timetests'   # database name for timetests results
DB_COLLECTIONS = ["commit", "nightly", "weekly"]


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
