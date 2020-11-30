# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility module."""

import os
import platform
import sys
from enum import Enum
from pathlib import Path

import distro
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


class UnsupportedOsError(Exception):
    """
    Exception for unsupported OS type
    """

    def __init__(self, *args, **kwargs):
        error_message = f'OS type "{get_os_type()}" is not currently supported'
        if args or kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(error_message)


class OsType(Enum):
    """
    Container for supported os types
    """
    WINDOWS = 'Windows'
    LINUX = 'Linux'
    DARWIN = 'Darwin'


def get_os_type():
    """
    Get OS type

    :return: OS type
    :rtype: String | Exception if it is not supported
    """
    return platform.system()


def os_type_is_windows():
    """Returns True if OS type is Windows. Otherwise returns False"""
    return get_os_type() == OsType.WINDOWS.value


def os_type_is_linux():
    """Returns True if OS type is Linux. Otherwise returns False"""
    return get_os_type() == OsType.LINUX.value


def os_type_is_darwin():
    """Returns True if OS type is Darwin. Otherwise returns False"""
    return get_os_type() == OsType.DARWIN.value


def get_os_name():
    """
    Check OS type and return OS name

    :return: OS name
    :rtype: String | Exception if it is not supported
    """
    if os_type_is_linux():
        return distro.id().lower()
    if os_type_is_windows() or os_type_is_darwin():
        return get_os_type().lower()
    raise UnsupportedOsError()


def get_os_version():
    """
    Check OS version and return it

    :return: OS version
    :rtype: tuple | Exception if it is not supported
    """
    if os_type_is_linux():
        return distro.major_version(), distro.minor_version()
    if os_type_is_windows():
        return sys.getwindowsversion().major, sys.getwindowsversion().minor
    if os_type_is_darwin():
        return tuple(platform.mac_ver()[0].split(".")[:2])
    raise UnsupportedOsError()
