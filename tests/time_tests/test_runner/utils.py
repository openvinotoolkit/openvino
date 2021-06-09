# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility module."""

import os
import platform
import subprocess
import sys
import distro
import yaml
import numpy as np

from enum import Enum
from pathlib import Path
from pymongo import MongoClient

# constants
DATABASE = 'timetests'   # database name for timetests results
DB_COLLECTIONS = ["commit", "nightly", "weekly"]
PRODUCT_NAME = 'dldt'   # product name from build manifest

# Define a range to cut outliers which are < Q1 − IQR_CUTOFF * IQR, and > Q3 + IQR_CUTOFF * IQR
# https://en.wikipedia.org/wiki/Interquartile_range
IQR_CUTOFF = 1.5


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
    """ Upload timetest data to database."""
    client = MongoClient(db_url)
    collection = client[DATABASE][db_collection]
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
        'target_branch': repo_trigger['branch'],
        'version': manifest['components'][PRODUCT_NAME]['version']
    }


def calculate_iqr(stats: list):
    """IQR is calculated as the difference between the 3th and the 1th quantile of the data."""
    q1 = np.quantile(stats, 0.25)
    q3 = np.quantile(stats, 0.75)
    iqr = q3 - q1
    return iqr, q1, q3


def filter_timetest_result(stats: dict):
    """Identify and remove outliers from time_results."""
    filtered_stats = {}
    for step_name, time_results in stats.items():
        iqr, q1, q3 = calculate_iqr(time_results)
        cut_off = iqr * IQR_CUTOFF
        upd_time_results = [x for x in time_results if (q1 - cut_off < x < q3 + cut_off)]
        filtered_stats.update({step_name: upd_time_results if upd_time_results else time_results})
    return filtered_stats


class UnsupportedOsError(Exception):
    """Exception for unsupported OS type."""
    def __init__(self, *args, **kwargs):
        error_message = f'OS type "{get_os_type()}" is not currently supported'
        if args or kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(error_message)


class OsType(Enum):
    """Container for supported os types."""
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
    """Returns True if OS type is Windows. Otherwise returns False."""
    return get_os_type() == OsType.WINDOWS.value


def os_type_is_linux():
    """Returns True if OS type is Linux. Otherwise returns False."""
    return get_os_type() == OsType.LINUX.value


def os_type_is_darwin():
    """Returns True if OS type is Darwin. Otherwise returns False."""
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


def get_cpu_info():
    """
    Check OS version and returns name and frequency of cpu

    :return: CPU name and frequency
    :rtype: str
    """
    model = ''
    if os_type_is_linux():
        command = r"lscpu | sed -n 's/Model name:[ \t]*//p'"
        model = subprocess.check_output(command, shell=True)
    elif os_type_is_windows():
        command = 'wmic cpu get name | find /v "Name"'
        model = subprocess.check_output(command, shell=True)
    elif os_type_is_darwin():
        command = ['/usr/sbin/sysctl', "-n", "machdep.cpu.brand_string"]
        model = subprocess.check_output(command)
    else:
        raise UnsupportedOsError()
    info = model.decode('utf-8').strip()
    return info
