#!/usr/bin/env python3

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Common utilities for working with OSes or platforms
"""

import platform
import subprocess
import sys
from enum import Enum

import distro


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
