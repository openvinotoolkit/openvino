# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os


def get_list(key_name, delimiter=',', fallback=None):
    value = os.environ.get(key_name, fallback)
    if value != fallback:
        value = value.split(delimiter)
    elif not value:
        value = []
    return value


def get_bool(key_name, fallback=None):
    value = os.environ.get(key_name, fallback)
    if value != fallback:
        value = value.lower()
        if value == "true":
            value = True
        elif value == "false":
            value = False
        else:
            raise ValueError("Value of {} env variable is '{}'. Should be 'True' or 'False'.".format(key_name, value))
    return value


def get_int(key_name, fallback=None):
    value = os.environ.get(key_name, fallback)
    if value != fallback:
        try:
            value = int(value)
        except ValueError:
            raise ValueError("Value '{}' of {} env variable cannot be cast to int.".format(value, key_name))
    return value


def get_path(key_name, fallback=None):
    value = os.environ.get(key_name, fallback)
    if value:
        value = os.path.expanduser(value)
        value = os.path.realpath(value)
    return value
