# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import tempfile


def create_tmp_dir(parent_dir=tempfile.gettempdir()):
    """ Creates temporary directory with unique name and auto cleanup
    :param parent_dir: directory in which temporary directory is created
    :return: TemporaryDirectory object
    """
    parent_dir = tempfile.TemporaryDirectory(dir=parent_dir)
    if not os.path.exists(parent_dir.name):
        try:
            os.makedirs(parent_dir.name)
        except PermissionError as e:
            raise type(e)(
                'Failed to create directory {}. Permission denied. '.format(parent_dir))
    return parent_dir


def convert_output_key(name):
    """ Convert output name into IE-like name
    :param name: output name to convert
    :return: IE-like output name
    """
    if not isinstance(name, tuple):
        return name
    if len(name) != 2:
        raise Exception('stats name should be a string name or 2 elements tuple '
                        'with string as the first item and port number the second')
    return '{}.{}'.format(*name)


class Environment:
    def __init__(self, variable, value):
        self.variable = variable
        self.value = value
        self.was_set = (self.variable in os.environ)

    def __enter__(self):
        os.environ[self.variable] = self.value

    def __exit__(self, *args):
        if not self.was_set:
            del os.environ[self.variable]


def generate_pot_version():
    """
    Function generates version like in cmake
    custom_{branch_name}_{commit_hash}
    """
    try:
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
        return "custom_{}_{}".format(branch_name, commit_hash)
    except Exception as e:
        return "unknown version"


def get_version():
    version = generate_pot_version()
    if version == "unknown version":
        version_txt = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "version.txt")
        if os.path.isfile(version_txt):
            with open(version_txt) as f:
                version = f.readline().replace('\n', '')
    return version
