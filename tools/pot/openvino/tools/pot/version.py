# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

UNKNOWN_VERSION = "unknown version"


def generate_pot_version():
    """
    Function generates version like in cmake
    custom_{branch_name}_{commit_hash}
    """
    try:
        pot_dir = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=pot_dir)
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pot_dir)
        return "custom_{}_{}".format(branch_name.strip().decode(), commit_hash.strip().decode())
    except Exception: # pylint:disable=W0703
        return UNKNOWN_VERSION


def get_version():
    version = generate_pot_version()
    if version == UNKNOWN_VERSION:
        version_txt = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "version.txt")
        if os.path.isfile(version_txt):
            with open(version_txt) as f:
                version = f.readline().replace('\n', '')
    return version
